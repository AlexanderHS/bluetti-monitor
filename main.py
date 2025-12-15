import os
import shutil
import requests
from urllib.parse import urljoin
from fastapi import FastAPI, Query, Form
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import uvicorn
import cv2
import numpy as np
import pytesseract
from io import BytesIO
from fastapi.responses import Response, HTMLResponse, FileResponse
import aiosqlite
from datetime import datetime
from typing import List, Dict
import json
import logging
from pathlib import Path
from recommendations import analyze_recent_readings_for_recommendations
from switchbot_controller import switchbot_controller
from device_discovery import device_discovery
from template_classifier import template_classifier, get_training_status, extract_timestamp_from_filename
from comparison_storage import comparison_storage

load_dotenv()

# Configure logging with ISO 8601 timestamps
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add a global variable to track startup time
startup_time = None


# Database management
class BatteryDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self):
        """Initialize database with required tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Check if we need to migrate existing database
            await self._migrate_database_if_needed(db)
            
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS battery_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    battery_percentage INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    ocr_method TEXT,
                    total_attempts INTEGER,
                    raw_vote_data TEXT,
                    raw_battery_percentage INTEGER,
                    median_filtered BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp ON battery_readings(timestamp)
            """
            )
            await db.commit()

    async def _migrate_database_if_needed(self, db):
        """Migrate existing database to add new columns"""
        try:
            # Check if raw_battery_percentage column exists
            async with db.execute("PRAGMA table_info(battery_readings)") as cursor:
                columns = await cursor.fetchall()
                column_names = [col[1] for col in columns]
                
                if 'raw_battery_percentage' not in column_names:
                    logger.info("Migrating database: adding median filter columns")
                    await db.execute("ALTER TABLE battery_readings ADD COLUMN raw_battery_percentage INTEGER")
                    await db.execute("ALTER TABLE battery_readings ADD COLUMN median_filtered BOOLEAN DEFAULT FALSE")
                    
                    # Update existing records to set raw_battery_percentage = battery_percentage
                    await db.execute("UPDATE battery_readings SET raw_battery_percentage = battery_percentage WHERE raw_battery_percentage IS NULL")
                    await db.commit()
                    logger.info("Database migration completed")
                    
        except Exception as e:
            logger.warning(f"Database migration failed: {e}")

    async def calculate_median_filtered_reading(self, current_reading: int, confidence: float) -> int:
        """Calculate median of last 3 readings to smooth out false zeros"""
        try:
            # Get last 3 readings
            recent_readings = await self.get_recent_readings(limit=3)
            
            # If we don't have enough readings, return current reading
            if len(recent_readings) < 2:
                return current_reading
            
            # Extract battery percentages from recent readings
            readings = [r["battery_percentage"] for r in recent_readings]
            readings.append(current_reading)  # Add current reading
            
            # Calculate median
            readings.sort()
            median_reading = readings[len(readings) // 2]
            
            logger.info(f"Median filter: readings={readings} -> median={median_reading} (current={current_reading})")
            return median_reading
            
        except Exception as e:
            logger.warning(f"Median filter failed: {e}, using current reading")
            return current_reading

    async def insert_reading(
        self,
        battery_percentage: int,
        confidence: float,
        ocr_method: str = None,
        total_attempts: int = None,
        raw_vote_data: dict = None,
    ):
        """Insert a new battery reading with median filtering"""
        # Apply median filtering for readings with sufficient confidence
        filtered_percentage = battery_percentage
        is_filtered = False
        
        if confidence >= 0.8:  # Only apply median filter to high-confidence readings
            if confidence >= 1.0:  # Perfect confidence - trust immediately
                filtered_percentage = battery_percentage
                is_filtered = False
            else:
                filtered_percentage = await self.calculate_median_filtered_reading(battery_percentage, confidence)
                is_filtered = (filtered_percentage != battery_percentage)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO battery_readings 
                (timestamp, battery_percentage, confidence, ocr_method, total_attempts, raw_vote_data, raw_battery_percentage, median_filtered)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().timestamp(),
                    filtered_percentage,  # Store the filtered value as the main reading
                    confidence,
                    ocr_method,
                    total_attempts,
                    json.dumps(raw_vote_data) if raw_vote_data else None,
                    battery_percentage,  # Store original raw value
                    is_filtered
                ),
            )
            await db.commit()
            
            filter_msg = f" (filtered: {battery_percentage}% -> {filtered_percentage}%)" if is_filtered else ""
            logger.info(
                f"Stored battery reading: {filtered_percentage}% (confidence: {confidence}){filter_msg} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

    async def get_recent_readings(self, limit: int = 10) -> List[Dict]:
        """Get recent battery readings"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT timestamp, battery_percentage, confidence, ocr_method, total_attempts
                FROM battery_readings 
                ORDER BY timestamp DESC 
                LIMIT ?
            """,
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "timestamp": row[0],
                        "battery_percentage": row[1],
                        "confidence": row[2],
                        "ocr_method": row[3],
                        "total_attempts": row[4],
                        "seconds_ago": int(datetime.now().timestamp() - row[0]),
                    }
                    for row in rows
                ]


# Global database instance
db = BatteryDatabase(os.getenv("DATABASE_PATH", "./data/battery_readings.db"))


def get_crop_coordinates():
    """Get battery crop coordinates from environment"""
    return {
        "x1": int(os.getenv("BATTERY_CROP_X1", 400)),
        "y1": int(os.getenv("BATTERY_CROP_Y1", 200)),
        "x2": int(os.getenv("BATTERY_CROP_X2", 500)),
        "y2": int(os.getenv("BATTERY_CROP_Y2", 250)),
    }


def capture_image():
    """Helper function to capture image from webcam"""
    webcam_url = os.getenv("WEBCAM_URL")
    capture_endpoint = os.getenv("WEBCAM_CAPTURE_ENDPOINT", "/capture")

    if not webcam_url:
        raise Exception("WEBCAM_URL not configured")

    capture_url = urljoin(webcam_url, capture_endpoint)

    # Retry logic for network issues including "premature end of data segment"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(capture_url, timeout=10)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            error_msg = str(e).lower()
            # Check for specific recoverable errors
            if (
                "premature end of data segment" in error_msg
                or "connection" in error_msg
                or "timeout" in error_msg
            ):
                if attempt < max_retries - 1:
                    logger.debug(
                        f"Recoverable network error on attempt {attempt + 1}: {e}"
                    )
                    import time

                    time.sleep(1)  # Wait a bit longer for network issues
                    continue

            if attempt == max_retries - 1:
                raise e
            import time

            time.sleep(0.5)

    raise Exception(f"Failed to capture after {max_retries} attempts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database - worker handles background polling"""
    global startup_time
    startup_time = datetime.now().timestamp()

    logger.info("Starting up Bluetti Monitor API")

    # Initialize database
    await db.init_db()
    logger.info("Database initialized")

    # Initialize comparison storage
    await comparison_storage.init_db()
    logger.info("Comparison storage initialized")

    logger.info("Background polling is handled by separate worker service")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down API service...")


app = FastAPI(
    title=os.getenv("API_TITLE", "Bluetti Monitor API"),
    description="Monitor Bluetti solar generator battery status via ESP32 webcam",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint that verifies the monitoring system is working"""
    try:
        # Get the most recent reading
        recent_readings = await db.get_recent_readings(limit=1)

        # Check if we have any readings and if the most recent one is within 30 minutes
        thirty_minutes_ago = datetime.now().timestamp() - (30 * 60)

        if recent_readings and recent_readings[0]["timestamp"] >= thirty_minutes_ago:
            # We have a recent reading - healthy
            last_reading = recent_readings[0]
            return {
                "status": "healthy",
                "message": "Bluetti Monitor is running and collecting data",
                "last_reading_age_minutes": round(
                    (datetime.now().timestamp() - last_reading["timestamp"]) / 60, 1
                ),
                "battery_percentage": last_reading["battery_percentage"],
            }

        # No recent readings - check uptime
        if startup_time:
            uptime_minutes = (datetime.now().timestamp() - startup_time) / 60

            if uptime_minutes > 30:
                # We've been running for more than 30 minutes with no data - unhealthy
                return {
                    "status": "unhealthy",
                    "message": f"No battery readings in last 30 minutes (uptime: {uptime_minutes:.1f} minutes)",
                    "uptime_minutes": round(uptime_minutes, 1),
                }
            else:
                # We haven't been running long enough to expect data yet - healthy but no data
                return {
                    "status": "healthy",
                    "message": f"Bluetti Monitor is running but no data yet (uptime: {uptime_minutes:.1f} minutes)",
                    "uptime_minutes": round(uptime_minutes, 1),
                }
        else:
            # Startup time not available - assume healthy for now
            return {
                "status": "healthy",
                "message": "Bluetti Monitor is running (startup time not available)",
            }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "error": str(e),
        }


@app.get("/status")
async def get_battery_status():
    """Get recent battery status from database with change calculation"""
    try:
        # Get recent readings for analysis (last 5 readings)
        recent_readings = await db.get_recent_readings(limit=5)

        if not recent_readings:
            return {
                "success": False,
                "message": "No battery readings available",
                "last_reading": None,
            }

        latest = recent_readings[0]

        # Calculate change per minute if we have multiple readings
        change_per_minute = None
        change_direction = None
        readings_analyzed = 0

        if len(recent_readings) >= 2:
            # Use up to last 5 readings for change calculation
            readings_for_change = recent_readings[: min(5, len(recent_readings))]
            readings_analyzed = len(readings_for_change)

            # Calculate time span and percentage change
            oldest = readings_for_change[-1]
            newest = readings_for_change[0]

            time_diff_minutes = (newest["timestamp"] - oldest["timestamp"]) / 60
            percentage_diff = (
                newest["battery_percentage"] - oldest["battery_percentage"]
            )

            if time_diff_minutes > 0:
                change_per_minute = round(percentage_diff / time_diff_minutes, 2)

                if change_per_minute > 0.1:
                    change_direction = "charging"
                elif change_per_minute < -0.1:
                    change_direction = "discharging"
                else:
                    change_direction = "stable"

        return {
            "success": True,
            "battery_percentage": latest["battery_percentage"],
            "confidence": latest["confidence"],
            "seconds_ago": latest["seconds_ago"],
            "last_reading_timestamp": latest["timestamp"],
            "change_per_minute": change_per_minute,
            "change_direction": change_direction,
            "readings_analyzed_for_change": readings_analyzed,
            "ocr_method": latest["ocr_method"],
            "total_recent_readings": len(recent_readings),
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to get battery status: {str(e)}"}


@app.get("/status/history")
async def get_battery_history(limit: int = 20):
    """Get historical battery readings"""
    try:
        readings = await db.get_recent_readings(limit=limit)
        return {"success": True, "readings": readings, "count": len(readings)}
    except Exception as e:
        return {"success": False, "error": f"Failed to get battery history: {str(e)}"}


@app.get("/recommendations")
async def get_power_recommendations():
    """Get power management recommendations based on recent battery status"""
    try:
        # Get readings from the last 30 minutes
        thirty_minutes_ago = datetime.now().timestamp() - (30 * 60)

        # Get recent readings and filter to last 30 minutes
        recent_readings = await db.get_recent_readings(limit=100)
        readings_last_30min = [
            reading
            for reading in recent_readings
            if reading["timestamp"] >= thirty_minutes_ago
        ]

        # Use shared recommendation logic
        return analyze_recent_readings_for_recommendations(readings_last_30min)

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate recommendations: {str(e)}",
        }


@app.get("/camera/status")
async def camera_status():
    webcam_url = os.getenv("WEBCAM_URL")
    capture_endpoint = os.getenv("WEBCAM_CAPTURE_ENDPOINT", "/capture")

    if not webcam_url:
        return {"status": "error", "message": "WEBCAM_URL not configured"}

    capture_url = urljoin(webcam_url, capture_endpoint)

    try:
        response = requests.get(capture_url, timeout=10)
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            if "image" in content_type or len(response.content) > 1000:
                return {
                    "status": "ok",
                    "message": "Camera operational",
                    "capture_url": capture_url,
                }

        return {
            "status": "error",
            "message": "Image capture failed",
            "capture_url": capture_url,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Camera unreachable: {str(e)}",
            "capture_url": capture_url,
        }


@app.get("/capture/crop")
async def capture_crop(x1: int = None, y1: int = None, x2: int = None, y2: int = None):
    """Debug endpoint: Shows cropped battery area as image

    Optional parameters:
    - x1, y1, x2, y2: Override default crop coordinates
    """
    try:
        # Capture and decode image
        image_bytes = capture_image()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode image"}

        # Get crop coordinates (use parameters if provided, otherwise defaults)
        default_coords = get_crop_coordinates()
        coords = {
            "x1": x1 if x1 is not None else default_coords["x1"],
            "y1": y1 if y1 is not None else default_coords["y1"],
            "x2": x2 if x2 is not None else default_coords["x2"],
            "y2": y2 if y2 is not None else default_coords["y2"],
        }

        # Validate coordinates
        h, w = img.shape[:2]
        coords["x1"] = max(0, min(coords["x1"], w - 1))
        coords["y1"] = max(0, min(coords["y1"], h - 1))
        coords["x2"] = max(coords["x1"] + 1, min(coords["x2"], w))
        coords["y2"] = max(coords["y1"] + 1, min(coords["y2"], h))

        # Crop image
        cropped = img[coords["y1"] : coords["y2"], coords["x1"] : coords["x2"]]

        # Encode as JPEG and return
        _, buffer = cv2.imencode(".jpg", cropped)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return {"error": f"Crop failed: {str(e)}"}


@app.get("/capture/flip")
async def capture_flip():
    """Debug endpoint: Shows cropped + flipped battery area as image"""
    try:
        # Capture and decode image
        image_bytes = capture_image()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode image"}

        # Get crop coordinates and crop
        coords = get_crop_coordinates()
        result = img[coords["y1"] : coords["y2"], coords["x1"] : coords["x2"]]

        # Apply flip based on environment variable
        if os.getenv("IMAGE_FLIP_HORIZONTAL", "true").lower() == "true":
            result = cv2.flip(result, 1)  # Flip horizontally

        # Apply rotation based on environment variable
        if os.getenv("IMAGE_ROTATE_180", "false").lower() == "true":
            result = cv2.rotate(result, cv2.ROTATE_180)

        # Encode as JPEG and return
        _, buffer = cv2.imencode(".jpg", result)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return {"error": f"Flip failed: {str(e)}"}


@app.get("/capture/advanced")
async def capture_advanced(
    x1: int = Query(None, description="Left coordinate of crop area"),
    y1: int = Query(None, description="Top coordinate of crop area"),
    x2: int = Query(None, description="Right coordinate of crop area"),
    y2: int = Query(None, description="Bottom coordinate of crop area"),
    flip: bool = Query(None, description="Flip image horizontally"),
    rotate: bool = Query(None, description="Rotate image 180 degrees")
):
    """Advanced capture endpoint with customizable crop and transformations"""
    try:
        # Capture and decode image
        image_bytes = capture_image()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode image"}

        # Use provided coordinates or defaults
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            # Use provided coordinates
            result = img[y1:y2, x1:x2]
        else:
            # Use default crop coordinates
            coords = get_crop_coordinates()
            result = img[coords["y1"]:coords["y2"], coords["x1"]:coords["x2"]]

        # Apply flip if requested (or use env default if not specified)
        if flip is None:
            flip = os.getenv("IMAGE_FLIP_HORIZONTAL", "true").lower() == "true"
        if flip:
            result = cv2.flip(result, 1)  # Flip horizontally

        # Apply rotation if requested (or use env default if not specified)
        if rotate is None:
            rotate = os.getenv("IMAGE_ROTATE_180", "false").lower() == "true"
        if rotate:
            result = cv2.rotate(result, cv2.ROTATE_180)

        # Encode as JPEG and return
        _, buffer = cv2.imencode(".jpg", result)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return {"error": f"Advanced capture failed: {str(e)}"}


@app.get("/capture/ocr")
async def capture_ocr():
    """Debug endpoint: Shows OCR result from cropped+flipped area"""
    try:
        # Capture and decode image
        image_bytes = capture_image()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode image"}

        # Check if screen is off first - no point doing OCR if screen is off
        screen_analysis = analyze_screen_state(image_bytes)
        if screen_analysis.get("screen_state") == "off":
            return {
                "success": True,
                "screen_state": "off",
                "message": "Screen is off - OCR skipped",
                "screen_analysis": screen_analysis,
            }

        # Get crop coordinates and crop
        coords = get_crop_coordinates()
        result = img[coords["y1"] : coords["y2"], coords["x1"] : coords["x2"]]

        # Apply transformations based on environment variables
        if os.getenv("IMAGE_FLIP_HORIZONTAL", "true").lower() == "true":
            result = cv2.flip(result, 1)  # Flip horizontally
        if os.getenv("IMAGE_ROTATE_180", "false").lower() == "true":
            result = cv2.rotate(result, cv2.ROTATE_180)

        # Apply the optimal preprocessing we found through testing
        # Convert to grayscale and upscale (raw preprocessing works best)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upscaled = cv2.resize(
            gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
        )

        # Apply the optimal threshold: 160 with invert (from your testing)
        _, optimal_binary = cv2.threshold(upscaled, 160, 255, cv2.THRESH_BINARY_INV)

        # Use only the best performing configs (PSM 6 & 7 based on debug results)
        configs = [
            ("psm_6", "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"),
            ("psm_7", "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"),
        ]

        results = []
        for name, config in configs:
            try:
                raw_text = pytesseract.image_to_string(
                    optimal_binary, config=config
                ).strip()
                digits_only = "".join(c for c in raw_text if c.isdigit())

                if (
                    digits_only
                    and digits_only.isdigit()
                    and 0 <= int(digits_only) <= 100
                ):
                    # Score based on length preference (2 digits ideal) and PSM preference
                    length_score = 10 - abs(len(digits_only) - 2)
                    psm_bonus = (
                        2 if name == "psm_6" else 1
                    )  # Slight preference for PSM 6

                    results.append(
                        {
                            "text": digits_only,
                            "confidence": length_score + psm_bonus,
                            "method": name,
                            "raw_ocr": raw_text,
                        }
                    )
            except:
                continue

        # Pick best result
        if results:
            best = max(results, key=lambda x: x["confidence"])
            text = best["text"]
            processing_method = f"threshold_160_{best['method']}"
        else:
            text = ""
            processing_method = "threshold_160_failed"

        return {
            "success": True,
            "screen_state": "on",
            "crop_coordinates": coords,
            "image_size": [cropped.shape[1], cropped.shape[0]],
            "raw_text": text,
            "processing_method": processing_method,
            "battery_percentage": (
                int(text) if text.isdigit() and len(text) <= 3 else None
            ),
            "attempts_tried": len(configs),
        }

    except Exception as e:
        return {"error": f"OCR failed: {str(e)}"}


@app.get("/capture/processed")
async def capture_processed(method: str = "enhanced"):
    """Debug endpoint: Shows preprocessed image for OCR debugging

    Parameters:
    - method: "raw", "enhanced", "otsu", "adaptive", "manual", "invert", "blue_channel"
    """
    try:
        # Capture and decode image
        image_bytes = capture_image()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode image"}

        # Check if screen is off first
        screen_analysis = analyze_screen_state(image_bytes)
        if screen_analysis.get("screen_state") == "off":
            return {"error": "Screen is off - no preprocessing to show"}

        # Get crop coordinates and crop
        coords = get_crop_coordinates()
        result = img[coords["y1"] : coords["y2"], coords["x1"] : coords["x2"]]

        # Apply transformations based on environment variables
        if os.getenv("IMAGE_FLIP_HORIZONTAL", "true").lower() == "true":
            result = cv2.flip(result, 1)  # Flip horizontally
        if os.getenv("IMAGE_ROTATE_180", "false").lower() == "true":
            result = cv2.rotate(result, cv2.ROTATE_180)

        # Let's debug step by step
        if method == "raw":
            # Just the cropped and flipped image, upscaled
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            processed = cv2.resize(
                gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
            )

        elif method == "blue_channel":
            # Try using just the blue channel (white text on blue background)
            height, width = result.shape[:2]
            blue_channel = result[:, :, 0]  # Blue channel in BGR
            processed = cv2.resize(
                blue_channel, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
            )

        elif method == "enhanced":
            # Show the enhanced version before thresholding
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            resized = cv2.resize(
                gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
            )
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            processed = clahe.apply(blurred)

        elif method == "invert":
            # Simple invert of enhanced image
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            resized = cv2.resize(
                gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
            )
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            processed = cv2.bitwise_not(enhanced)  # Invert: white becomes black

        # Thresholding methods
        else:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            resized = cv2.resize(
                gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
            )
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)

            if method == "otsu":
                _, processed = cv2.threshold(
                    enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            elif method == "adaptive":
                processed = cv2.adaptiveThreshold(
                    enhanced,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                )
            elif method == "manual":
                _, processed = cv2.threshold(
                    enhanced, 180, 255, cv2.THRESH_BINARY_INV
                )  # Lower threshold
            else:
                return {
                    "error": f"Unknown method: {method}. Use 'raw', 'enhanced', 'blue_channel', 'invert', 'otsu', 'adaptive', or 'manual'"
                }

        # Encode and return
        _, buffer = cv2.imencode(".jpg", processed)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}


@app.get("/capture/threshold")
async def capture_threshold(
    threshold: int = 180, invert: bool = True, preprocess: str = "enhanced"
):
    """Debug endpoint: Test different threshold values manually

    Parameters:
    - threshold: Threshold value (0-255)
    - invert: True for THRESH_BINARY_INV (white->black), False for THRESH_BINARY (black->white)
    - preprocess: "raw", "enhanced", "blue_channel"
    """
    try:
        # Capture and decode image
        image_bytes = capture_image()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode image"}

        # Check if screen is off first
        screen_analysis = analyze_screen_state(image_bytes)
        if screen_analysis.get("screen_state") == "off":
            return {"error": "Screen is off - no thresholding to show"}

        # Get crop coordinates and crop
        coords = get_crop_coordinates()
        result = img[coords["y1"] : coords["y2"], coords["x1"] : coords["x2"]]

        # Apply transformations based on environment variables
        if os.getenv("IMAGE_FLIP_HORIZONTAL", "true").lower() == "true":
            result = cv2.flip(result, 1)  # Flip horizontally
        if os.getenv("IMAGE_ROTATE_180", "false").lower() == "true":
            result = cv2.rotate(result, cv2.ROTATE_180)

        # Apply requested preprocessing
        if preprocess == "raw":
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            processed_img = cv2.resize(
                gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
            )

        elif preprocess == "blue_channel":
            height, width = result.shape[:2]
            blue_channel = result[:, :, 0]  # Blue channel
            processed_img = cv2.resize(
                blue_channel, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
            )

        elif preprocess == "enhanced":
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            resized = cv2.resize(
                gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
            )
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            processed_img = clahe.apply(blurred)

        else:
            return {
                "error": f"Unknown preprocess: {preprocess}. Use 'raw', 'enhanced', or 'blue_channel'"
            }

        # Apply manual threshold with specified value
        if invert:
            _, thresholded = cv2.threshold(
                processed_img, threshold, 255, cv2.THRESH_BINARY_INV
            )
        else:
            _, thresholded = cv2.threshold(
                processed_img, threshold, 255, cv2.THRESH_BINARY
            )

        # Encode as JPEG and return
        _, buffer = cv2.imencode(".jpg", thresholded)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
        return {"error": f"Threshold test failed: {str(e)}"}


@app.get("/capture/ocr_debug")
async def capture_ocr_debug(threshold: int = 160):
    """Debug endpoint: Shows detailed OCR results from all configs

    Parameters:
    - threshold: Threshold value to use (default 160)
    """
    try:
        # Capture and decode image
        image_bytes = capture_image()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Failed to decode image"}

        # Check if screen is off first
        screen_analysis = analyze_screen_state(image_bytes)
        if screen_analysis.get("screen_state") == "off":
            return {
                "success": False,
                "screen_state": "off",
                "message": "Screen is off - OCR debug skipped",
            }

        # Get crop coordinates and crop
        coords = get_crop_coordinates()
        result = img[coords["y1"] : coords["y2"], coords["x1"] : coords["x2"]]

        # Apply transformations based on environment variables
        if os.getenv("IMAGE_FLIP_HORIZONTAL", "true").lower() == "true":
            result = cv2.flip(result, 1)  # Flip horizontally
        if os.getenv("IMAGE_ROTATE_180", "false").lower() == "true":
            result = cv2.rotate(result, cv2.ROTATE_180)

        # Apply preprocessing with specified threshold
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upscaled = cv2.resize(
            gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
        )
        _, binary = cv2.threshold(upscaled, threshold, 255, cv2.THRESH_BINARY_INV)

        # Try all OCR configs and collect detailed results
        configs = [
            (
                "psm_6",
                "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789%",
            ),  # Allow % symbol
            ("psm_7", "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789%"),
            ("psm_8", "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789%"),
            ("psm_13", "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789%"),
            (
                "psm_6_digits_only",
                "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789",
            ),
            (
                "psm_7_digits_only",
                "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
            ),
            (
                "psm_8_digits_only",
                "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
            ),
            (
                "psm_13_digits_only",
                "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
            ),
        ]

        ocr_results = []
        for name, config in configs:
            try:
                raw_text = pytesseract.image_to_string(binary, config=config).strip()

                # Extract just the digits for battery percentage
                digits_only = "".join(c for c in raw_text if c.isdigit())

                ocr_results.append(
                    {
                        "config_name": name,
                        "raw_text": raw_text,
                        "raw_text_repr": repr(raw_text),  # Shows escape chars
                        "digits_only": digits_only,
                        "is_valid_percentage": (
                            digits_only.isdigit()
                            and len(digits_only) <= 3
                            and 0 <= int(digits_only) <= 100
                            if digits_only
                            else False
                        ),
                        "percentage_value": (
                            int(digits_only)
                            if digits_only and digits_only.isdigit()
                            else None
                        ),
                    }
                )
            except Exception as e:
                ocr_results.append({"config_name": name, "error": str(e)})

        # Find the most consistent digit result
        digit_results = [
            r
            for r in ocr_results
            if r.get("digits_only") and r.get("is_valid_percentage")
        ]
        if digit_results:
            # Count frequency of each result
            from collections import Counter

            digit_counts = Counter(r["digits_only"] for r in digit_results)
            most_common = digit_counts.most_common(1)[0]
            best_guess = most_common[0]
            confidence = most_common[1] / len(digit_results)
        else:
            best_guess = None
            confidence = 0.0

        return {
            "success": True,
            "threshold_used": threshold,
            "crop_coordinates": coords,
            "ocr_results": ocr_results,
            "best_guess": best_guess,
            "confidence": round(confidence, 3),
            "total_attempts": len(configs),
        }

    except Exception as e:
        return {"error": f"OCR debug failed: {str(e)}"}


@app.get("/capture/ocr_advanced")
async def capture_ocr_advanced():
    """Advanced OCR endpoint with voting across multiple thresholds, PSM modes, and image captures"""
    try:
        # Check if screen is on with a single capture first
        test_image = capture_image()
        screen_analysis = analyze_screen_state(test_image)
        if screen_analysis.get("screen_state") == "off":
            return {
                "success": False,
                "screen_state": "off",
                "message": "Screen is off - advanced OCR skipped",
            }

        # Test parameters - covering different lighting conditions
        thresholds = [145, 150, 155, 160, 165, 170]  # From dark/night to bright/sunny
        psm_modes = [6, 7, 8, 13]  # All the modes we've tested
        num_captures = 3  # Multiple captures to reduce randomness

        all_results = []
        capture_details = []

        # Take multiple image captures and test each
        for capture_num in range(num_captures):
            try:
                # Capture image
                image_bytes = capture_image()
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    capture_details.append(
                        {"capture": capture_num + 1, "error": "Failed to decode image"}
                    )
                    continue

                # Get crop coordinates and crop
                coords = get_crop_coordinates()
                result = img[coords["y1"] : coords["y2"], coords["x1"] : coords["x2"]]

                # Apply transformations based on environment variables
                if os.getenv("IMAGE_FLIP_HORIZONTAL", "true").lower() == "true":
                    result = cv2.flip(result, 1)  # Flip horizontally
                if os.getenv("IMAGE_ROTATE_180", "false").lower() == "true":
                    result = cv2.rotate(result, cv2.ROTATE_180)

                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape
                upscaled = cv2.resize(
                    gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
                )

                capture_results = []

                # Test all threshold/PSM combinations
                for threshold in thresholds:
                    _, binary = cv2.threshold(
                        upscaled, threshold, 255, cv2.THRESH_BINARY_INV
                    )

                    for psm_mode in psm_modes:
                        try:
                            config = f"--oem 3 --psm {psm_mode} -c tessedit_char_whitelist=0123456789%"
                            raw_text = pytesseract.image_to_string(
                                binary, config=config
                            ).strip()

                            if raw_text:  # Not empty
                                digits_only = "".join(
                                    c for c in raw_text if c.isdigit()
                                )

                                if digits_only:
                                    # Smart digit handling with special cases
                                    if len(digits_only) == 4 and digits_only.startswith(
                                        "100"
                                    ):
                                        # Special case: 100% shows as "1000", "1004", etc.
                                        percentage_candidate = 100
                                        source = "4digit_100percent"
                                    elif len(digits_only) == 3:
                                        # 3-digit: assume last digit is % symbol misread
                                        percentage_candidate = int(digits_only[:2])
                                        source = "3digit_truncated"
                                    elif len(digits_only) == 2:
                                        percentage_candidate = int(digits_only)
                                        source = "2digit"
                                    elif len(digits_only) == 1:
                                        percentage_candidate = int(digits_only)
                                        source = "1digit"
                                    else:
                                        continue  # Skip other cases

                                    # Validate percentage range
                                    if 0 <= percentage_candidate <= 100:
                                        capture_results.append(
                                            {
                                                "threshold": threshold,
                                                "psm": psm_mode,
                                                "raw_text": raw_text,
                                                "digits_only": digits_only,
                                                "percentage": percentage_candidate,
                                                "source": source,
                                            }
                                        )
                                        all_results.append(percentage_candidate)
                        except:
                            continue

                capture_details.append(
                    {
                        "capture": capture_num + 1,
                        "results_count": len(capture_results),
                        "results": capture_results,
                    }
                )

            except Exception as e:
                capture_details.append(
                    {"capture": capture_num + 1, "error": f"Capture failed: {str(e)}"}
                )

        # Enhanced voting mechanism with conflict resolution
        if all_results:
            from collections import Counter, defaultdict

            vote_counts = Counter(all_results)

            # Group results by source type for analysis
            source_analysis = defaultdict(list)
            for detail in capture_details:
                for result in detail.get("results", []):
                    source_analysis[result["source"]].append(result["percentage"])

            # Detect potential conflicts (e.g., 9 vs 90, 1 vs 10)
            def are_conflicting(num1, num2):
                # Check if one number is the other with a trailing zero
                return (num1 * 10 == num2) or (num2 * 10 == num1)

            # Smart conflict resolution
            resolved_votes = Counter()
            conflict_detected = False

            for percentage, count in vote_counts.items():
                # Check if this percentage conflicts with any other
                conflicts_with = []
                for other_percentage, other_count in vote_counts.items():
                    if percentage != other_percentage and are_conflicting(
                        percentage, other_percentage
                    ):
                        conflicts_with.append((other_percentage, other_count))

                if conflicts_with:
                    conflict_detected = True
                    # Resolve conflict: prefer 2-digit sources over 1-digit
                    current_sources = [
                        r["source"]
                        for detail in capture_details
                        for r in detail.get("results", [])
                        if r["percentage"] == percentage
                    ]

                    # Weight: 2digit > 3digit_truncated > 1digit
                    source_weights = {
                        "2digit": 3,
                        "3digit_truncated": 2,
                        "1digit": 1,
                        "4digit_100percent": 4,
                    }
                    current_weight = max(
                        source_weights.get(src, 0) for src in current_sources
                    )

                    # Compare with conflicting results
                    should_prefer_current = True
                    for conflict_pct, conflict_count in conflicts_with:
                        conflict_sources = [
                            r["source"]
                            for detail in capture_details
                            for r in detail.get("results", [])
                            if r["percentage"] == conflict_pct
                        ]
                        conflict_weight = max(
                            source_weights.get(src, 0) for src in conflict_sources
                        )

                        # If conflict has higher weight or significantly more votes, prefer it
                        if conflict_weight > current_weight or (
                            conflict_weight == current_weight
                            and conflict_count > count * 1.5
                        ):
                            should_prefer_current = False
                            break

                    if should_prefer_current:
                        resolved_votes[percentage] = count
                else:
                    resolved_votes[percentage] = count

            # Use resolved votes for final decision
            most_common = resolved_votes.most_common()
            winner = most_common[0] if most_common else (0, 0)

            # Calculate confidence
            total_votes = len(all_results)
            winning_votes = winner[1]
            confidence = winning_votes / total_votes if total_votes > 0 else 0

            # Check for ties (multiple results with same vote count)
            ties = [result for result, count in most_common if count == winning_votes]

            return {
                "success": True,
                "battery_percentage": winner[0],
                "confidence": round(confidence, 3),
                "total_attempts": total_votes,
                "winning_votes": winning_votes,
                "vote_distribution": dict(vote_counts),
                "resolved_vote_distribution": (
                    dict(resolved_votes) if conflict_detected else None
                ),
                "conflict_resolution_applied": conflict_detected,
                "tied_results": ties if len(ties) > 1 else None,
                "captures_attempted": num_captures,
                "captures_succeeded": len(
                    [c for c in capture_details if "error" not in c]
                ),
                "source_analysis": dict(source_analysis),
                "test_parameters": {"thresholds": thresholds, "psm_modes": psm_modes},
                "detailed_results": capture_details,
            }
        else:
            return {
                "success": False,
                "message": "No valid OCR results obtained",
                "captures_attempted": num_captures,
                "detailed_results": capture_details,
            }

    except Exception as e:
        return {"error": f"Advanced OCR failed: {str(e)}"}


# Background polling is now handled by the separate worker service


def analyze_screen_state(image_bytes):
    """
    Analyze image to determine if Bluetti screen is on or off.
    Returns: {"screen_state": "on"|"off", "confidence": float, "metrics": dict}
    """
    # Convert bytes to opencv image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "screen_state": "unknown",
            "confidence": 0.0,
            "error": "Failed to decode image",
        }

    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calculate overall brightness (V channel mean)
    brightness = np.mean(hsv[:, :, 2])

    # Calculate blue content (Bluetti screens have significant blue when on)
    # Blue hue range in HSV: roughly 100-130
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    blue_percentage = (np.sum(blue_mask > 0) / (img.shape[0] * img.shape[1])) * 100

    # Simple classification based on brightness and blue content
    # Screen ON: Higher brightness + significant blue content
    # Screen OFF: Lower brightness + minimal blue content

    metrics = {
        "brightness": float(brightness),
        "blue_percentage": float(blue_percentage),
        "image_size": [img.shape[1], img.shape[0]],  # width, height
    }

    # Classification logic based on your real data
    # Screen ON: brightness ~58, blue ~42%
    # Screen OFF: brightness ~27, blue ~0.08%

    if blue_percentage > 20:  # Strong blue indicates screen on
        screen_state = "on"
        confidence = min(0.95, 0.7 + (blue_percentage / 50))
    elif brightness > 50 and blue_percentage > 5:  # Medium brightness + some blue
        screen_state = "on"
        confidence = 0.8
    elif brightness < 35 and blue_percentage < 1:  # Low brightness + minimal blue
        screen_state = "off"
        confidence = min(0.9, 0.6 + (1 - brightness / 50))
    else:
        # Uncertain cases - use brightness as tiebreaker
        if brightness > 45:
            screen_state = "on"
            confidence = 0.6
        else:
            screen_state = "off"
            confidence = 0.6

    return {
        "screen_state": screen_state,
        "confidence": round(confidence, 3),
        "metrics": metrics,
    }


@app.get("/capture/analyze")
async def capture_and_analyze():
    """
    Capture image from webcam and analyze screen state
    """
    webcam_url = os.getenv("WEBCAM_URL")
    capture_endpoint = os.getenv("WEBCAM_CAPTURE_ENDPOINT", "/capture")

    if not webcam_url:
        return {"error": "WEBCAM_URL not configured"}

    capture_url = urljoin(webcam_url, capture_endpoint)

    # Retry logic for network issues
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            # Capture image
            response = requests.get(capture_url, timeout=10)

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
                if attempt < max_retries - 1:
                    continue
                return {
                    "error": "Failed to capture image after retries",
                    "status_code": response.status_code,
                    "capture_url": capture_url,
                    "attempts": attempt + 1,
                }

            # Analyze the captured image
            analysis = analyze_screen_state(response.content)

            result = {
                "success": True,
                "capture_url": capture_url,
                "timestamp": response.headers.get("date"),
                **analysis,
            }

            # Add battery percentage if screen is on
            if analysis.get("screen_state") == "on":
                try:
                    # Convert response content to opencv image for OCR
                    nparr = np.frombuffer(response.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is not None:
                        # Use the same OCR logic
                        coords = get_crop_coordinates()
                        result = img[
                            coords["y1"] : coords["y2"], coords["x1"] : coords["x2"]
                        ]

                        # Apply transformations based on environment variables
                        if os.getenv("IMAGE_FLIP_HORIZONTAL", "true").lower() == "true":
                            result = cv2.flip(result, 1)  # Flip horizontally
                        if os.getenv("IMAGE_ROTATE_180", "false").lower() == "true":
                            result = cv2.rotate(result, cv2.ROTATE_180)

                        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                        height, width = gray.shape
                        upscaled = cv2.resize(
                            gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC
                        )
                        _, binary = cv2.threshold(
                            upscaled, 160, 255, cv2.THRESH_BINARY_INV
                        )

                        # Try PSM 6 and 7 (best performers from debug)
                        for psm_mode in [6, 7]:
                            try:
                                config = f"--oem 3 --psm {psm_mode} -c tessedit_char_whitelist=0123456789"
                                raw_text = pytesseract.image_to_string(
                                    binary, config=config
                                ).strip()
                                digits_only = "".join(
                                    c for c in raw_text if c.isdigit()
                                )

                                if (
                                    digits_only
                                    and digits_only.isdigit()
                                    and 0 <= int(digits_only) <= 100
                                ):
                                    result["battery_percentage"] = int(digits_only)
                                    result["battery_ocr_method"] = f"psm_{psm_mode}"
                                    break
                            except:
                                continue

                        if "battery_percentage" not in result:
                            result["battery_percentage"] = None
                            result["battery_ocr_method"] = "failed"
                    else:
                        result["battery_percentage"] = None
                        result["battery_ocr_method"] = "image_decode_failed"

                except Exception as e:
                    result["battery_percentage"] = None
                    result["battery_ocr_method"] = f"error: {str(e)}"
            else:
                result["battery_percentage"] = None
                result["battery_ocr_method"] = "screen_off"

            # Add retry info if we had to retry
            if attempt > 0:
                result["attempts"] = attempt + 1

            return result

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                # Wait briefly before retry
                import time

                time.sleep(0.5)
                continue

    return {
        "error": f"Capture failed after {max_retries} attempts: {last_error}",
        "capture_url": capture_url,
        "attempts": max_retries,
    }


def get_device_states():
    """
    Query device states from the device control API using dynamic discovery

    Returns:
        dict: Dictionary with device states for all discovered devices
              Returns empty dict on error
    """
    return device_discovery.get_device_states()


@app.get("/devices")
async def get_devices():
    """Get discovered devices and segmentation info"""
    try:
        discovery_result = device_discovery.discover_devices()
        segmentation_info = device_discovery.get_segmentation_info()

        return {
            "success": discovery_result["success"],
            "discovery": discovery_result,
            "segmentation": segmentation_info,
            "device_states": get_device_states()
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to get devices: {str(e)}"}


@app.get("/switchbot/status")
async def get_switchbot_status():
    """Get SwitchBot controller status and rate limiting info"""
    try:
        # Query device states to provide accurate interval info
        device_states = get_device_states()

        # Check if any inputs or outputs are on
        input_on = device_discovery.is_any_input_on()
        output_on = device_discovery.is_any_output_on()

        status = await switchbot_controller.get_status(input_on=input_on, output_on=output_on)
        return {"success": True, "device_states": device_states, **status}
    except Exception as e:
        return {"success": False, "error": f"Failed to get SwitchBot status: {str(e)}"}


@app.post("/switchbot/tap")
async def tap_screen_manual(force: bool = False):
    """
    Manually tap the screen using SwitchBot

    Args:
        force: If True, bypass 15-minute rate limiting (use with caution)
    """
    try:
        result = await switchbot_controller.tap_screen(force=force)
        return result
    except Exception as e:
        return {"success": False, "error": f"Failed to tap screen: {str(e)}"}


@app.get("/training/status")
async def get_training_status_endpoint():
    """
    Get template matching training status and coverage statistics

    Returns:
        Dictionary with coverage stats, images per percentage, and total images
    """
    try:
        status = get_training_status()
        return {"success": True, **status}
    except Exception as e:
        return {"success": False, "error": f"Failed to get training status: {str(e)}"}


@app.post("/training/enable")
async def enable_training_collection(enabled: bool = True):
    """
    Manually enable or disable training image collection

    Args:
        enabled: True to enable collection, False to disable (default: True)

    Returns:
        Success status and current collection state
    """
    try:
        template_classifier.enable_collection(enabled)
        status = get_training_status()
        return {
            "success": True,
            "collection_enabled": status["collection_enabled"],
            "message": f"Collection {'enabled' if enabled else 'disabled'}"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to change collection state: {str(e)}"}


@app.post("/training/label/{category}")
async def manually_label_last_image(category: str):
    """
    Manually label/relabel the last captured image with a specific category

    Args:
        category: Correct category value (0-100 or "invalid")

    Returns:
        Success status
    """
    try:
        # Validate category
        if category != "invalid":
            try:
                cat_int = int(category)
                if not (0 <= cat_int <= 100):
                    return {
                        "success": False,
                        "error": f"Invalid category {category}, must be 0-100 or 'invalid'"
                    }
                category = cat_int  # Convert to int for consistency
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid category {category}, must be 0-100 or 'invalid'"
                }

        # Capture and process current image
        image_bytes = capture_image()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "error": "Failed to decode image"}

        # Get crop coordinates and process (same as /capture/flip)
        coords = get_crop_coordinates()
        result = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]

        # Apply transformations
        if os.getenv("IMAGE_FLIP_HORIZONTAL", "true").lower() == "true":
            result = cv2.flip(result, 1)
        if os.getenv("IMAGE_ROTATE_180", "false").lower() == "true":
            result = cv2.rotate(result, cv2.ROTATE_180)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', result)
        processed_image = buffer.tobytes()

        # Save with manual label
        success = template_classifier.manually_label_image(processed_image, category)

        if success:
            status = get_training_status()
            return {
                "success": True,
                "message": f"Successfully labeled image as {category}",
                "coverage_stats": status
            }
        else:
            return {
                "success": False,
                "error": "Failed to save labeled image"
            }

    except Exception as e:
        return {"success": False, "error": f"Failed to label image: {str(e)}"}


@app.get("/training/images")
async def list_training_images(category: str = None, filter: str = "all"):
    """
    List all training images, optionally filtered by category and verification status.

    Args:
        category: Optional filter for specific category (0-100 or "invalid")
        filter: "all" or "unverified" (default: "all")

    Returns:
        Dictionary with images grouped by category, including verified status
    """
    try:
        training_dir = Path(template_classifier.training_data_dir)
        result = {}

        # Determine which categories to check
        if category is not None:
            categories_to_check = [category]
        else:
            # Check all percentages (0-100) plus "invalid"
            categories_to_check = [str(i) for i in range(101)] + ["invalid"]

        for cat in categories_to_check:
            cat_dir = training_dir / str(cat)
            if cat_dir.exists():
                images = sorted(cat_dir.glob("*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
                if images:
                    # Build list with verified status
                    image_list = []
                    for img in images:
                        is_verified = template_classifier.is_verified(img.name)

                        # Apply filter
                        if filter == "unverified" and is_verified:
                            continue

                        image_list.append({
                            "filename": img.name,
                            "verified": is_verified
                        })

                    if image_list:
                        result[cat] = image_list

        return {"success": True, "images": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/training/image/{category}/{filename}")
async def get_training_image(category: str, filename: str, preprocess: bool = False):
    """
    Serve a specific training image.

    Args:
        category: The category folder (0-100 or "invalid")
        filename: The image filename
        preprocess: If true, apply classifier preprocessing (histogram equalization)

    Returns:
        The image file (raw or preprocessed)
    """
    try:
        # Validate category
        if category != "invalid":
            try:
                cat_int = int(category)
                if not (0 <= cat_int <= 100):
                    return {"success": False, "error": "Invalid category"}
            except ValueError:
                return {"success": False, "error": "Invalid category"}

        training_dir = Path(template_classifier.training_data_dir)
        image_path = training_dir / str(category) / filename

        if not image_path.exists():
            return {"success": False, "error": "Image not found"}

        # Security check: ensure path is within training_data
        if not str(image_path.resolve()).startswith(str(training_dir.resolve())):
            return {"success": False, "error": "Invalid path"}

        if preprocess:
            # Return preprocessed version (as classifier sees it)
            with open(image_path, 'rb') as f:
                raw_data = f.read()
            preprocessed = template_classifier.get_preprocessed_image(raw_data)
            if preprocessed:
                from fastapi.responses import Response
                return Response(content=preprocessed, media_type="image/jpeg")
            # Fall back to raw if preprocessing fails
            return FileResponse(image_path, media_type="image/jpeg")

        return FileResponse(image_path, media_type="image/jpeg")
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/training/reclassify")
async def reclassify_training_image(
    filename: str = Form(...),
    from_category: str = Form(...),
    to_category: str = Form(...)
):
    """
    Move an image from one category folder to another (reclassify it).

    Args:
        filename: The image filename
        from_category: Current category folder (0-100 or "invalid")
        to_category: Target category folder (0-100 or "invalid")

    Returns:
        Success status
    """
    try:
        # Validate categories
        def is_valid_category(cat):
            if cat == "invalid":
                return True
            try:
                cat_int = int(cat)
                return 0 <= cat_int <= 100
            except (ValueError, TypeError):
                return False

        if not is_valid_category(from_category) or not is_valid_category(to_category):
            return {"success": False, "error": "Invalid category values"}

        if from_category == to_category:
            return {"success": False, "error": "Source and target are the same"}

        training_dir = Path(template_classifier.training_data_dir)
        source_path = training_dir / str(from_category) / filename
        target_dir = training_dir / str(to_category)
        target_path = target_dir / filename

        if not source_path.exists():
            return {"success": False, "error": "Source image not found"}

        # Security check
        if not str(source_path.resolve()).startswith(str(training_dir.resolve())):
            return {"success": False, "error": "Invalid source path"}

        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        # Check FIFO - if target has max images, apply smart deletion
        existing_images = sorted(target_dir.glob("*.jpg"))
        if len(existing_images) >= template_classifier.max_images_per_percentage:
            # Smart FIFO: prefer deleting unverified images
            unverified_images = [img for img in existing_images if not template_classifier.is_verified(img.name)]
            verified_images = [img for img in existing_images if template_classifier.is_verified(img.name)]

            if unverified_images:
                oldest = unverified_images[0]
                oldest.unlink()
                logger.info(f"FIFO rotation in {to_category}/: deleted unverified {oldest.name}")
            elif verified_images:
                oldest = verified_images[0]
                oldest.unlink()
                logger.info(f"FIFO rotation in {to_category}/: deleted verified {oldest.name}")

        # Move the file
        shutil.move(str(source_path), str(target_path))

        # Mark as verified (reclassification implies human review)
        # Remove .verified. from filename first if present, then add it back
        final_path = target_path
        if not template_classifier.is_verified(target_path.name):
            # Add .verified. suffix
            stem = target_path.stem
            suffix = target_path.suffix
            verified_name = f"{stem}.verified{suffix}"
            verified_path = target_dir / verified_name
            target_path.rename(verified_path)
            final_path = verified_path
            logger.info(f"Auto-verified after reclassification: {verified_name}")

        # Reload templates
        template_classifier._load_templates()

        logger.info(f"Reclassified {filename}: {from_category} -> {to_category}")

        # Link ground truth to comparison record (if exists)
        timestamp = extract_timestamp_from_filename(filename)
        if timestamp is not None:
            # Determine if target category is invalid
            is_invalid = (to_category == "invalid")
            human_percentage = None if is_invalid else int(to_category)

            # Update comparison record by timestamp
            updated = await comparison_storage.update_verification_by_timestamp(
                timestamp=timestamp,
                human_percentage=human_percentage,
                is_invalid=is_invalid
            )

            if updated:
                logger.info(f"Linked reclassification to comparison record: {to_category}")
            else:
                logger.debug(f"No matching comparison record found for timestamp {timestamp}")

        return {
            "success": True,
            "message": f"Moved {filename} from {from_category} to {to_category}",
            "coverage_stats": get_training_status()
        }
    except Exception as e:
        logger.error(f"Reclassify failed: {e}")
        return {"success": False, "error": str(e)}


@app.delete("/training/image/{category}/{filename}")
async def delete_training_image(category: str, filename: str):
    """
    Delete a specific training image.

    Args:
        category: The category folder (0-100 or "invalid")
        filename: The image filename

    Returns:
        Success status
    """
    try:
        # Validate category
        if category != "invalid":
            try:
                cat_int = int(category)
                if not (0 <= cat_int <= 100):
                    return {"success": False, "error": "Invalid category"}
            except ValueError:
                return {"success": False, "error": "Invalid category"}

        training_dir = Path(template_classifier.training_data_dir)
        image_path = training_dir / str(category) / filename

        if not image_path.exists():
            return {"success": False, "error": "Image not found"}

        # Security check
        if not str(image_path.resolve()).startswith(str(training_dir.resolve())):
            return {"success": False, "error": "Invalid path"}

        image_path.unlink()
        template_classifier._load_templates()

        logger.info(f"Deleted training image: {category}/{filename}")

        return {
            "success": True,
            "message": f"Deleted {filename} from {category}",
            "coverage_stats": get_training_status()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/training/verify/{category}/{filename}")
async def verify_training_image(category: str, filename: str):
    """
    Mark a specific training image as verified.

    Args:
        category: The category folder (0-100 or "invalid")
        filename: The image filename

    Returns:
        Success status
    """
    try:
        # Validate category
        if category != "invalid":
            try:
                cat_int = int(category)
                if not (0 <= cat_int <= 100):
                    return {"success": False, "error": "Invalid category"}
            except ValueError:
                return {"success": False, "error": "Invalid category"}

        training_dir = Path(template_classifier.training_data_dir)
        image_path = training_dir / str(category) / filename

        if not image_path.exists():
            return {"success": False, "error": "Image not found"}

        # Security check
        if not str(image_path.resolve()).startswith(str(training_dir.resolve())):
            return {"success": False, "error": "Invalid path"}

        # Mark as verified
        success = template_classifier.mark_verified(image_path)

        if success:
            # Reload templates
            template_classifier._load_templates()

            return {
                "success": True,
                "message": f"Marked {filename} as verified"
            }
        else:
            return {"success": False, "error": "Failed to mark as verified"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/training/verify-all")
async def verify_all_images(categories: str = Form(...)):
    """
    Verify all visible images in specified categories (bulk action).

    Args:
        categories: JSON string with category list (e.g., '["50", "51"]')

    Returns:
        Success status with count of verified images
    """
    try:
        import json
        category_list = json.loads(categories)

        training_dir = Path(template_classifier.training_data_dir)
        verified_count = 0
        failed_count = 0

        for category in category_list:
            # Validate category
            if category != "invalid":
                try:
                    cat_int = int(category)
                    if not (0 <= cat_int <= 100):
                        continue
                except (ValueError, TypeError):
                    continue

            cat_dir = training_dir / str(category)
            if cat_dir.exists():
                images = list(cat_dir.glob("*.jpg"))
                for img_path in images:
                    # Skip already verified images
                    if template_classifier.is_verified(img_path.name):
                        continue

                    # Security check
                    if not str(img_path.resolve()).startswith(str(training_dir.resolve())):
                        continue

                    success = template_classifier.mark_verified(img_path)
                    if success:
                        verified_count += 1
                    else:
                        failed_count += 1

        # Reload templates
        template_classifier._load_templates()

        return {
            "success": True,
            "message": f"Verified {verified_count} images" + (f" ({failed_count} failed)" if failed_count > 0 else ""),
            "verified_count": verified_count,
            "failed_count": failed_count
        }

    except Exception as e:
        logger.error(f"Verify all failed: {e}")
        return {"success": False, "error": str(e)}


@app.get("/comparisons/stats")
async def get_comparison_stats():
    """Get aggregated comparison statistics"""
    try:
        stats = await comparison_storage.get_statistics()
        return {"success": True, **stats}
    except Exception as e:
        return {"success": False, "error": f"Failed to get comparison stats: {str(e)}"}


@app.get("/comparisons/records")
async def get_comparison_records(
    limit: int = 50,
    offset: int = 0,
    filter: str = "all",
    value: int = None
):
    """
    Get comparison records with pagination and filtering

    Args:
        limit: Maximum number of records to return (default: 50)
        offset: Number of records to skip (default: 0)
        filter: "all" or "disagreements" (default: "all")
        value: Optional filter for specific battery percentage value
    """
    try:
        records = await comparison_storage.get_records(
            limit=limit,
            offset=offset,
            filter_type=filter,
            value=value
        )
        return {"success": True, "records": records, "count": len(records)}
    except Exception as e:
        return {"success": False, "error": f"Failed to get comparison records: {str(e)}"}


@app.get("/comparisons/image/{filename}")
async def get_comparison_image(filename: str):
    """
    Serve a specific comparison image

    Args:
        filename: The image filename
    """
    try:
        image_path = comparison_storage.comparison_images_dir / filename

        if not image_path.exists():
            return {"success": False, "error": "Image not found"}

        # Security check: ensure path is within comparison_images
        if not str(image_path.resolve()).startswith(str(comparison_storage.comparison_images_dir.resolve())):
            return {"success": False, "error": "Invalid path"}

        return FileResponse(image_path, media_type="image/jpeg")
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/comparisons/verify/{record_id}")
async def verify_comparison_record(record_id: int, human_percentage: int = Form(...)):
    """
    Update comparison record with human-verified ground truth

    Args:
        record_id: Record ID to update
        human_percentage: Human-verified battery percentage (0-100)
    """
    try:
        # Validate percentage
        if not (0 <= human_percentage <= 100):
            return {"success": False, "error": "Invalid percentage value (must be 0-100)"}

        success = await comparison_storage.update_human_verification(record_id, human_percentage)

        if success:
            return {
                "success": True,
                "message": f"Updated record {record_id} with human verification: {human_percentage}%"
            }
        else:
            return {"success": False, "error": "Failed to update record"}

    except Exception as e:
        return {"success": False, "error": f"Failed to verify record: {str(e)}"}


@app.get("/training/review", response_class=HTMLResponse)
async def training_review_ui():
    """
    Serve the training image review UI.
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Image Review</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; color: #eee; padding: 20px;
        }
        h1 { margin-bottom: 10px; color: #00d4ff; }
        .tabs {
            display: flex; gap: 10px; margin-bottom: 20px;
            border-bottom: 2px solid #16213e;
        }
        .tab {
            padding: 12px 24px; cursor: pointer; background: #16213e;
            border-radius: 8px 8px 0 0; transition: all 0.2s;
        }
        .tab:hover { background: #0f3460; }
        .tab.active { background: #00d4ff; color: #1a1a2e; font-weight: bold; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .stats {
            background: #16213e; padding: 15px; border-radius: 8px;
            margin-bottom: 20px; display: flex; gap: 30px; flex-wrap: wrap;
        }
        .stat { text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #00d4ff; }
        .stat-label { font-size: 0.9em; color: #888; }
        .controls {
            background: #16213e; padding: 15px; border-radius: 8px;
            margin-bottom: 20px; display: flex; gap: 15px; align-items: center; flex-wrap: wrap;
        }
        select, button, input {
            padding: 10px 15px; border-radius: 5px; border: none;
            font-size: 1em; cursor: pointer;
        }
        select { background: #0f3460; color: #eee; }
        button { background: #00d4ff; color: #1a1a2e; font-weight: bold; }
        button:hover { background: #00a8cc; }
        button.danger { background: #e94560; color: white; }
        button.danger:hover { background: #c73e54; }
        .grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }
        .image-card {
            background: #16213e; border-radius: 8px; overflow: hidden;
            cursor: pointer; transition: transform 0.2s, box-shadow 0.2s;
            position: relative;
        }
        .image-card:hover { transform: translateY(-3px); box-shadow: 0 5px 20px rgba(0,212,255,0.3); }
        .image-card.selected { box-shadow: 0 0 0 3px #00d4ff; }
        .image-card img { width: 100%; height: 150px; object-fit: cover; }
        .image-card .info { padding: 10px; font-size: 0.85em; }
        .image-card .filename { color: #888; word-break: break-all; }
        .verified-badge {
            position: absolute; top: 8px; right: 8px;
            background: #28a745; color: white; padding: 4px 8px;
            border-radius: 4px; font-size: 0.75em; font-weight: bold;
        }
        .verify-btn {
            background: #28a745; color: white; padding: 5px 10px;
            border: none; border-radius: 3px; cursor: pointer;
            font-size: 0.9em; margin-top: 5px;
        }
        .verify-btn:hover { background: #218838; }
        .modal {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.9); z-index: 1000; justify-content: center; align-items: center;
        }
        .modal.active { display: flex; }
        .modal-content {
            background: #16213e; padding: 20px; border-radius: 10px;
            max-width: 90%; max-height: 90%; overflow: auto; text-align: center;
        }
        .modal-content img { max-width: 100%; max-height: 60vh; margin-bottom: 20px; }
        .modal-actions { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
        .percentage-grid {
            display: grid; grid-template-columns: repeat(10, 1fr); gap: 5px;
            margin: 15px 0;
        }
        .pct-btn {
            padding: 8px; font-size: 0.9em; background: #0f3460;
            border: none; border-radius: 3px; color: #eee; cursor: pointer;
        }
        .pct-btn:hover { background: #00d4ff; color: #1a1a2e; }
        .pct-btn.current { background: #e94560; }
        .close-btn {
            position: absolute; top: 20px; right: 30px; font-size: 2em;
            color: #eee; cursor: pointer;
        }
        .toast {
            position: fixed; bottom: 20px; right: 20px; padding: 15px 25px;
            background: #00d4ff; color: #1a1a2e; border-radius: 5px;
            font-weight: bold; display: none; z-index: 1001;
        }
        .toast.error { background: #e94560; color: white; }
        .toast.show { display: block; }
        .empty-state { text-align: center; padding: 50px; color: #666; }
        .percentage-selector { margin-bottom: 15px; }
        .percentage-selector label { margin-right: 10px; }
        /* Comparison Stats specific styles */
        .value-table {
            width: 100%; border-collapse: collapse; background: #16213e;
            border-radius: 8px; overflow: hidden; margin-bottom: 20px;
        }
        .value-table th, .value-table td {
            padding: 12px; text-align: left; border-bottom: 1px solid #0f3460;
        }
        .value-table th { background: #0f3460; color: #00d4ff; font-weight: bold; }
        .value-table tr:hover { background: #0f3460; cursor: pointer; }
        .value-table tr.low-agreement { background: #3d1a1a; }
        .value-table tr.medium-agreement { background: #3d2a1a; }
        .comparison-record {
            background: #16213e; padding: 15px; border-radius: 8px;
            margin-bottom: 10px; display: flex; gap: 15px; align-items: center;
        }
        .comparison-record img { width: 150px; height: auto; border-radius: 4px; cursor: pointer; }
        .comparison-record .details { flex: 1; }
        .comparison-record .details div { margin-bottom: 5px; }
        .comparison-record .agreement-badge {
            padding: 5px 10px; border-radius: 4px; font-weight: bold;
        }
        .error-pattern-table {
            width: 100%; border-collapse: collapse; background: #16213e;
            border-radius: 8px; overflow: hidden;
        }
        .error-pattern-table th, .error-pattern-table td {
            padding: 10px; text-align: left; border-bottom: 1px solid #0f3460;
        }
        .error-pattern-table th { background: #0f3460; color: #00d4ff; font-weight: bold; }
        .error-pattern-table .llm-error { color: #ff6b6b; }
        .error-pattern-table .template-error { color: #ffa500; }
        .comparison-record .agreement-badge.agree { background: #28a745; }
        .comparison-record .agreement-badge.disagree { background: #e94560; }
        .summary-cards {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin-bottom: 20px;
        }
        .summary-card {
            background: #16213e; padding: 20px; border-radius: 8px; text-align: center;
        }
        .summary-card .value { font-size: 2.5em; font-weight: bold; color: #00d4ff; }
        .summary-card .label { color: #888; margin-top: 5px; }
        .summary-card.green .value { color: #28a745; }
        .summary-card.yellow .value { color: #ffc107; }
        .summary-card.red .value { color: #e94560; }
    </style>
</head>
<body>
    <h1>Training & Comparison Review</h1>

    <div class="tabs">
        <div class="tab active" onclick="switchTab('training')">Training Images</div>
        <div class="tab" onclick="switchTab('comparisons')">Comparison Stats</div>
    </div>

    <!-- Training Images Tab -->
    <div id="training-tab" class="tab-content active">
    <div class="stats" id="stats">
        <div class="stat">
            <div class="stat-value" id="coverage">-</div>
            <div class="stat-label">Coverage</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="total-images">-</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="values-covered">-</div>
            <div class="stat-label">Values Covered</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="comparison-mode">-</div>
            <div class="stat-label">Comparison Mode</div>
        </div>
    </div>

    <div class="controls">
        <label>View percentage:</label>
        <select id="percentage-filter">
            <option value="all">All with images</option>
        </select>
        <label style="margin-left: 20px;">Filter:</label>
        <select id="verification-filter">
            <option value="unverified">Unverified only</option>
            <option value="all">All images</option>
        </select>
        <button onclick="loadImages()">Refresh</button>
        <button onclick="verifyAllVisible()" style="background: #28a745;">Verify All Visible</button>
        <label style="margin-left: 20px; display: flex; align-items: center; gap: 5px; cursor: pointer;">
            <input type="checkbox" id="preprocess-toggle" onchange="loadImages()" style="width: 18px; height: 18px; cursor: pointer;">
            <span title="Show images with histogram equalization (as classifier sees them)">Classifier View</span>
        </label>
    </div>

    <div class="grid" id="image-grid"></div>
    </div>

    <!-- Comparison Stats Tab -->
    <div id="comparisons-tab" class="tab-content">
        <div class="summary-cards">
            <div class="summary-card" id="total-comparisons-card">
                <div class="value" id="total-comparisons">-</div>
                <div class="label">Total Comparisons</div>
            </div>
            <div class="summary-card" id="agreement-rate-card">
                <div class="value" id="agreement-rate">-</div>
                <div class="label">Agreement Rate</div>
            </div>
            <div class="summary-card" id="llm-accuracy-card">
                <div class="value" id="llm-accuracy">-</div>
                <div class="label">LLM Accuracy</div>
                <div style="font-size: 0.75em; color: #888; margin-top: 5px;" id="llm-accuracy-count">-</div>
            </div>
            <div class="summary-card" id="template-accuracy-card">
                <div class="value" id="template-accuracy">-</div>
                <div class="label">Template Accuracy</div>
                <div style="font-size: 0.75em; color: #888; margin-top: 5px;" id="template-accuracy-count">-</div>
            </div>
            <div class="summary-card">
                <div class="value" id="recent-disagreements">-</div>
                <div class="label">Recent Disagreements (24h)</div>
            </div>
        </div>

        <h2 style="margin-bottom: 15px; color: #00d4ff;">Per-Value Agreement</h2>
        <table class="value-table">
            <thead>
                <tr>
                    <th>Value</th>
                    <th>Count</th>
                    <th>Agreements</th>
                    <th>Disagreements</th>
                    <th>Agreement Rate</th>
                </tr>
            </thead>
            <tbody id="value-table-body">
                <tr><td colspan="5" style="text-align: center; color: #666;">Loading...</td></tr>
            </tbody>
        </table>

        <h2 style="margin-bottom: 15px; color: #00d4ff;">Error Patterns</h2>
        <div id="error-patterns-section" style="margin-bottom: 30px;">
            <p style="color: #888; font-style: italic;">No error patterns available yet. Verify some training images to see which method makes which errors.</p>
        </div>

        <h2 style="margin-bottom: 15px; color: #00d4ff;">Comparison Records</h2>
        <div class="controls" style="margin-bottom: 15px;">
            <select id="comparison-filter">
                <option value="all">All Records</option>
                <option value="disagreements">Disagreements Only</option>
            </select>
            <select id="comparison-value-filter">
                <option value="">All Values</option>
            </select>
            <button onclick="loadComparisonRecords()">Refresh</button>
        </div>

        <div id="comparison-records-list"></div>
    </div>

    <div class="modal" id="modal">
        <span class="close-btn" onclick="closeModal()">&times;</span>
        <div class="modal-content">
            <img id="modal-image" src="" alt="Training image">
            <p id="modal-info"></p>
            <p class="percentage-selector">
                <label>Reclassify to:</label>
            </p>
            <div class="percentage-grid" id="percentage-grid"></div>
            <div class="modal-actions">
                <button class="danger" onclick="deleteImage()">Delete Image</button>
                <button onclick="closeModal()">Cancel</button>
            </div>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        let currentImage = null;
        let allImages = {};

        async function loadStats() {
            try {
                const res = await fetch('/training/status');
                const data = await res.json();
                if (data.success) {
                    document.getElementById('coverage').textContent = data.coverage_percentage.toFixed(1) + '%';
                    document.getElementById('total-images').textContent = data.total_images;
                    document.getElementById('values-covered').textContent = data.values_with_examples + '/101';
                    document.getElementById('comparison-mode').textContent = data.comparison_mode_active ? 'Active' : 'Inactive';
                }
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        }

        async function loadImages() {
            try {
                const verificationFilter = document.getElementById('verification-filter').value;
                const res = await fetch(`/training/images?filter=${verificationFilter}`);
                const data = await res.json();
                if (data.success) {
                    allImages = data.images;
                    updateFilter();
                    renderImages();
                }
            } catch (e) {
                console.error('Failed to load images:', e);
            }
        }

        function updateFilter() {
            const select = document.getElementById('percentage-filter');
            const currentValue = select.value;
            select.innerHTML = '<option value="all">All with images</option>';

            // Sort categories: "invalid" first, then numeric 0-100
            Object.keys(allImages).sort((a, b) => {
                if (a === 'invalid') return -1;
                if (b === 'invalid') return 1;
                return parseInt(a) - parseInt(b);
            }).forEach(cat => {
                const images = allImages[cat];
                const count = images.length;
                const opt = document.createElement('option');
                opt.value = cat;
                const displayLabel = cat === 'invalid' ? 'invalid' : cat + '%';
                opt.textContent = displayLabel + ' (' + count + ' images)';
                select.appendChild(opt);
            });

            select.value = currentValue;
        }

        function getImageUrl(category, filename) {
            const preprocess = document.getElementById('preprocess-toggle').checked;
            return `/training/image/${category}/${filename}${preprocess ? '?preprocess=true' : ''}`;
        }

        function renderImages() {
            const grid = document.getElementById('image-grid');
            const filter = document.getElementById('percentage-filter').value;

            let html = '';
            let categories = filter === 'all'
                ? Object.keys(allImages).sort((a, b) => {
                    // Sort: "invalid" first, then numeric 0-100
                    if (a === 'invalid') return -1;
                    if (b === 'invalid') return 1;
                    return parseInt(a) - parseInt(b);
                })
                : [filter];

            if (categories.length === 0 || (categories.length === 1 && !allImages[categories[0]])) {
                grid.innerHTML = '<div class="empty-state">No images found</div>';
                return;
            }

            categories.forEach(cat => {
                if (!allImages[cat]) return;
                allImages[cat].forEach(imageData => {
                    const filename = imageData.filename;
                    const isVerified = imageData.verified;
                    const displayLabel = cat === 'invalid' ? 'invalid' : cat + '%';
                    const verifiedBadge = isVerified ? '<div class="verified-badge">✓ Verified</div>' : '';
                    const verifyButton = !isVerified ? `<button class="verify-btn" onclick="event.stopPropagation(); verifyImage('${cat}', '${filename}')">Verify</button>` : '';

                    html += `
                        <div class="image-card" onclick="openModal('${cat}', '${filename}')">
                            ${verifiedBadge}
                            <img src="${getImageUrl(cat, filename)}" alt="${displayLabel}" loading="lazy">
                            <div class="info">
                                <strong>${displayLabel}</strong>
                                <div class="filename">${filename}</div>
                                ${verifyButton}
                            </div>
                        </div>
                    `;
                });
            });

            grid.innerHTML = html;
        }

        function openModal(category, filename) {
            currentImage = { category, filename };
            document.getElementById('modal-image').src = getImageUrl(category, filename);
            document.getElementById('modal-info').textContent = `Current: ${category} | ${filename}`;

            // Build percentage grid with "invalid" option
            let gridHtml = '';

            // Add "invalid" button first
            const isCurrentInvalid = category === 'invalid';
            gridHtml += `<button class="pct-btn ${isCurrentInvalid ? 'current' : ''}"
                onclick="${isCurrentInvalid ? '' : `reclassify('invalid')`}"
                ${isCurrentInvalid ? 'disabled' : ''}
                style="grid-column: span 2; background: #e94560;">invalid</button>`;

            // Add percentage buttons 0-100
            for (let i = 0; i <= 100; i++) {
                const isCurrent = i == category;
                gridHtml += `<button class="pct-btn ${isCurrent ? 'current' : ''}"
                    onclick="${isCurrent ? '' : `reclassify(${i})`}"
                    ${isCurrent ? 'disabled' : ''}>${i}</button>`;
            }
            document.getElementById('percentage-grid').innerHTML = gridHtml;

            document.getElementById('modal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
            currentImage = null;
        }

        async function reclassify(toCategory) {
            if (!currentImage) return;

            try {
                const res = await fetch('/training/reclassify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: `filename=${encodeURIComponent(currentImage.filename)}&from_category=${currentImage.category}&to_category=${toCategory}`
                });
                const data = await res.json();

                if (data.success) {
                    showToast(`Moved to ${toCategory}`);
                    closeModal();
                    loadImages();
                    loadStats();
                } else {
                    showToast(data.error || 'Failed to reclassify', true);
                }
            } catch (e) {
                showToast('Error: ' + e.message, true);
            }
        }

        async function deleteImage() {
            if (!currentImage) return;
            if (!confirm('Delete this image?')) return;

            try {
                const res = await fetch(`/training/image/${currentImage.category}/${currentImage.filename}`, {
                    method: 'DELETE'
                });
                const data = await res.json();

                if (data.success) {
                    showToast('Image deleted');
                    closeModal();
                    loadImages();
                    loadStats();
                } else {
                    showToast(data.error || 'Failed to delete', true);
                }
            } catch (e) {
                showToast('Error: ' + e.message, true);
            }
        }

        function showToast(message, isError = false) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast show' + (isError ? ' error' : '');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        async function verifyImage(category, filename) {
            try {
                const res = await fetch(`/training/verify/${category}/${filename}`, {
                    method: 'POST'
                });
                const data = await res.json();

                if (data.success) {
                    showToast('Image verified');
                    loadImages();
                    loadStats();
                } else {
                    showToast(data.error || 'Failed to verify', true);
                }
            } catch (e) {
                showToast('Error: ' + e.message, true);
            }
        }

        async function verifyAllVisible() {
            try {
                const filter = document.getElementById('percentage-filter').value;

                // Determine which categories are visible
                let categoriesToVerify = [];
                if (filter === 'all') {
                    categoriesToVerify = Object.keys(allImages);
                } else {
                    categoriesToVerify = [filter];
                }

                if (categoriesToVerify.length === 0) {
                    showToast('No images to verify', true);
                    return;
                }

                const res = await fetch('/training/verify-all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: `categories=${encodeURIComponent(JSON.stringify(categoriesToVerify))}`
                });
                const data = await res.json();

                if (data.success) {
                    showToast(data.message);
                    loadImages();
                    loadStats();
                } else {
                    showToast(data.error || 'Failed to verify all', true);
                }
            } catch (e) {
                showToast('Error: ' + e.message, true);
            }
        }

        // Event listeners
        document.getElementById('percentage-filter').addEventListener('change', renderImages);
        document.getElementById('verification-filter').addEventListener('change', loadImages);
        document.getElementById('modal').addEventListener('click', (e) => {
            if (e.target.id === 'modal') closeModal();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
        });

        // Tab switching
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));

            // Show selected tab
            if (tabName === 'training') {
                document.getElementById('training-tab').classList.add('active');
                event.target.classList.add('active');
            } else if (tabName === 'comparisons') {
                document.getElementById('comparisons-tab').classList.add('active');
                event.target.classList.add('active');
                loadComparisonStats();
                loadComparisonRecords();
            }
        }

        // Load comparison statistics
        async function loadComparisonStats() {
            try {
                const res = await fetch('/comparisons/stats');
                const data = await res.json();

                if (data.success) {
                    // Update summary cards
                    document.getElementById('total-comparisons').textContent = data.total_comparisons;
                    document.getElementById('agreement-rate').textContent = (data.agreement_rate * 100).toFixed(1) + '%';
                    document.getElementById('recent-disagreements').textContent = data.recent_disagreements;

                    // Color-code agreement rate
                    const rateCard = document.getElementById('agreement-rate-card');
                    const rate = data.agreement_rate;
                    rateCard.className = 'summary-card';
                    if (rate >= 0.9) {
                        rateCard.classList.add('green');
                    } else if (rate >= 0.7) {
                        rateCard.classList.add('yellow');
                    } else {
                        rateCard.classList.add('red');
                    }

                    // Update accuracy cards
                    if (data.accuracy && data.accuracy.llm.total_verified > 0) {
                        const llmAcc = data.accuracy.llm.accuracy_rate;
                        const templateAcc = data.accuracy.template.accuracy_rate;
                        const verifiedCount = data.accuracy.llm.total_verified;

                        document.getElementById('llm-accuracy').textContent = (llmAcc * 100).toFixed(1) + '%';
                        document.getElementById('llm-accuracy-count').textContent = `(based on ${verifiedCount} verified)`;
                        document.getElementById('template-accuracy').textContent = (templateAcc * 100).toFixed(1) + '%';
                        document.getElementById('template-accuracy-count').textContent = `(based on ${verifiedCount} verified)`;

                        // Color-code accuracy cards
                        const llmCard = document.getElementById('llm-accuracy-card');
                        llmCard.className = 'summary-card';
                        if (llmAcc >= 0.9) llmCard.classList.add('green');
                        else if (llmAcc >= 0.7) llmCard.classList.add('yellow');
                        else llmCard.classList.add('red');

                        const templateCard = document.getElementById('template-accuracy-card');
                        templateCard.className = 'summary-card';
                        if (templateAcc >= 0.9) templateCard.classList.add('green');
                        else if (templateAcc >= 0.7) templateCard.classList.add('yellow');
                        else templateCard.classList.add('red');
                    } else {
                        document.getElementById('llm-accuracy').textContent = '-';
                        document.getElementById('llm-accuracy-count').textContent = '(no verified data)';
                        document.getElementById('template-accuracy').textContent = '-';
                        document.getElementById('template-accuracy-count').textContent = '(no verified data)';
                    }

                    // Update error patterns section
                    const errorPatternsSection = document.getElementById('error-patterns-section');
                    if (data.error_patterns && data.error_patterns.length > 0) {
                        let html = '<table class="error-pattern-table"><thead><tr><th>Method</th><th>Predicted</th><th>Actual</th><th>Count</th></tr></thead><tbody>';
                        for (const pattern of data.error_patterns) {
                            const methodClass = pattern.method === 'llm' ? 'llm-error' : 'template-error';
                            html += `
                                <tr>
                                    <td class="${methodClass}">${pattern.method.toUpperCase()}</td>
                                    <td>${pattern.predicted}${typeof pattern.predicted === 'number' ? '%' : ''}</td>
                                    <td>${pattern.actual}${typeof pattern.actual === 'number' || !isNaN(pattern.actual) ? '%' : ''}</td>
                                    <td>${pattern.count}</td>
                                </tr>
                            `;
                        }
                        html += '</tbody></table>';
                        errorPatternsSection.innerHTML = html;
                    } else {
                        errorPatternsSection.innerHTML = '<p style="color: #888; font-style: italic;">No error patterns available yet. Verify some training images to see which method makes which errors.</p>';
                    }

                    // Update per-value table
                    const tableBody = document.getElementById('value-table-body');
                    if (Object.keys(data.by_value).length === 0) {
                        tableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #666;">No comparison data yet</td></tr>';
                    } else {
                        let html = '';
                        const sortedValues = Object.keys(data.by_value).sort((a, b) => parseInt(a) - parseInt(b));

                        for (const value of sortedValues) {
                            const stats = data.by_value[value];
                            const disagreements = stats.count - stats.agreements;
                            let rowClass = '';
                            if (stats.agreement_rate < 0.7) {
                                rowClass = 'low-agreement';
                            } else if (stats.agreement_rate < 0.9) {
                                rowClass = 'medium-agreement';
                            }

                            html += `
                                <tr class="${rowClass}" onclick="filterByValue(${value})">
                                    <td>${value}%</td>
                                    <td>${stats.count}</td>
                                    <td>${stats.agreements}</td>
                                    <td>${disagreements}</td>
                                    <td>${(stats.agreement_rate * 100).toFixed(1)}%</td>
                                </tr>
                            `;
                        }
                        tableBody.innerHTML = html;

                        // Populate value filter dropdown
                        const valueFilter = document.getElementById('comparison-value-filter');
                        valueFilter.innerHTML = '<option value="">All Values</option>';
                        for (const value of sortedValues) {
                            valueFilter.innerHTML += `<option value="${value}">${value}%</option>`;
                        }
                    }
                }
            } catch (e) {
                console.error('Failed to load comparison stats:', e);
            }
        }

        // Filter records by value
        function filterByValue(value) {
            document.getElementById('comparison-value-filter').value = value;
            loadComparisonRecords();
        }

        // Load comparison records
        async function loadComparisonRecords() {
            try {
                const filterType = document.getElementById('comparison-filter').value;
                const value = document.getElementById('comparison-value-filter').value;

                let url = `/comparisons/records?limit=50&filter=${filterType}`;
                if (value) {
                    url += `&value=${value}`;
                }

                const res = await fetch(url);
                const data = await res.json();

                if (data.success) {
                    const list = document.getElementById('comparison-records-list');

                    if (data.records.length === 0) {
                        list.innerHTML = '<div class="empty-state">No comparison records found</div>';
                    } else {
                        let html = '';
                        for (const record of data.records) {
                            const llmPct = record.gemini_percentage !== null ? record.gemini_percentage : record.groq_percentage;
                            const llmLabel = record.llm_source === 'groq' ? 'Groq' : 'Gemini';
                            const agreementClass = record.agreement ? 'agree' : 'disagree';
                            const agreementText = record.agreement ? '✓ Agree' : '✗ Disagree';

                            // Determine ground truth
                            let groundTruth = null;
                            let humanVerified = '<div><em>Not verified</em></div>';
                            let llmVerdict = '';
                            let templateVerdict = '';

                            if (record.human_verified_invalid) {
                                groundTruth = 'invalid';
                                humanVerified = '<div><strong>Human:</strong> <span style="color: #ff6b6b;">Invalid</span></div>';

                                // Check LLM accuracy
                                if (llmPct !== null) {
                                    llmVerdict = '<div style="color: #ff6b6b; font-size: 0.9em;">✗ LLM wrong</div>';
                                }

                                // Check Template accuracy
                                if (record.template_percentage !== null) {
                                    templateVerdict = '<div style="color: #ff6b6b; font-size: 0.9em;">✗ Template wrong</div>';
                                }
                            } else if (record.human_verified_percentage !== null) {
                                groundTruth = record.human_verified_percentage;
                                humanVerified = `<div><strong>Human:</strong> <span style="color: #28a745;">${groundTruth}%</span></div>`;

                                // Check LLM accuracy
                                if (llmPct === groundTruth) {
                                    llmVerdict = '<div style="color: #28a745; font-size: 0.9em;">✓ LLM correct</div>';
                                } else if (llmPct !== null) {
                                    llmVerdict = '<div style="color: #ff6b6b; font-size: 0.9em;">✗ LLM wrong</div>';
                                }

                                // Check Template accuracy
                                if (record.template_percentage === groundTruth) {
                                    templateVerdict = '<div style="color: #28a745; font-size: 0.9em;">✓ Template correct</div>';
                                } else if (record.template_percentage !== null) {
                                    templateVerdict = '<div style="color: #ff6b6b; font-size: 0.9em;">✗ Template wrong</div>';
                                }
                            }

                            html += `
                                <div class="comparison-record">
                                    <img src="/comparisons/image/${record.image_filename}"
                                         onclick="openComparisonModal(${record.id}, '${record.image_filename}')"
                                         alt="Comparison image">
                                    <div class="details">
                                        <div><strong>${llmLabel}:</strong> ${llmPct !== null ? llmPct + '%' : 'Failed'} ${llmVerdict}</div>
                                        <div><strong>Template:</strong> ${record.template_percentage !== null ? record.template_percentage + '% (conf: ' + record.template_confidence.toFixed(3) + ')' : 'Failed'} ${templateVerdict}</div>
                                        ${humanVerified}
                                        <div style="font-size: 0.85em; color: #888;">${new Date(record.timestamp * 1000).toLocaleString()}</div>
                                    </div>
                                    <div class="agreement-badge ${agreementClass}">${agreementText}</div>
                                </div>
                            `;
                        }
                        list.innerHTML = html;
                    }
                }
            } catch (e) {
                console.error('Failed to load comparison records:', e);
            }
        }

        // Open comparison modal (reuse existing modal)
        function openComparisonModal(recordId, filename) {
            document.getElementById('modal-image').src = `/comparisons/image/${filename}`;
            document.getElementById('modal-info').textContent = `Comparison Record #${recordId}`;
            document.getElementById('percentage-grid').innerHTML = '<p>Human verification for comparisons coming soon</p>';
            document.getElementById('modal').classList.add('active');
        }

        // Update comparison filter listener
        document.getElementById('comparison-filter').addEventListener('change', loadComparisonRecords);
        document.getElementById('comparison-value-filter').addEventListener('change', loadComparisonRecords);

        // Initial load
        loadStats();
        loadImages();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    uvicorn.run("main:app", host=host, port=port, log_level="info")
