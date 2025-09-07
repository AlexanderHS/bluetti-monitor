import os
import requests
from urllib.parse import urljoin
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import uvicorn
import cv2
import numpy as np
import pytesseract
from io import BytesIO
from fastapi.responses import Response
import asyncio
import aiosqlite
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import json
import logging
from pathlib import Path

load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
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
            await db.execute("""
                CREATE TABLE IF NOT EXISTS battery_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    battery_percentage INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    ocr_method TEXT,
                    total_attempts INTEGER,
                    raw_vote_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON battery_readings(timestamp)
            """)
            await db.commit()
    
    async def insert_reading(self, battery_percentage: int, confidence: float, 
                           ocr_method: str = None, total_attempts: int = None, 
                           raw_vote_data: dict = None):
        """Insert a new battery reading"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO battery_readings 
                (timestamp, battery_percentage, confidence, ocr_method, total_attempts, raw_vote_data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                battery_percentage,
                confidence,
                ocr_method,
                total_attempts,
                json.dumps(raw_vote_data) if raw_vote_data else None
            ))
            await db.commit()
            logger.info(f"Stored battery reading: {battery_percentage}% (confidence: {confidence}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    async def get_recent_readings(self, limit: int = 10) -> List[Dict]:
        """Get recent battery readings"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT timestamp, battery_percentage, confidence, ocr_method, total_attempts
                FROM battery_readings 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,)) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "timestamp": row[0],
                        "battery_percentage": row[1],
                        "confidence": row[2],
                        "ocr_method": row[3],
                        "total_attempts": row[4],
                        "seconds_ago": int(datetime.now().timestamp() - row[0])
                    }
                    for row in rows
                ]

# Global database instance
db = BatteryDatabase(os.getenv("DATABASE_PATH", "./data/battery_readings.db"))

def get_crop_coordinates():
    """Get battery crop coordinates from environment"""
    return {
        'x1': int(os.getenv('BATTERY_CROP_X1', 400)),
        'y1': int(os.getenv('BATTERY_CROP_Y1', 200)), 
        'x2': int(os.getenv('BATTERY_CROP_X2', 500)),
        'y2': int(os.getenv('BATTERY_CROP_Y2', 250))
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
            if "premature end of data segment" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                if attempt < max_retries - 1:
                    logger.debug(f"Recoverable network error on attempt {attempt + 1}: {e}")
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
    """Initialize database and start background tasks"""
    global startup_time
    startup_time = datetime.now().timestamp()
    
    logger.info("Starting up Bluetti Monitor API")
    
    # Initialize database
    await db.init_db()
    logger.info("Database initialized")
    
    # Start background polling task
    task = asyncio.create_task(background_polling_task())
    logger.info("Background polling task started")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down...")
    task.cancel()

app = FastAPI(
    title=os.getenv("API_TITLE", "Bluetti Monitor API"),
    description="Monitor Bluetti solar generator battery status via ESP32 webcam",
    version="1.0.0",
    lifespan=lifespan
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
                "last_reading_age_minutes": round((datetime.now().timestamp() - last_reading["timestamp"]) / 60, 1),
                "battery_percentage": last_reading["battery_percentage"]
            }
        
        # No recent readings - check uptime
        if startup_time:
            uptime_minutes = (datetime.now().timestamp() - startup_time) / 60
            
            if uptime_minutes > 30:
                # We've been running for more than 30 minutes with no data - unhealthy
                return {
                    "status": "unhealthy",
                    "message": f"No battery readings in last 30 minutes (uptime: {uptime_minutes:.1f} minutes)",
                    "uptime_minutes": round(uptime_minutes, 1)
                }
            else:
                # We haven't been running long enough to expect data yet - healthy but no data
                return {
                    "status": "healthy",
                    "message": f"Bluetti Monitor is running but no data yet (uptime: {uptime_minutes:.1f} minutes)",
                    "uptime_minutes": round(uptime_minutes, 1)
                }
        else:
            # Startup time not available - assume healthy for now
            return {
                "status": "healthy",
                "message": "Bluetti Monitor is running (startup time not available)"
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "error": str(e)
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
                "last_reading": None
            }
        
        latest = recent_readings[0]
        
        # Calculate change per minute if we have multiple readings
        change_per_minute = None
        change_direction = None
        readings_analyzed = 0
        
        if len(recent_readings) >= 2:
            # Use up to last 5 readings for change calculation
            readings_for_change = recent_readings[:min(5, len(recent_readings))]
            readings_analyzed = len(readings_for_change)
            
            # Calculate time span and percentage change
            oldest = readings_for_change[-1]
            newest = readings_for_change[0]
            
            time_diff_minutes = (newest["timestamp"] - oldest["timestamp"]) / 60
            percentage_diff = newest["battery_percentage"] - oldest["battery_percentage"]
            
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
            "total_recent_readings": len(recent_readings)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get battery status: {str(e)}"
        }

@app.get("/status/history")
async def get_battery_history(limit: int = 20):
    """Get historical battery readings"""
    try:
        readings = await db.get_recent_readings(limit=limit)
        return {
            "success": True,
            "readings": readings,
            "count": len(readings)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get battery history: {str(e)}"
        }

@app.get("/recommendations")
async def get_power_recommendations():
    """Get power management recommendations based on recent battery status"""
    try:
        # Get readings from the last 30 minutes
        thirty_minutes_ago = datetime.now().timestamp() - (30 * 60)
        
        # Get recent readings and filter to last 30 minutes
        recent_readings = await db.get_recent_readings(limit=100)  # Get more to ensure we have enough
        readings_last_30min = [
            reading for reading in recent_readings 
            if reading["timestamp"] >= thirty_minutes_ago
        ]
        
        # Case 1: No readings in the last 30 minutes - we're blind
        if not readings_last_30min:
            return {
                "success": True,
                "status": "blind",
                "message": "No recent readings in the last 30 minutes",
                "recommendations": {
                    "input": "turn_off",
                    "output_1": "turn_off", 
                    "output_2": "turn_off"
                },
                "reasoning": "No recent battery data available - turning off all outputs and input for safety",
                "last_reading_age_minutes": None,
                "battery_percentage": None
            }
        
        # Case 2: We have recent readings - analyze battery percentage
        latest_reading = readings_last_30min[0]  # Most recent reading
        battery_percentage = latest_reading["battery_percentage"]
        last_reading_age_minutes = (datetime.now().timestamp() - latest_reading["timestamp"]) / 60
        
        # Determine recommendations based on battery percentage
        if battery_percentage < 20:
            # Below 20% - turn off outputs, turn on input (charge)
            recommendations = {
                "input": "turn_on",
                "output_1": "turn_off",
                "output_2": "turn_off"
            }
            reasoning = f"Battery at {battery_percentage}% - critical low, charging needed"
            
        elif 20 <= battery_percentage < 60:
            # Between 20-60% - turn off input, turn off outputs (conservation)
            recommendations = {
                "input": "turn_off",
                "output_1": "turn_off", 
                "output_2": "turn_off"
            }
            reasoning = f"Battery at {battery_percentage}% - low, conserving power"
            
        elif 60 <= battery_percentage < 80:
            # Between 60-80% - turn off input, turn on one output (moderate drain)
            recommendations = {
                "input": "turn_off",
                "output_1": "turn_on",
                "output_2": "turn_off"
            }
            reasoning = f"Battery at {battery_percentage}% - moderate level, using one output for moderate drain"
            
        else:  # battery_percentage >= 80
            # Above 80% - turn off input, turn on both outputs (rapid drain)
            recommendations = {
                "input": "turn_off",
                "output_1": "turn_on",
                "output_2": "turn_on"
            }
            reasoning = f"Battery at {battery_percentage}% - high level, using both outputs for rapid drain"
        
        return {
            "success": True,
            "status": "active",
            "message": "Recommendations based on recent battery status",
            "recommendations": recommendations,
            "reasoning": reasoning,
            "battery_percentage": battery_percentage,
            "last_reading_age_minutes": round(last_reading_age_minutes, 1),
            "readings_in_last_30min": len(readings_last_30min),
            "confidence": latest_reading.get("confidence", None),
            "last_reading_timestamp": latest_reading["timestamp"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate recommendations: {str(e)}"
        }

@app.get("/camera/status")
async def camera_status():
    webcam_url = os.getenv("WEBCAM_URL")
    capture_endpoint = os.getenv("WEBCAM_CAPTURE_ENDPOINT", "/capture")
    
    if not webcam_url:
        return {
            "status": "error",
            "message": "WEBCAM_URL not configured"
        }
    
    capture_url = urljoin(webcam_url, capture_endpoint)
    
    try:
        response = requests.get(capture_url, timeout=10)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type or len(response.content) > 1000:
                return {
                    "status": "ok",
                    "message": "Camera operational",
                    "capture_url": capture_url
                }
        
        return {
            "status": "error", 
            "message": "Image capture failed",
            "capture_url": capture_url
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Camera unreachable: {str(e)}",
            "capture_url": capture_url
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
            'x1': x1 if x1 is not None else default_coords['x1'],
            'y1': y1 if y1 is not None else default_coords['y1'],
            'x2': x2 if x2 is not None else default_coords['x2'],
            'y2': y2 if y2 is not None else default_coords['y2']
        }
        
        # Validate coordinates
        h, w = img.shape[:2]
        coords['x1'] = max(0, min(coords['x1'], w-1))
        coords['y1'] = max(0, min(coords['y1'], h-1))
        coords['x2'] = max(coords['x1']+1, min(coords['x2'], w))
        coords['y2'] = max(coords['y1']+1, min(coords['y2'], h))
        
        # Crop image
        cropped = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
        
        # Encode as JPEG and return
        _, buffer = cv2.imencode('.jpg', cropped)
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
        
        # Get crop coordinates, crop, and flip
        coords = get_crop_coordinates()
        cropped = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
        flipped = cv2.flip(cropped, 1)  # Flip horizontally
        
        # Encode as JPEG and return
        _, buffer = cv2.imencode('.jpg', flipped)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        return {"error": f"Flip failed: {str(e)}"}

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
                "screen_analysis": screen_analysis
            }
        
        # Get crop coordinates, crop, and flip
        coords = get_crop_coordinates()
        cropped = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
        flipped = cv2.flip(cropped, 1)
        
        # Apply the optimal preprocessing we found through testing
        # Convert to grayscale and upscale (raw preprocessing works best)
        gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upscaled = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        
        # Apply the optimal threshold: 160 with invert (from your testing)
        _, optimal_binary = cv2.threshold(upscaled, 160, 255, cv2.THRESH_BINARY_INV)
        
        # Use only the best performing configs (PSM 6 & 7 based on debug results)
        configs = [
            ('psm_6', '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'),
            ('psm_7', '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789')
        ]
        
        results = []
        for name, config in configs:
            try:
                raw_text = pytesseract.image_to_string(optimal_binary, config=config).strip()
                digits_only = ''.join(c for c in raw_text if c.isdigit())
                
                if digits_only and digits_only.isdigit() and 0 <= int(digits_only) <= 100:
                    # Score based on length preference (2 digits ideal) and PSM preference
                    length_score = 10 - abs(len(digits_only) - 2)
                    psm_bonus = 2 if name == 'psm_6' else 1  # Slight preference for PSM 6
                    
                    results.append({
                        'text': digits_only,
                        'confidence': length_score + psm_bonus,
                        'method': name,
                        'raw_ocr': raw_text
                    })
            except:
                continue
        
        # Pick best result
        if results:
            best = max(results, key=lambda x: x['confidence'])
            text = best['text']
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
            "battery_percentage": int(text) if text.isdigit() and len(text) <= 3 else None,
            "attempts_tried": len(configs)
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
        
        # Get crop coordinates, crop, and flip
        coords = get_crop_coordinates()
        cropped = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
        flipped = cv2.flip(cropped, 1)
        
        # Let's debug step by step
        if method == "raw":
            # Just the cropped and flipped image, upscaled
            gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            processed = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            
        elif method == "blue_channel":
            # Try using just the blue channel (white text on blue background)
            height, width = flipped.shape[:2]
            blue_channel = flipped[:, :, 0]  # Blue channel in BGR
            processed = cv2.resize(blue_channel, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            
        elif method == "enhanced":
            # Show the enhanced version before thresholding
            gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            resized = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            processed = clahe.apply(blurred)
            
        elif method == "invert":
            # Simple invert of enhanced image
            gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            resized = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(blurred)
            processed = cv2.bitwise_not(enhanced)  # Invert: white becomes black
            
        # Thresholding methods
        else:
            gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            resized = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(blurred)
            
            if method == "otsu":
                _, processed = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == "adaptive":
                processed = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
            elif method == "manual":
                _, processed = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY_INV)  # Lower threshold
            else:
                return {"error": f"Unknown method: {method}. Use 'raw', 'enhanced', 'blue_channel', 'invert', 'otsu', 'adaptive', or 'manual'"}
        
        # Encode and return
        _, buffer = cv2.imencode('.jpg', processed)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

@app.get("/capture/threshold")
async def capture_threshold(threshold: int = 180, invert: bool = True, preprocess: str = "enhanced"):
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
        
        # Get crop coordinates, crop, and flip
        coords = get_crop_coordinates()
        cropped = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
        flipped = cv2.flip(cropped, 1)
        
        # Apply requested preprocessing
        if preprocess == "raw":
            gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            processed_img = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            
        elif preprocess == "blue_channel":
            height, width = flipped.shape[:2]
            blue_channel = flipped[:, :, 0]  # Blue channel
            processed_img = cv2.resize(blue_channel, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            
        elif preprocess == "enhanced":
            gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            resized = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            processed_img = clahe.apply(blurred)
            
        else:
            return {"error": f"Unknown preprocess: {preprocess}. Use 'raw', 'enhanced', or 'blue_channel'"}
        
        # Apply manual threshold with specified value
        if invert:
            _, thresholded = cv2.threshold(processed_img, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresholded = cv2.threshold(processed_img, threshold, 255, cv2.THRESH_BINARY)
        
        # Encode as JPEG and return
        _, buffer = cv2.imencode('.jpg', thresholded)
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
                "message": "Screen is off - OCR debug skipped"
            }
        
        # Get crop coordinates, crop, and flip
        coords = get_crop_coordinates()
        cropped = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
        flipped = cv2.flip(cropped, 1)
        
        # Apply preprocessing with specified threshold
        gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upscaled = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(upscaled, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Try all OCR configs and collect detailed results
        configs = [
            ('psm_6', '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789%'),   # Allow % symbol
            ('psm_7', '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789%'),
            ('psm_8', '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789%'),
            ('psm_13', '--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789%'),
            ('psm_6_digits_only', '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'),
            ('psm_7_digits_only', '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'),
            ('psm_8_digits_only', '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'),
            ('psm_13_digits_only', '--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789')
        ]
        
        ocr_results = []
        for name, config in configs:
            try:
                raw_text = pytesseract.image_to_string(binary, config=config).strip()
                
                # Extract just the digits for battery percentage
                digits_only = ''.join(c for c in raw_text if c.isdigit())
                
                ocr_results.append({
                    "config_name": name,
                    "raw_text": raw_text,
                    "raw_text_repr": repr(raw_text),  # Shows escape chars
                    "digits_only": digits_only,
                    "is_valid_percentage": digits_only.isdigit() and len(digits_only) <= 3 and 0 <= int(digits_only) <= 100 if digits_only else False,
                    "percentage_value": int(digits_only) if digits_only and digits_only.isdigit() else None
                })
            except Exception as e:
                ocr_results.append({
                    "config_name": name,
                    "error": str(e)
                })
        
        # Find the most consistent digit result
        digit_results = [r for r in ocr_results if r.get('digits_only') and r.get('is_valid_percentage')]
        if digit_results:
            # Count frequency of each result
            from collections import Counter
            digit_counts = Counter(r['digits_only'] for r in digit_results)
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
            "total_attempts": len(configs)
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
                "message": "Screen is off - advanced OCR skipped"
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
                    capture_details.append({"capture": capture_num + 1, "error": "Failed to decode image"})
                    continue
                
                # Get crop coordinates, crop, and flip
                coords = get_crop_coordinates()
                cropped = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
                flipped = cv2.flip(cropped, 1)
                gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape
                upscaled = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
                
                capture_results = []
                
                # Test all threshold/PSM combinations
                for threshold in thresholds:
                    _, binary = cv2.threshold(upscaled, threshold, 255, cv2.THRESH_BINARY_INV)
                    
                    for psm_mode in psm_modes:
                        try:
                            config = f'--oem 3 --psm {psm_mode} -c tessedit_char_whitelist=0123456789%'
                            raw_text = pytesseract.image_to_string(binary, config=config).strip()
                            
                            if raw_text:  # Not empty
                                digits_only = ''.join(c for c in raw_text if c.isdigit())
                                
                                if digits_only:
                                    # Smart digit handling with special cases
                                    if len(digits_only) == 4 and digits_only.startswith("100"):
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
                                        capture_results.append({
                                            "threshold": threshold,
                                            "psm": psm_mode,
                                            "raw_text": raw_text,
                                            "digits_only": digits_only,
                                            "percentage": percentage_candidate,
                                            "source": source
                                        })
                                        all_results.append(percentage_candidate)
                        except:
                            continue
                
                capture_details.append({
                    "capture": capture_num + 1,
                    "results_count": len(capture_results),
                    "results": capture_results
                })
                
            except Exception as e:
                capture_details.append({
                    "capture": capture_num + 1,
                    "error": f"Capture failed: {str(e)}"
                })
        
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
                    if percentage != other_percentage and are_conflicting(percentage, other_percentage):
                        conflicts_with.append((other_percentage, other_count))
                
                if conflicts_with:
                    conflict_detected = True
                    # Resolve conflict: prefer 2-digit sources over 1-digit
                    current_sources = [r["source"] for detail in capture_details 
                                     for r in detail.get("results", []) 
                                     if r["percentage"] == percentage]
                    
                    # Weight: 2digit > 3digit_truncated > 1digit
                    source_weights = {"2digit": 3, "3digit_truncated": 2, "1digit": 1, "4digit_100percent": 4}
                    current_weight = max(source_weights.get(src, 0) for src in current_sources)
                    
                    # Compare with conflicting results
                    should_prefer_current = True
                    for conflict_pct, conflict_count in conflicts_with:
                        conflict_sources = [r["source"] for detail in capture_details 
                                          for r in detail.get("results", []) 
                                          if r["percentage"] == conflict_pct]
                        conflict_weight = max(source_weights.get(src, 0) for src in conflict_sources)
                        
                        # If conflict has higher weight or significantly more votes, prefer it
                        if conflict_weight > current_weight or (conflict_weight == current_weight and conflict_count > count * 1.5):
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
                "resolved_vote_distribution": dict(resolved_votes) if conflict_detected else None,
                "conflict_resolution_applied": conflict_detected,
                "tied_results": ties if len(ties) > 1 else None,
                "captures_attempted": num_captures,
                "captures_succeeded": len([c for c in capture_details if "error" not in c]),
                "source_analysis": dict(source_analysis),
                "test_parameters": {
                    "thresholds": thresholds,
                    "psm_modes": psm_modes
                },
                "detailed_results": capture_details
            }
        else:
            return {
                "success": False,
                "message": "No valid OCR results obtained",
                "captures_attempted": num_captures,
                "detailed_results": capture_details
            }
            
    except Exception as e:
        return {"error": f"Advanced OCR failed: {str(e)}"}

# Background polling task
async def background_polling_task():
    """Background task that periodically captures and stores battery readings"""
    polling_interval = int(os.getenv("POLLING_INTERVAL_SECONDS", 60))
    confidence_threshold = float(os.getenv("POLLING_CONFIDENCE_THRESHOLD", 0.6))
    
    logger.info(f"Starting background polling: interval={polling_interval}s, min_confidence={confidence_threshold}")
    
    while True:
        try:
            # Use the advanced OCR endpoint logic directly
            test_image = capture_image()
            screen_analysis = analyze_screen_state(test_image)
            
            if screen_analysis.get("screen_state") == "off":
                logger.debug("Screen is off, skipping OCR polling")
                await asyncio.sleep(polling_interval)
                continue
            
            # Get advanced OCR result (simplified version of the endpoint)
            thresholds = [145, 150, 155, 160, 165, 170]
            psm_modes = [6, 7, 8, 13]
            num_captures = 3
            
            all_results = []
            
            for capture_num in range(num_captures):
                try:
                    image_bytes = capture_image()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        continue
                    
                    coords = get_crop_coordinates()
                    cropped = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
                    flipped = cv2.flip(cropped, 1)
                    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
                    height, width = gray.shape
                    upscaled = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
                    
                    for threshold in thresholds:
                        _, binary = cv2.threshold(upscaled, threshold, 255, cv2.THRESH_BINARY_INV)
                        
                        for psm_mode in psm_modes:
                            try:
                                config = f'--oem 3 --psm {psm_mode} -c tessedit_char_whitelist=0123456789%'
                                raw_text = pytesseract.image_to_string(binary, config=config).strip()
                                
                                if raw_text:
                                    digits_only = ''.join(c for c in raw_text if c.isdigit())
                                    
                                    if digits_only:
                                        if len(digits_only) == 4 and digits_only.startswith("100"):
                                            percentage_candidate = 100
                                        elif len(digits_only) == 3:
                                            percentage_candidate = int(digits_only[:2])
                                        elif len(digits_only) in [1, 2]:
                                            percentage_candidate = int(digits_only)
                                        else:
                                            continue
                                        
                                        if 0 <= percentage_candidate <= 100:
                                            all_results.append(percentage_candidate)
                            except:
                                continue
                except:
                    continue
            
            # Process results with voting
            if all_results:
                from collections import Counter
                vote_counts = Counter(all_results)
                most_common = vote_counts.most_common()
                winner = most_common[0]
                
                confidence = winner[1] / len(all_results)
                
                if confidence >= confidence_threshold:
                    # Check plausibility against last reading
                    should_store = True
                    plausibility_msg = ""
                    
                    try:
                        last_readings = await db.get_recent_readings(limit=1)
                        if last_readings:
                            last_reading = last_readings[0]
                            time_diff_minutes = (datetime.now().timestamp() - last_reading["timestamp"]) / 60
                            percentage_diff = abs(winner[0] - last_reading["battery_percentage"])
                            
                            # Plausibility thresholds (more lenient for longer time gaps)
                            if time_diff_minutes < 2 and percentage_diff > 8:
                                # Very implausible - require very high confidence
                                if confidence < 0.9:
                                    should_store = False
                                    plausibility_msg = f"implausible change rejected: {last_reading['battery_percentage']}% → {winner[0]}% in {time_diff_minutes:.1f}min (conf: {confidence})"
                            elif time_diff_minutes < 5 and percentage_diff > 15:
                                # Somewhat implausible - require higher confidence
                                if confidence < 0.85:
                                    should_store = False  
                                    plausibility_msg = f"implausible change rejected: {last_reading['battery_percentage']}% → {winner[0]}% in {time_diff_minutes:.1f}min (conf: {confidence})"
                    except:
                        # If we can't check plausibility, proceed normally
                        pass
                    
                    if should_store:
                        # Store in database
                        await db.insert_reading(
                            battery_percentage=winner[0],
                            confidence=confidence,
                            ocr_method="background_advanced",
                            total_attempts=len(all_results),
                            raw_vote_data=dict(vote_counts)
                        )
                    else:
                        logger.debug(f"Plausibility check failed: {plausibility_msg}")
                else:
                    logger.debug(f"Low confidence reading skipped: {winner[0]}% (confidence: {confidence})")
            else:
                logger.debug("No valid OCR results in background polling")
                
        except Exception as e:
            logger.error(f"Background polling error: {e}")
        
        await asyncio.sleep(polling_interval)

def analyze_screen_state(image_bytes):
    """
    Analyze image to determine if Bluetti screen is on or off.
    Returns: {"screen_state": "on"|"off", "confidence": float, "metrics": dict}
    """
    # Convert bytes to opencv image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"screen_state": "unknown", "confidence": 0.0, "error": "Failed to decode image"}
    
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
        "image_size": [img.shape[1], img.shape[0]]  # width, height
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
        "metrics": metrics
    }

@app.get("/capture/analyze")
async def capture_and_analyze():
    """
    Capture image from webcam and analyze screen state
    """
    webcam_url = os.getenv("WEBCAM_URL")
    capture_endpoint = os.getenv("WEBCAM_CAPTURE_ENDPOINT", "/capture")
    
    if not webcam_url:
        return {
            "error": "WEBCAM_URL not configured"
        }
    
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
                    "attempts": attempt + 1
                }
            
            # Analyze the captured image
            analysis = analyze_screen_state(response.content)
            
            result = {
                "success": True,
                "capture_url": capture_url,
                "timestamp": response.headers.get("date"),
                **analysis
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
                        cropped = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
                        flipped = cv2.flip(cropped, 1)
                        gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
                        height, width = gray.shape
                        upscaled = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
                        _, binary = cv2.threshold(upscaled, 160, 255, cv2.THRESH_BINARY_INV)
                        
                        # Try PSM 6 and 7 (best performers from debug)
                        for psm_mode in [6, 7]:
                            try:
                                config = f'--oem 3 --psm {psm_mode} -c tessedit_char_whitelist=0123456789'
                                raw_text = pytesseract.image_to_string(binary, config=config).strip()
                                digits_only = ''.join(c for c in raw_text if c.isdigit())
                                
                                if digits_only and digits_only.isdigit() and 0 <= int(digits_only) <= 100:
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
        "attempts": max_retries
    }

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info"
    )