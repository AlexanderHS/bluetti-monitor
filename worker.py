#!/usr/bin/env python3
"""
Bluetti Monitor Worker - Handles background polling with parallelized OCR processing
"""
import os
import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import logging
import time
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
import json

import requests
import cv2
import numpy as np
import pytesseract
from urllib.parse import urljoin
import aiosqlite
from pathlib import Path
from dotenv import load_dotenv
from gemini_ocr import gemini_ocr
from groq_ocr import groq_ocr
from switchbot_controller import switchbot_controller
from device_discovery import device_discovery
from recommendations import calculate_device_recommendations, analyze_recent_readings_for_recommendations
from template_classifier import template_classifier, log_comparison
from comparison_storage import comparison_storage
from influxdb_writer import influxdb_writer

load_dotenv()

# Configure logging with ISO 8601 timestamps
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Database management (same as main.py)
class BatteryDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            # Check if we need to migrate existing database
            await self._migrate_database_if_needed(db)
            
            await db.execute("""
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
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON battery_readings(timestamp)
            """)
            # Worker startup tracking table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS worker_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
""")
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

    async def insert_reading(self, battery_percentage: int, confidence: float,
                           ocr_method: str = None, total_attempts: int = None,
                           raw_vote_data: dict = None) -> int:
        """Insert a new battery reading with median filtering.

        Returns:
            int: The filtered battery percentage that was stored
        """
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
            await db.execute("""
                INSERT INTO battery_readings
                (timestamp, battery_percentage, confidence, ocr_method, total_attempts, raw_vote_data, raw_battery_percentage, median_filtered)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                filtered_percentage,  # Store the filtered value as the main reading
                confidence,
                ocr_method,
                total_attempts,
                json.dumps(raw_vote_data) if raw_vote_data else None,
                battery_percentage,  # Store original raw value
                is_filtered
            ))
            await db.commit()

            filter_msg = f" (filtered: {battery_percentage}% -> {filtered_percentage}%)" if is_filtered else ""
            logger.info(f"Stored battery reading: {filtered_percentage}% (confidence: {confidence}){filter_msg} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return filtered_percentage  # Return the filtered value for use in device control
    
    async def get_recent_readings(self, limit: int = 10) -> List[Dict]:
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
    
    async def set_worker_state(self, key: str, value: str):
        """Set a worker state value"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO worker_state (key, value, timestamp)
                VALUES (?, ?, ?)
            """, (key, value, datetime.now().timestamp()))
            await db.commit()
    
    async def get_worker_state(self, key: str) -> Optional[str]:
        """Get a worker state value"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT value FROM worker_state WHERE key = ?
            """, (key,)) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else None

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
        influxdb_writer.write_camera_status(reachable=False)
        raise Exception("WEBCAM_URL not configured")

    capture_url = urljoin(webcam_url, capture_endpoint)

    # Retry logic for network issues
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(capture_url, timeout=10)
            if response.status_code == 200:
                influxdb_writer.write_camera_status(reachable=True)
                return response.content
        except Exception as e:
            error_msg = str(e).lower()
            if "premature end of data segment" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                if attempt < max_retries - 1:
                    logger.debug(f"Recoverable network error on attempt {attempt + 1}: {e}")
                    time.sleep(1)
                    continue

            if attempt == max_retries - 1:
                influxdb_writer.write_camera_status(reachable=False)
                raise e
            time.sleep(0.5)

    influxdb_writer.write_camera_status(reachable=False)
    raise Exception(f"Failed to capture after {max_retries} attempts")

def analyze_screen_state(image_bytes):
    """Analyze image to determine if Bluetti screen is on or off"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"screen_state": "unknown", "confidence": 0.0, "error": "Failed to decode image"}
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    blue_percentage = (np.sum(blue_mask > 0) / (img.shape[0] * img.shape[1])) * 100
    
    if blue_percentage > 20:
        screen_state = "on"
        confidence = min(0.95, 0.7 + (blue_percentage / 50))
    elif brightness > 50 and blue_percentage > 5:
        screen_state = "on" 
        confidence = 0.8
    elif brightness < 35 and blue_percentage < 1:
        screen_state = "off"
        confidence = min(0.9, 0.6 + (1 - brightness / 50))
    else:
        screen_state = "on" if brightness > 45 else "off"
        confidence = 0.6
    
    return {
        "screen_state": screen_state,
        "confidence": round(confidence, 3),
        "metrics": {
            "brightness": float(brightness),
            "blue_percentage": float(blue_percentage),
            "image_size": [img.shape[1], img.shape[0]]
        }
    }

def process_single_ocr_task(args: Tuple) -> Optional[int]:
    """
    Process a single OCR task with specific threshold and PSM mode.
    This function runs in a separate process for parallelization.
    
    Args:
        args: Tuple of (image_data, threshold, psm_mode, crop_coords)
        
    Returns:
        Battery percentage if valid, None otherwise
    """
    try:
        image_data, threshold, psm_mode, crop_coords = args
        
        # Decode image with error handling for corrupted JPEG
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None
        except Exception:
            # Handle corrupted JPEG data silently
            return None
        
        # Crop image
        result = img[crop_coords['y1']:crop_coords['y2'], crop_coords['x1']:crop_coords['x2']]

        # Apply transformations based on environment variables
        if os.getenv("IMAGE_FLIP_HORIZONTAL", "true").lower() == "true":
            result = cv2.flip(result, 1)  # Flip horizontally
        if os.getenv("IMAGE_ROTATE_180", "false").lower() == "true":
            result = cv2.rotate(result, cv2.ROTATE_180)

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upscaled = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        
        # Apply threshold
        _, binary = cv2.threshold(upscaled, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Run OCR
        config = f'--oem 3 --psm {psm_mode} -c tessedit_char_whitelist=0123456789%'
        raw_text = pytesseract.image_to_string(binary, config=config).strip()
        
        if raw_text:
            digits_only = ''.join(c for c in raw_text if c.isdigit())
            
            if digits_only:
                # Smart digit handling
                if len(digits_only) == 4 and digits_only.startswith("100"):
                    percentage_candidate = 100
                elif len(digits_only) == 3:
                    percentage_candidate = int(digits_only[:2])
                elif len(digits_only) in [1, 2]:
                    percentage_candidate = int(digits_only)
                else:
                    return None
                
                if 0 <= percentage_candidate <= 100:
                    return percentage_candidate
        
        return None
        
    except Exception as e:
        # Log errors in the main process, not here to avoid multiprocessing issues
        return None

def capture_and_process_image_for_gemini():
    """
    Capture image from ESP32, crop and flip it for Gemini OCR (same as /capture/flip endpoint)
    
    Returns:
        bytes: Processed JPEG image data ready for Gemini
    """
    # Capture from ESP32 directly
    image_bytes = capture_image()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise Exception("Failed to decode image from ESP32")
    
    # Get crop coordinates and crop (same as /capture/flip endpoint)
    coords = get_crop_coordinates()
    result = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]

    # Apply transformations based on environment variables
    if os.getenv("IMAGE_FLIP_HORIZONTAL", "true").lower() == "true":
        result = cv2.flip(result, 1)  # Flip horizontally
    if os.getenv("IMAGE_ROTATE_180", "false").lower() == "true":
        result = cv2.rotate(result, cv2.ROTATE_180)

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', result)
    return buffer.tobytes()

async def groq_fallback_analysis_with_voting() -> Dict:
    """
    Perform OCR analysis using GROQ Llama 4 Scout as fallback
    Uses same voting mechanism as Gemini
    
    Returns:
        Dictionary with OCR results and confidence based on agreement
    """
    try:
        if not groq_ocr.is_available():
            return {
                "success": False,
                "message": "GROQ OCR not available - check API key",
                "method": "groq_fallback"
            }
        
        results = []
        total_processing_time = 0
        
        # Take 3 captures for majority voting (same as Gemini)
        for attempt in range(3):
            try:
                start_time = time.time()
                
                # Get processed image directly
                processed_image = capture_and_process_image_for_gemini()
                
                # Send to GROQ
                result = groq_ocr.analyze_image_with_groq(processed_image)
                
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                if result["success"]:
                    results.append({
                        "percentage": result["percentage"],
                        "processing_time": result.get("processing_time", 0),
                        "raw_response": result.get("raw_response")
                    })
                else:
                    logger.warning(f"GROQ OCR attempt {attempt + 1} failed: {result['error']}")
                
                # Small delay between captures
                if attempt < 2:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"GROQ OCR attempt {attempt + 1} exception: {e}")
        
        # Analyze results with majority voting (same logic as Gemini)
        if not results:
            return {
                "success": False,
                "message": "All 3 GROQ OCR attempts failed",
                "total_attempts": 3,
                "processing_time": round(total_processing_time, 2),
                "method": "groq_fallback"
            }
        
        # Count votes for each percentage, considering adjacent values as agreement
        from collections import Counter
        percentages = [r["percentage"] for r in results]

        # Group adjacent values (±1) together
        vote_groups = {}
        for p in percentages:
            # Find if this percentage belongs to an existing group
            found_group = False
            for group_key in list(vote_groups.keys()):
                if abs(p - group_key) <= 1:
                    vote_groups[group_key].append(p)
                    found_group = True
                    break
            if not found_group:
                vote_groups[p] = [p]

        # Find the largest group
        largest_group = max(vote_groups.values(), key=len)
        winner_percentage = round(sum(largest_group) / len(largest_group))  # Use average of group
        winner_votes = len(largest_group)
        total_valid_votes = len(results)

        # Calculate real confidence based on agreement
        confidence = winner_votes / total_valid_votes if total_valid_votes > 0 else 0

        # Cap confidence at 0.5 for zero readings to prevent false zeros from being accepted
        if winner_percentage == 0:
            confidence = min(confidence, 0.5)
            logger.debug("Zero reading detected - capping confidence at 0.5 to prevent false acceptance")

        # Accept if we have majority agreement (at least 2/3)
        if confidence >= 2/3:
            success = True
            raw_counts = Counter(percentages)
            if len(set(largest_group)) > 1:
                message = f"{winner_votes}/{total_valid_votes} GROQ calls agreed on {winner_percentage}% (adjacent values: {sorted(set(largest_group))})"
            else:
                message = f"{winner_votes}/{total_valid_votes} GROQ calls agreed on {winner_percentage}%"
        else:
            success = False
            raw_counts = Counter(percentages)
            message = f"No majority: got {dict(raw_counts)} - confidence too low"
            winner_percentage = None
        
        return {
            "success": success,
            "battery_percentage": winner_percentage,
            "confidence": round(confidence, 3),
            "total_attempts": 3,
            "valid_responses": total_valid_votes,
            "vote_distribution": dict(raw_counts),
            "processing_time": round(total_processing_time, 2),
            "method": "groq_fallback",
            "message": message,
            "detailed_results": results
        }
            
    except Exception as e:
        logger.error(f"GROQ OCR fallback voting failed: {e}")
        return {
            "success": False,
            "message": f"GROQ OCR voting exception: {str(e)}",
            "method": "groq_fallback",
            "total_attempts": 3
        }

async def gemini_ocr_analysis_with_voting() -> Dict:
    """
    Perform OCR analysis using Gemini vision model with 3-capture majority voting
    Includes automatic fallback to GROQ if Gemini fails completely
    Direct image processing - no HTTP self-calls
    
    Environment variables:
    - PREFER_GROQ_OCR=true: Use GROQ as primary OCR provider (for testing)
    
    Returns:
        Dictionary with OCR results and real confidence based on agreement
    """
    # Check if GROQ should be prioritized for testing
    prefer_groq = os.getenv("PREFER_GROQ_OCR", "false").lower() == "true"
    
    if prefer_groq:
        logger.info("🔄 PREFER_GROQ_OCR=true - using GROQ as primary OCR provider")
        groq_result = await groq_fallback_analysis_with_voting()
        if groq_result["success"]:
            logger.info(f"✅ GROQ primary successful: {groq_result['battery_percentage']}% (confidence: {groq_result['confidence']})")
            return groq_result
        else:
            logger.warning("GROQ primary failed - falling back to Gemini")
            # Continue to Gemini below
    
    try:
        results = []
        total_processing_time = 0
        
        # Take 3 captures for majority voting
        for attempt in range(3):
            try:
                start_time = time.time()
                
                # Get processed image directly (no HTTP calls)
                processed_image = capture_and_process_image_for_gemini()
                
                # Send directly to Gemini
                result = gemini_ocr._analyze_image_with_gemini(processed_image)
                
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                if result["success"]:
                    results.append({
                        "percentage": result["percentage"],
                        "processing_time": result.get("processing_time", 0),
                        "raw_response": result.get("raw_response")
                    })
                else:
                    logger.warning(f"Gemini OCR attempt {attempt + 1} failed: {result['error']}")
                
                # Small delay between captures to avoid overwhelming the system
                if attempt < 2:  # Don't delay after the last attempt
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Gemini OCR attempt {attempt + 1} exception: {e}")
        
        # Analyze results with majority voting
        if not results:
            logger.warning("All 3 Gemini OCR attempts failed - trying GROQ fallback")
            
            # Try GROQ fallback
            groq_result = await groq_fallback_analysis_with_voting()
            if groq_result["success"]:
                logger.info(f"🔄 GROQ fallback successful: {groq_result['battery_percentage']}% (confidence: {groq_result['confidence']})")
                return groq_result
            else:
                logger.error("Both Gemini and GROQ OCR failed completely")
                return {
                    "success": False,
                    "message": "All 3 Gemini OCR attempts failed, GROQ fallback also failed",
                    "total_attempts": 6,  # 3 Gemini + 3 GROQ
                    "processing_time": round(total_processing_time + groq_result.get("processing_time", 0), 2),
                    "method": "gemini_with_groq_fallback",
                    "gemini_error": "All attempts failed",
                    "groq_error": groq_result.get("message", "Unknown error")
                }
        
        # Count votes for each percentage, considering adjacent values as agreement
        from collections import Counter
        percentages = [r["percentage"] for r in results]

        # Group adjacent values (±1) together
        vote_groups = {}
        for p in percentages:
            # Find if this percentage belongs to an existing group
            found_group = False
            for group_key in list(vote_groups.keys()):
                if abs(p - group_key) <= 1:
                    vote_groups[group_key].append(p)
                    found_group = True
                    break
            if not found_group:
                vote_groups[p] = [p]

        # Find the largest group
        largest_group = max(vote_groups.values(), key=len)
        winner_percentage = round(sum(largest_group) / len(largest_group))  # Use average of group
        winner_votes = len(largest_group)
        total_valid_votes = len(results)

        # Calculate real confidence based on agreement
        confidence = winner_votes / total_valid_votes if total_valid_votes > 0 else 0

        # Cap confidence at 0.5 for zero readings to prevent false zeros from being accepted
        if winner_percentage == 0:
            confidence = min(confidence, 0.5)
            logger.debug("Zero reading detected - capping confidence at 0.5 to prevent false acceptance")

        # Only accept if we have majority agreement (at least 2/3)
        if confidence >= 2/3:  # At least 2 out of 3 agree
            success = True
            raw_counts = Counter(percentages)
            if len(set(largest_group)) > 1:
                message = f"{winner_votes}/{total_valid_votes} Gemini calls agreed on {winner_percentage}% (adjacent values: {sorted(set(largest_group))})"
            else:
                message = f"{winner_votes}/{total_valid_votes} Gemini calls agreed on {winner_percentage}%"
        else:
            success = False  # All different results
            raw_counts = Counter(percentages)
            message = f"No majority: got {dict(raw_counts)} - confidence too low"
            winner_percentage = None
        
        return {
            "success": success,
            "battery_percentage": winner_percentage,
            "confidence": round(confidence, 3),
            "total_attempts": 3,
            "valid_responses": total_valid_votes,
            "vote_distribution": dict(raw_counts),
            "processing_time": round(total_processing_time, 2),
            "method": "gemini_direct_voting",
            "message": message,
            "detailed_results": results
        }
            
    except Exception as e:
        logger.error(f"Gemini OCR direct voting failed: {e}")
        return {
            "success": False,
            "message": f"Gemini OCR voting exception: {str(e)}",
            "method": "gemini_direct_voting",
            "total_attempts": 3
        }

def is_controllable_device(device_name: str) -> bool:
    """
    Check if device should be controlled during calibration.

    Only devices with 'light', 'input', or 'output' in their names are considered
    safe for automated control. This prevents accidentally controlling critical
    infrastructure like desktop computers or servers.

    Args:
        device_name: Name of the device to check

    Returns:
        bool: True if device is safe to control, False otherwise
    """
    if not device_name:
        return False

    name_lower = device_name.lower()
    controllable_keywords = ['light', 'input', 'output']
    return any(keyword in name_lower for keyword in controllable_keywords)

def get_all_devices():
    """
    Query all devices from the device control API.

    Returns:
        list: List of device dictionaries with name and state information
              Returns empty list on error
    """
    control_api_host = os.getenv("DEVICE_CONTROL_HOST", "10.0.0.142")
    control_api_port = os.getenv("DEVICE_CONTROL_PORT", "8084")
    devices_url = f"http://{control_api_host}:{control_api_port}/devices"

    try:
        response = requests.get(
            devices_url,
            headers={"accept": "application/json"},
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("devices", [])
        else:
            logger.warning(f"Failed to get devices: HTTP {response.status_code}")
            return []

    except Exception as e:
        logger.warning(f"Error getting devices: {e}")
        return []

def get_controllable_devices():
    """
    Query and filter devices that are safe for automated control.

    Returns:
        list: List of device names that match controllable patterns (light/input/output)
    """
    all_devices = get_all_devices()
    controllable = []
    skipped = []

    for device in all_devices:
        name = device.get("name", "")
        if is_controllable_device(name):
            controllable.append(name)
        else:
            skipped.append(name)

    if controllable:
        logger.debug(f"Controllable devices: {', '.join(controllable)}")
    if skipped:
        logger.debug(f"Skipped devices (not light/input/output): {', '.join(skipped)}")

    return controllable

def get_device_states():
    """
    Query device states from the device control API using dynamic discovery

    Returns:
        dict: Dictionary with device states, e.g., {"input": False, "output_2": True}
              Returns empty dict on error
    """
    return device_discovery.get_device_states()

async def control_device(device_name: str, turn_on: bool, force: bool = False):
    """
    Control a device via the device control API

    Args:
        device_name: Name of the device (e.g., "input", "output_1", "output_2")
        turn_on: True to turn on, False to turn off
        force: True to force the command regardless of current state
    """
    control_api_host = os.getenv("DEVICE_CONTROL_HOST", "10.0.0.142")
    control_api_port = os.getenv("DEVICE_CONTROL_PORT", "8084")
    control_url = f"http://{control_api_host}:{control_api_port}/device/control"

    payload = {
        "device_name": device_name,
        "turn_on": turn_on,
        "force": force
    }

    try:
        response = requests.post(
            control_url,
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=10
        )

        if response.status_code == 200:
            logger.debug(f"Successfully controlled device {device_name}: {'on' if turn_on else 'off'}")
            # Write device state to InfluxDB on successful control
            influxdb_writer.write_device_state(device_name=device_name, is_on=turn_on)
            return True
        else:
            logger.error(f"Failed to control device {device_name}: HTTP {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"Error controlling device {device_name}: {e}")
        return False

async def control_devices_based_on_battery(
    battery_percentage: int,
    force: bool = False,
    current_output_on: bool = None,
    current_input_on: bool = None
):
    """
    Control devices based on battery percentage using shared recommendation logic.

    Only controls devices that are safe for automated control (containing 'light',
    'input', or 'output' in their names). This prevents accidentally controlling
    critical infrastructure like computers and servers during startup calibration.

    Args:
        battery_percentage: Current battery percentage
        force: Force device commands regardless of current state (used for startup calibration)
        current_output_on: Current state of outputs (True=on, False=off, None=unknown)
                          Used for output hysteresis zone decisions
        current_input_on: Current state of inputs (True=on, False=off, None=unknown)
                         Used for input hysteresis zone decisions
    """
    result = calculate_device_recommendations(battery_percentage, current_output_on, current_input_on)
    recommendations = result["recommendations"]
    reasoning = result["reasoning"]

    if force:
        logger.info(f"🔄 Startup calibration: {reasoning}")
        # Get list of controllable devices from the API
        controllable_devices = get_controllable_devices()
        logger.info(f"🔒 Calibration safety filter: only controlling devices with light/input/output in names")

        # Filter recommendations to only include controllable devices
        filtered_recommendations = {}
        skipped_recommendations = {}

        for device_name, action in recommendations.items():
            if is_controllable_device(device_name):
                # Double-check device exists in the API's device list
                if device_name in controllable_devices:
                    filtered_recommendations[device_name] = action
                else:
                    logger.debug(f"  ⚠️  Skipping {device_name}: recommended but not found in device API")
            else:
                skipped_recommendations[device_name] = action

        if skipped_recommendations:
            skipped_names = ', '.join(skipped_recommendations.keys())
            logger.info(f"  ⏭️  Skipped non-controllable devices: {skipped_names}")

        if filtered_recommendations:
            controlled_names = ', '.join(filtered_recommendations.keys())
            logger.info(f"  ✅ Calibrating controllable devices: {controlled_names}")
        else:
            logger.warning("  ⚠️  No controllable devices found to calibrate!")

        recommendations = filtered_recommendations
    else:
        logger.info(f"📋 Device control: {reasoning} [states: in={current_input_on}, out={current_output_on}]")

    # Control each device based on (filtered) recommendations
    success_count = 0
    total_devices = len(recommendations)

    for device_name, action in recommendations.items():
        turn_on = action == "turn_on"
        logger.info(f"  → Sending: {device_name} {'ON' if turn_on else 'OFF'} (force={force})")

        if await control_device(device_name, turn_on, force=force):
            success_count += 1
        else:
            logger.error(f"Failed to control {device_name}")

    if total_devices == 0:
        logger.debug("No devices to control")
        return True
    elif success_count == total_devices:
        logger.debug(f"Successfully controlled all {total_devices} devices")
        return True
    else:
        logger.warning(f"Only {success_count}/{total_devices} devices controlled successfully")
        return False

async def background_worker():
    """Main worker loop that performs background polling with parallelized OCR"""
    # Initialize database
    db = BatteryDatabase(os.getenv("DATABASE_PATH", "./data/battery_readings.db"))
    await db.init_db()

    # Initialize comparison storage
    await comparison_storage.init_db()

    # Configuration
    polling_interval = int(os.getenv("POLLING_INTERVAL_SECONDS", 60))
    confidence_threshold = float(os.getenv("POLLING_CONFIDENCE_THRESHOLD", 1.0))  # Default to 100% confidence
    screen_tap_enabled = os.getenv("SCREEN_TAP_ENABLED", "true").lower() == "true"

    # Strategy configuration
    enable_gemini = os.getenv("ENABLE_GEMINI_STRATEGY", "true").lower() == "true"
    enable_template = os.getenv("ENABLE_TEMPLATE_STRATEGY", "true").lower() == "true"
    primary_strategy = os.getenv("PRIMARY_STRATEGY", "llm").lower()

    # Validate strategy configuration
    if not enable_gemini and not enable_template:
        logger.error("ERROR: Both strategies disabled! Set ENABLE_GEMINI_STRATEGY=true or ENABLE_TEMPLATE_STRATEGY=true")
        import sys
        sys.exit(1)

    # Log strategy configuration
    strategy_msg = []
    if enable_gemini:
        strategy_msg.append("LLM (Gemini/Groq)")
    if enable_template:
        strategy_msg.append("Template Matching")
    logger.info(f"Strategy configuration: Enabled={', '.join(strategy_msg)}, Primary={primary_strategy}")

    if enable_gemini and enable_template:
        logger.info("Comparison mode: Both strategies active - will log comparisons and store statistics")
    else:
        logger.info("Comparison mode: Single strategy - no comparison logging")

    # Cooldown configuration for 100% confidence readings
    cooldown_seconds = int(os.getenv("OCR_COOLDOWN_SECONDS", 60))  # Default 60 second cooldown
    last_perfect_reading_time = 0  # Track when we last got a 100% confidence reading
    
    # Reset startup sync flag on worker startup 
    await db.set_worker_state("startup_sync_complete", "false")
    logger.info("Worker startup: reset device sync flag - first recommendation will be forced")
    
    logger.info(f"Worker starting: interval={polling_interval}s, min_confidence={confidence_threshold}, screen_tap={screen_tap_enabled}")
    
    # Define constants first
    consecutive_screen_off_count = 0
    max_screen_off_before_tap = 2  # Tap after 2 consecutive "screen off" detections
    consecutive_switchbot_failures = 0
    max_switchbot_failures_before_bypass = 5  # Bypass screen tapping after 5 consecutive failures
    max_failure_hours = float(os.getenv("SWITCHBOT_MAX_FAILURE_HOURS", 0.25))  # Default 15 minutes

    # Track monitoring mode for change detection
    last_monitoring_mode = None  # "active" or "idle"

    # Track safe shutdown state to prevent log spam
    in_safe_shutdown = False
    safe_shutdown_last_log_time = 0

    # Track consecutive 100% readings to filter false positives
    consecutive_100_readings = 0

    # Log SwitchBot configuration details
    logger.debug(f"SwitchBot resilience configuration:")
    logger.debug(f"  - Container suicide after: {max_failure_hours} hours of no successful taps")
    logger.debug(f"  - Emergency bypass after: {max_switchbot_failures_before_bypass} consecutive failures")
    logger.debug(f"  - Fresh object creation: Every tap creates new SwitchBot API object")

    while True:
        try:
            # Discover devices at start of each cycle
            discovery_result = device_discovery.discover_devices()

            # Query device states once and reuse for monitoring mode checks
            device_states = get_device_states()

            # Check if any inputs or outputs are on (pass states to avoid redundant API calls)
            input_on = device_discovery.is_any_input_on(device_states)
            output_on = device_discovery.is_any_output_on(device_states)
            # Log device states when we have a valid reading (logged later with battery %)
            is_daylight = switchbot_controller._is_daylight_hours()

            # Determine current monitoring mode
            mode_factors = []
            if input_on:
                mode_factors.append("input(s) ON")
            if output_on:
                mode_factors.append("output(s) ON")
            if is_daylight:
                mode_factors.append("daylight hours")

            current_mode = "active" if mode_factors else "idle"

            # Log mode changes at INFO level
            if last_monitoring_mode != current_mode:
                interval = switchbot_controller.active_interval if current_mode == "active" else switchbot_controller.idle_interval
                if current_mode == "active":
                    logger.info(f"🔄 Switched to ACTIVE monitoring ({interval/60:.0f} min taps): {', '.join(mode_factors)}")
                else:
                    logger.info(f"💤 Switched to IDLE monitoring ({interval/60:.0f} min taps): nighttime, all devices OFF")
                last_monitoring_mode = current_mode

            # Check if screen is on (capture directly from ESP32 for screen analysis)
            test_image = capture_image()
            screen_analysis = analyze_screen_state(test_image)
            
            should_skip_ocr = False  # Flag to make logic clearer
            
            if screen_analysis.get("screen_state") == "off":
                consecutive_screen_off_count += 1
                logger.debug(f"Screen is off (count: {consecutive_screen_off_count})")

                # Check if we should enter safe shutdown mode due to consecutive SwitchBot failures
                if consecutive_switchbot_failures >= max_switchbot_failures_before_bypass:
                    # Only log on first entry or every 10 minutes to avoid spam
                    current_time = time.time()
                    should_log = not in_safe_shutdown or (current_time - safe_shutdown_last_log_time) >= 600

                    if should_log:
                        if not in_safe_shutdown:
                            logger.critical(f"🚨 SwitchBot failure threshold reached - {consecutive_switchbot_failures} consecutive failures")
                            logger.critical("⚠️  Cannot reliably tap screen - entering SAFE SHUTDOWN mode")
                        else:
                            logger.critical(f"⚠️  Still in SAFE SHUTDOWN mode - SwitchBot unreliable for {(current_time - safe_shutdown_last_log_time) / 60:.0f} minutes")

                        logger.critical("🔌 Turning off all discovered devices to prevent damage from false readings")

                        # Turn off all devices for safety - we can't trust any readings without screen control
                        for inp in discovery_result.get("inputs", []):
                            await control_device(inp["name"], False, force=True)
                        for out in discovery_result.get("outputs", []):
                            await control_device(out["name"], False, force=True)

                        logger.info("✅ Safe shutdown complete - all devices turned off")
                        logger.info("💡 Tip: Check SwitchBot connectivity and configuration")

                        safe_shutdown_last_log_time = current_time
                        in_safe_shutdown = True
                    else:
                        # Still in safe shutdown but not logging - just turn off devices silently
                        for inp in discovery_result.get("inputs", []):
                            await control_device(inp["name"], False, force=True)
                        for out in discovery_result.get("outputs", []):
                            await control_device(out["name"], False, force=True)

                    # Skip OCR entirely - we can't trust readings without screen control
                    should_skip_ocr = True

                # If screen has been off for multiple cycles and tapping is enabled, try to turn it on
                elif (screen_tap_enabled and
                    consecutive_screen_off_count >= max_screen_off_before_tap):

                    # Check rate limit with device states (already queried at start of loop)
                    if switchbot_controller.can_tap_screen(input_on=input_on, output_on=output_on):
                        logger.debug(f"Screen has been off for {consecutive_screen_off_count} cycles, attempting to tap it on")

                        tap_result = await switchbot_controller.tap_screen(input_on=input_on, output_on=output_on)
                        if tap_result.get("success"):
                            # Reset failure counter on successful tap
                            consecutive_switchbot_failures = 0
                            consecutive_screen_off_count = 0

                            # Exit safe shutdown mode if we were in it
                            if in_safe_shutdown:
                                logger.info("✅ Exiting SAFE SHUTDOWN mode - SwitchBot working again")
                                in_safe_shutdown = False

                            # Wait a moment for screen to turn on, then retry
                            await asyncio.sleep(3)

                            # Re-check screen state after tapping
                            test_image = capture_image()
                            screen_analysis = analyze_screen_state(test_image)

                            if screen_analysis.get("screen_state") == "on":
                                logger.debug("Successfully turned screen on with SwitchBot tap!")
                                # Screen is now on, continue to OCR processing below
                            else:
                                logger.debug("Screen tap didn't turn screen on, will retry next cycle - SKIPPING OCR")
                                should_skip_ocr = True
                        else:
                            # Check if this was rate limiting vs an actual failure
                            # Rate limiting is NOT a failure - it means the tap interval policy is working correctly
                            if tap_result.get('rate_limited'):
                                # Rate limited - this is expected behavior, not a failure
                                # Log at INFO level and skip OCR, but do NOT increment failure counter
                                time_remaining = tap_result.get('time_until_next_tap_seconds', 0)
                                logger.info(f"Screen tap skipped (rate limited) - {time_remaining/60:.1f} min until next tap allowed")
                                should_skip_ocr = True
                            else:
                                # Actual SwitchBot failure - increment failure counter and log details
                                consecutive_switchbot_failures += 1
                                error_msg = tap_result.get('error', 'Unknown error')
                                error_details = tap_result.get('details', '')
                                error_code = tap_result.get('error_code', 0)

                                # Build detailed error message
                                full_error = f"{error_msg}"
                                if error_details:
                                    full_error += f" - {error_details}"
                                if error_code:
                                    full_error += f" (HTTP {error_code})"

                                logger.error(f"SwitchBot tap failed ({consecutive_switchbot_failures}/{max_switchbot_failures_before_bypass}): {full_error}")

                                # Log additional diagnostic info on first few failures
                                if consecutive_switchbot_failures <= 2:
                                    logger.error(f"  Device states: input={'ON' if input_on else 'OFF'}, output={'ON' if output_on else 'OFF'}")
                                    logger.error(f"  Monitoring mode: {current_mode}")

                                if consecutive_switchbot_failures >= max_switchbot_failures_before_bypass:
                                    logger.error(f"SwitchBot failure threshold reached - will bypass screen tapping on next cycle")

                                should_skip_ocr = True
                    else:
                        # Rate limited - log and skip
                        time_until_next = switchbot_controller.get_time_until_next_tap(input_on=input_on, output_on=output_on)
                        logger.debug(f"Screen tap rate limited - {time_until_next/60:.1f} min until next tap - SKIPPING OCR")
                        should_skip_ocr = True
                else:
                    # Screen is off but we haven't reached the tap threshold yet
                    logger.debug(f"Screen off, waiting for {max_screen_off_before_tap - consecutive_screen_off_count} more cycles before tapping - SKIPPING OCR")
                    should_skip_ocr = True
            else:
                # Screen is on, reset the counter and exit safe shutdown
                consecutive_screen_off_count = 0
                if in_safe_shutdown:
                    logger.info("✅ Exiting SAFE SHUTDOWN mode - screen is back on")
                    in_safe_shutdown = False
                logger.debug("Screen detected as ON")
                
            # Check if we should skip OCR
            if should_skip_ocr:
                logger.debug("⏭️  Skipping OCR cycle as planned")
                await asyncio.sleep(polling_interval)
                continue

            # Check if we're in cooldown period after 100% confidence reading
            current_time = time.time()
            time_since_perfect = current_time - last_perfect_reading_time
            if time_since_perfect < cooldown_seconds:
                remaining_cooldown = cooldown_seconds - time_since_perfect
                logger.info(f"⏸️  Cooldown: {remaining_cooldown:.0f}s remaining")
                await asyncio.sleep(polling_interval)
                continue

            # If we get here, screen should be on - proceed with OCR
            logger.debug("🎯 Screen is ON - Running configured OCR strategies")
            start_time = time.time()

            # Variables to hold results from each strategy
            gemini_result = None
            groq_result = None
            template_result = None
            llm_source = "none"

            # Strategy 1: Run LLM OCR if enabled
            if enable_gemini:
                logger.debug("Running LLM strategy (Gemini/Groq with voting)")
                gemini_result = await gemini_ocr_analysis_with_voting()

                # Determine which LLM was actually used
                if gemini_result.get("success"):
                    method = gemini_result.get("method", "gemini_direct_voting")
                    if "groq" in method:
                        llm_source = "groq"
                    else:
                        llm_source = "gemini"

            # Strategy 2: Run template matching if enabled
            processed_image = None
            if enable_template:
                try:
                    # Capture and process image for template matching
                    processed_image = capture_and_process_image_for_gemini()
                    logger.debug("Running template matching strategy")
                    template_result = template_classifier.classify_image(processed_image)
                except Exception as e:
                    logger.warning(f"Template matching failed: {e}")
                    template_result = {
                        "success": False,
                        "error": str(e),
                        "percentage": None,
                        "confidence": 0.0
                    }

            # Determine which result to use for logic (PRIMARY_STRATEGY)
            primary_result = None
            primary_percentage = None
            primary_confidence = 0.0

            if primary_strategy == "template" and enable_template:
                if template_result and template_result.get("success"):
                    primary_result = template_result
                    primary_percentage = template_result["percentage"]
                    primary_confidence = template_result["confidence"]
                    logger.debug(f"Using template result as primary: {primary_percentage}%")
                elif enable_gemini and gemini_result and gemini_result.get("success"):
                    # Fallback to LLM if template failed
                    primary_result = gemini_result
                    primary_percentage = gemini_result["battery_percentage"]
                    primary_confidence = gemini_result["confidence"]
                    logger.debug(f"Template failed, falling back to LLM: {primary_percentage}%")
            else:  # primary_strategy == "llm" or default
                if enable_gemini and gemini_result and gemini_result.get("success"):
                    primary_result = gemini_result
                    primary_percentage = gemini_result["battery_percentage"]
                    primary_confidence = gemini_result["confidence"]
                    logger.debug(f"Using LLM result as primary: {primary_percentage}%")
                elif enable_template and template_result and template_result.get("success"):
                    # Fallback to template if LLM failed
                    primary_result = template_result
                    primary_percentage = template_result["percentage"]
                    primary_confidence = template_result["confidence"]
                    logger.debug(f"LLM failed, falling back to template: {primary_percentage}%")

            # COLLECTION MODE: Save image with LLM's label if LLM succeeded (always active)
            if enable_gemini and gemini_result and gemini_result.get("success"):
                try:
                    if processed_image is None:
                        processed_image = capture_and_process_image_for_gemini()
                    llm_percentage = gemini_result["battery_percentage"]
                    save_success = template_classifier.save_labeled_image(processed_image, llm_percentage)
                    if save_success:
                        logger.debug(f"Collected training image for {llm_percentage}%")
                except Exception as e:
                    logger.warning(f"Failed to collect training image: {e}")

            # COMPARISON MODE: Log and save comparison if both strategies ran
            if enable_gemini and enable_template:
                # Log to stdout (existing behavior)
                if gemini_result and gemini_result.get("success"):
                    log_comparison(gemini_result["battery_percentage"], template_result)

                # Save to comparison storage
                try:
                    if processed_image is None:
                        processed_image = capture_and_process_image_for_gemini()

                    gemini_pct = gemini_result["battery_percentage"] if gemini_result and gemini_result.get("success") else None
                    template_pct = template_result["percentage"] if template_result and template_result.get("success") else None
                    template_conf = template_result.get("confidence") if template_result else None

                    await comparison_storage.save_comparison(
                        image_data=processed_image,
                        gemini_percentage=gemini_pct,
                        groq_percentage=None,  # Will be set if groq was used
                        template_percentage=template_pct,
                        template_confidence=template_conf,
                        llm_source=llm_source
                    )
                except Exception as e:
                    logger.error(f"Failed to save comparison record: {e}")

            processing_time = time.time() - start_time

            # Process primary result if we have one
            if primary_result and primary_percentage is not None:

                if primary_confidence >= confidence_threshold:
                    # Check plausibility against last reading
                    should_store = True
                    plausibility_msg = ""

                    # Special handling for 100% readings to filter false positives
                    # Require 3 consecutive 100% readings before believing it
                    if primary_percentage == 100:
                        consecutive_100_readings += 1
                        if consecutive_100_readings < 3:
                            should_store = False
                            plausibility_msg = f"100% reading #{consecutive_100_readings}/3 - waiting for confirmation"
                            logger.info(f"⚠️ 100% reading #{consecutive_100_readings}/3 ignored - need 3 consecutive to confirm")
                        else:
                            logger.info(f"✅ 100% reading CONFIRMED after {consecutive_100_readings} consecutive reads")
                    else:
                        # Reset counter when we see non-100% reading
                        if consecutive_100_readings > 0:
                            logger.debug(f"Reset consecutive 100% counter (was {consecutive_100_readings})")
                        consecutive_100_readings = 0

                    try:
                        last_readings = await db.get_recent_readings(limit=1)
                        if last_readings:
                            last_reading = last_readings[0]
                            time_diff_minutes = (datetime.now().timestamp() - last_reading["timestamp"]) / 60
                            percentage_diff = abs(primary_percentage - last_reading["battery_percentage"])

                            # Plausibility checks
                            if time_diff_minutes < 2 and percentage_diff > 8:
                                if primary_confidence < 0.9:
                                    should_store = False
                                    plausibility_msg = f"implausible change rejected: {last_reading['battery_percentage']}% → {primary_percentage}% in {time_diff_minutes:.1f}min (conf: {primary_confidence})"
                            elif time_diff_minutes < 5 and percentage_diff > 15:
                                if primary_confidence < 0.85:
                                    should_store = False
                                    plausibility_msg = f"implausible change rejected: {last_reading['battery_percentage']}% → {primary_percentage}% in {time_diff_minutes:.1f}min (conf: {primary_confidence})"
                    except:
                        pass

                    if should_store:
                        # Determine OCR method for logging
                        ocr_method = f"worker_{primary_strategy}"
                        total_attempts = primary_result.get("total_attempts", 1)
                        raw_vote_data = primary_result.get("raw_response")

                        # Store in database and get the filtered percentage
                        filtered_percentage = await db.insert_reading(
                            battery_percentage=primary_percentage,
                            confidence=primary_confidence,
                            ocr_method=ocr_method,
                            total_attempts=total_attempts,
                            raw_vote_data=raw_vote_data
                        )
                        # Write battery reading to InfluxDB
                        influxdb_writer.write_battery_reading(
                            battery_percentage=filtered_percentage,
                            ocr_confidence=primary_confidence,
                            ocr_strategy=primary_strategy
                        )

                        # Compact logging - combine reading and voting details
                        vote_info = ""
                        if "vote_distribution" in primary_result and len(primary_result.get('vote_distribution', {})) > 1:
                            vote_info = f" [{primary_result['vote_distribution']}]"
                        logger.info(f"📊 {primary_percentage}% (conf: {primary_confidence:.2f}, {primary_strategy}){vote_info}")

                        # If we got 100% confidence, start cooldown period
                        if primary_confidence >= 1.0:
                            last_perfect_reading_time = time.time()
                            logger.info(f"⏸️  Perfect confidence achieved - entering {cooldown_seconds}s cooldown period")

                        # Check if this is the first successful reading after startup
                        startup_sync_done = await db.get_worker_state("startup_sync_complete")
                        force_devices = startup_sync_done != "true"

                        # Control devices based on the filtered percentage
                        # Pass current device states for hysteresis logic
                        device_control_success = await control_devices_based_on_battery(
                            filtered_percentage,
                            force=force_devices,
                            current_output_on=output_on,
                            current_input_on=input_on
                        )
                        
                        # Mark startup sync as complete after first successful device control
                        if force_devices:
                            await db.set_worker_state("startup_sync_complete", "true")
                            logger.debug("Startup device synchronization complete - subsequent controls will be non-forced")
                        
                        # Log health status for debugging
                        logger.debug(f"Worker health: DB write ✅, Device control {'✅' if device_control_success else '❌'}")
                        
                    else:
                        logger.debug(f"Plausibility check failed: {plausibility_msg}")
                else:
                    logger.warning(f"❌ Low confidence reading skipped: {primary_percentage}% (confidence: {primary_confidence})")
            else:
                # No valid result from any strategy
                error_messages = []
                if enable_gemini and gemini_result:
                    error_messages.append(f"LLM: {gemini_result.get('message', 'failed')}")
                if enable_template and template_result:
                    error_messages.append(f"Template: {template_result.get('error', 'failed')}")
                logger.info(f"❌ OCR failed: {', '.join(error_messages) if error_messages else 'No valid results'}")
                
        except Exception as e:
            logger.error(f"Background worker error: {e}")
        
        # Check suicide condition (container should exit if SwitchBot broken too long)
        suicide_check = switchbot_controller.check_suicide_condition()
        if suicide_check["should_exit"]:
            logger.critical("💀 Exiting container due to prolonged SwitchBot failure - Docker will restart")
            import sys
            sys.exit(1)
        
        await asyncio.sleep(polling_interval)

if __name__ == "__main__":
    logger.info("Starting Bluetti Monitor Worker")
    asyncio.run(background_worker())