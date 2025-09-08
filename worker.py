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

load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))
logger = logging.getLogger(__name__)

# Database management (same as main.py)
class BatteryDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def init_db(self):
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
    
    async def insert_reading(self, battery_percentage: int, confidence: float, 
                           ocr_method: str = None, total_attempts: int = None, 
                           raw_vote_data: dict = None):
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
        raise Exception("WEBCAM_URL not configured")
    
    capture_url = urljoin(webcam_url, capture_endpoint)
    
    # Retry logic for network issues
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(capture_url, timeout=10)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            error_msg = str(e).lower()
            if "premature end of data segment" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                if attempt < max_retries - 1:
                    logger.debug(f"Recoverable network error on attempt {attempt + 1}: {e}")
                    time.sleep(1)
                    continue
            
            if attempt == max_retries - 1:
                raise e
            time.sleep(0.5)
    
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
        
        # Crop and flip
        cropped = img[crop_coords['y1']:crop_coords['y2'], crop_coords['x1']:crop_coords['x2']]
        flipped = cv2.flip(cropped, 1)
        gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
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
    
    # Get crop coordinates, crop, and flip (same as /capture/flip endpoint)
    coords = get_crop_coordinates()
    cropped = img[coords['y1']:coords['y2'], coords['x1']:coords['x2']]
    flipped = cv2.flip(cropped, 1)  # Flip horizontally
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', flipped)
    return buffer.tobytes()

async def gemini_ocr_analysis_with_voting() -> Dict:
    """
    Perform OCR analysis using Gemini vision model with 3-capture majority voting
    Direct image processing - no HTTP self-calls
    
    Returns:
        Dictionary with OCR results and real confidence based on agreement
    """
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
            return {
                "success": False,
                "message": "All 3 Gemini OCR attempts failed",
                "total_attempts": 3,
                "processing_time": round(total_processing_time, 2),
                "method": "gemini_direct_voting"
            }
        
        # Count votes for each percentage
        from collections import Counter
        percentages = [r["percentage"] for r in results]
        vote_counts = Counter(percentages)
        most_common = vote_counts.most_common()
        
        # Determine confidence based on agreement
        winner_percentage = most_common[0][0]
        winner_votes = most_common[0][1]
        total_valid_votes = len(results)
        
        # Calculate real confidence based on agreement
        confidence = winner_votes / total_valid_votes if total_valid_votes > 0 else 0
        
        # Only accept if we have majority agreement (at least 2/3)
        if confidence >= 2/3:  # At least 2 out of 3 agree
            success = True
            message = f"{winner_votes}/{total_valid_votes} Gemini calls agreed on {winner_percentage}%"
        else:
            success = False  # All different results
            message = f"No majority: got {dict(vote_counts)} - confidence too low"
            winner_percentage = None
        
        return {
            "success": success,
            "battery_percentage": winner_percentage,
            "confidence": round(confidence, 3),
            "total_attempts": 3,
            "valid_responses": total_valid_votes,
            "vote_distribution": dict(vote_counts),
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

async def control_device(device_name: str, turn_on: bool, force: bool = False):
    """
    Control a device via the device control API
    
    Args:
        device_name: Name of the device (e.g., "input", "output_1", "output_2")
        turn_on: True to turn on, False to turn off
        force: True to force the command regardless of current state
    """
    control_api_host = os.getenv("DEVICE_CONTROL_HOST", "10.0.0.109")
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
            logger.info(f"Successfully controlled device {device_name}: {'on' if turn_on else 'off'}")
            return True
        else:
            logger.error(f"Failed to control device {device_name}: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Error controlling device {device_name}: {e}")
        return False

"""
Shared recommendation logic for Bluetti power management

This module contains the core logic for determining device control recommendations
based on battery status. It's used by both the API endpoint and the worker.
"""
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_device_recommendations(battery_percentage: int) -> Dict:
    """
    Calculate device control recommendations based on battery percentage
    
    Args:
        battery_percentage: Current battery percentage (0-100)
        
    Returns:
        Dictionary with device recommendations and reasoning
    """
    if battery_percentage < 10:
        # Below 10% - turn off outputs, turn on input (charge)
        recommendations = {
            "input": "turn_on",
            "output_1": "turn_off",
            "output_2": "turn_off"
        }
        reasoning = f"Battery at {battery_percentage}% - critical low, charging needed"
        
    elif 10 <= battery_percentage < 60:
        # Between 10-60% - turn off input, turn off outputs (conservation)
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
        "recommendations": recommendations,
        "reasoning": reasoning
    }

def analyze_recent_readings_for_recommendations(readings_last_30min: List[Dict]) -> Dict:
    """
    Analyze recent readings and generate recommendations
    
    Args:
        readings_last_30min: List of battery readings from the last 30 minutes
        
    Returns:
        Dictionary with recommendation result including status, recommendations, and metadata
    """
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
    
    # Get recommendations based on battery percentage
    recommendation_result = calculate_device_recommendations(battery_percentage)
    
    return {
        "success": True,
        "status": "active",
        "message": "Recommendations based on recent battery status",
        "recommendations": recommendation_result["recommendations"],
        "reasoning": recommendation_result["reasoning"],
        "battery_percentage": battery_percentage,
        "last_reading_age_minutes": round(last_reading_age_minutes, 1),
        "readings_in_last_30min": len(readings_last_30min),
        "confidence": latest_reading.get("confidence", None),
        "last_reading_timestamp": latest_reading["timestamp"]
    }

async def control_devices_based_on_battery(battery_percentage: int, force: bool = False):
    """
    Control devices based on battery percentage using shared recommendation logic
    
    Args:
        battery_percentage: Current battery percentage
        force: Force device commands regardless of current state
    """
    result = calculate_device_recommendations(battery_percentage)
    recommendations = result["recommendations"]
    reasoning = result["reasoning"]
    
    if force:
        logger.info(f"Device control decision (FORCED): {reasoning}")
    else:
        logger.info(f"Device control decision: {reasoning}")
    
    # Control each device based on recommendations
    success_count = 0
    total_devices = 0
    
    for device_name, action in recommendations.items():
        total_devices += 1
        turn_on = action == "turn_on"
        
        if await control_device(device_name, turn_on, force=force):
            success_count += 1
        else:
            logger.error(f"Failed to control {device_name}")
    
    if success_count == total_devices:
        if force:
            logger.info(f"Successfully forced all {total_devices} devices to desired state")
        else:
            logger.info(f"Successfully controlled all {total_devices} devices")
        return True
    else:
        logger.warning(f"Only {success_count}/{total_devices} devices controlled successfully")
        return False

"""
Shared SwitchBot controller for Bluetti screen management

This module handles SwitchBot operations including device initialization,
rate limiting, and screen tapping functionality.
"""
import os
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from switchbot import SwitchBot
    SWITCHBOT_AVAILABLE = True
except ImportError:
    logger.warning("python-switchbot not installed - SwitchBot functionality disabled")
    SWITCHBOT_AVAILABLE = False


class SwitchBotController:
    """Handle SwitchBot operations for screen control with rate limiting"""
    
    def __init__(self):
        self.token = os.getenv("SWITCH_BOT_TOKEN")
        self.secret = os.getenv("SWITCH_BOT_SECRET")
        self.switchbot = None
        self.bot_device = None
        self._initialized = False
        self.last_tap_time = 0  # Track last tap timestamp
        self.min_tap_interval = 15 * 60  # 15 minutes in seconds
        
        # Circuit breaker state for resilience
        self.failure_count = 0
        self.max_failures = 5  # Open circuit after 5 consecutive failures
        self.circuit_open_time = 0
        self.circuit_timeout = 300  # 5 minutes before trying again
        self.circuit_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = 0
        
    async def initialize(self) -> bool:
        """Initialize SwitchBot connection and find the bot device"""
        if self._initialized:
            return True
            
        if not SWITCHBOT_AVAILABLE:
            logger.warning("python-switchbot package not available")
            return False
            
        if not self.token or not self.secret:
            logger.warning("SWITCH_BOT_TOKEN or SWITCH_BOT_SECRET not configured - screen tapping disabled")
            return False
            
        try:
            self.switchbot = SwitchBot(token=self.token, secret=self.secret)
            devices = self.switchbot.devices()
            
            # Find the first bot device (since user mentioned they have only one)
            for device in devices:
                if hasattr(device, 'press') or 'Bot' in str(type(device)):
                    self.bot_device = device
                    logger.info(f"Found SwitchBot device: {device}")
                    break
                    
            if not self.bot_device:
                logger.warning("No SwitchBot Bot device found")
                return False
                
            self._initialized = True
            logger.info("SwitchBot controller initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SwitchBot: {e}")
            return False
    
    def get_time_until_next_tap(self) -> float:
        """Get time in seconds until next tap is allowed"""
        current_time = time.time()
        time_since_last_tap = current_time - self.last_tap_time
        return max(0, self.min_tap_interval - time_since_last_tap)
    
    def can_tap_screen(self) -> bool:
        """Check if enough time has passed since last tap (15 minute minimum)"""
        return self.get_time_until_next_tap() == 0
    
    def _check_circuit_state(self) -> bool:
        """Check and update circuit breaker state"""
        current_time = time.time()
        
        if self.circuit_state == "OPEN":
            # Check if timeout has expired
            if current_time - self.circuit_open_time >= self.circuit_timeout:
                self.circuit_state = "HALF_OPEN"
                logger.info("SwitchBot circuit breaker moving to HALF_OPEN - attempting test")
                return True
            else:
                return False  # Circuit still open
                
        elif self.circuit_state == "HALF_OPEN":
            # In half-open state, allow one test request
            return True
            
        else:  # CLOSED
            return True
    
    def _record_success(self):
        """Record successful operation"""
        self.failure_count = 0
        if self.circuit_state == "HALF_OPEN":
            self.circuit_state = "CLOSED"
            logger.info("SwitchBot circuit breaker closed - service recovered")
    
    def _record_failure(self):
        """Record failed operation and potentially open circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.max_failures:
            self.circuit_state = "OPEN"
            self.circuit_open_time = time.time()
            logger.error(f"SwitchBot circuit breaker opened after {self.max_failures} failures - disabling for {self.circuit_timeout}s")
        elif self.circuit_state == "HALF_OPEN":
            # Test failed in half-open, go back to open
            self.circuit_state = "OPEN"
            self.circuit_open_time = time.time()
            logger.warning("SwitchBot circuit breaker test failed - returning to OPEN state")
    
    def is_healthy(self) -> dict:
        """Get health status of SwitchBot controller"""
        current_time = time.time()
        
        if self.circuit_state == "OPEN":
            time_remaining = self.circuit_timeout - (current_time - self.circuit_open_time)
            return {
                "healthy": False,
                "state": "CIRCUIT_OPEN",
                "failure_count": self.failure_count,
                "time_until_retry": max(0, time_remaining),
                "last_failure": self.last_failure_time
            }
        elif self.circuit_state == "HALF_OPEN":
            return {
                "healthy": False,
                "state": "TESTING",
                "failure_count": self.failure_count,
                "message": "Testing if service has recovered"
            }
        else:
            return {
                "healthy": True,
                "state": "OPERATIONAL",
                "failure_count": self.failure_count
            }
    
    async def tap_screen(self, force: bool = False) -> dict:
        """
        Tap the screen using SwitchBot to turn it on with circuit breaker protection
        
        Args:
            force: If True, bypass rate limiting (use with caution)
            
        Returns:
            Dictionary with success status and details
        """
        # Check circuit breaker state first
        if not self._check_circuit_state():
            health = self.is_healthy()
            return {
                "success": False,
                "error": "SwitchBot circuit breaker open",
                "details": f"Service unavailable for {health['time_until_retry']:.0f} more seconds",
                "health_status": health
            }
        
        if not await self.initialize():
            self._record_failure()
            return {
                "success": False,
                "error": "SwitchBot not initialized",
                "details": "Check token/secret configuration or device availability"
            }
            
        if not force and not self.can_tap_screen():
            time_remaining = self.get_time_until_next_tap()
            minutes_remaining = time_remaining / 60
            return {
                "success": False,
                "error": "Rate limited",
                "details": f"Next tap allowed in {minutes_remaining:.1f} minutes",
                "time_until_next_tap_seconds": time_remaining
            }
            
        try:
            logger.info(f"Tapping screen with SwitchBot{'(FORCED)' if force else ''}...")
            self.bot_device.press()
            self.last_tap_time = time.time()  # Record the tap time
            
            # Record success for circuit breaker
            self._record_success()
            
            next_tap_time = time.time() + self.min_tap_interval
            logger.info(f"Screen tap completed - next tap allowed at {time.strftime('%H:%M:%S', time.localtime(next_tap_time))}")
            
            return {
                "success": True,
                "message": "Screen tapped successfully",
                "tap_time": self.last_tap_time,
                "next_tap_allowed_at": next_tap_time,
                "forced": force
            }
            
        except Exception as e:
            logger.error(f"Failed to tap screen with SwitchBot: {e}")
            self._record_failure()  # Record failure for circuit breaker
            return {
                "success": False,
                "error": "Tap failed",
                "details": str(e),
                "health_status": self.is_healthy()
            }
    
    async def get_status(self) -> dict:
        """Get SwitchBot controller status"""
        if not await self.initialize():
            return {
                "initialized": False,
                "device_found": False,
                "can_tap": False,
                "error": "Not initialized"
            }
        
        time_until_next = self.get_time_until_next_tap()
        
        return {
            "initialized": True,
            "device_found": self.bot_device is not None,
            "device_info": str(self.bot_device) if self.bot_device else None,
            "can_tap": time_until_next == 0,
            "time_until_next_tap_seconds": time_until_next,
            "time_until_next_tap_minutes": round(time_until_next / 60, 1),
            "last_tap_time": self.last_tap_time if self.last_tap_time > 0 else None,
            "rate_limit_minutes": self.min_tap_interval / 60,
            "health_status": self.is_healthy(),
            "circuit_state": self.circuit_state,
            "failure_count": self.failure_count
        }


# Global instance for shared use
switchbot_controller = SwitchBotController()

async def background_worker():
    """Main worker loop that performs background polling with parallelized OCR"""
    # Initialize database
    db = BatteryDatabase(os.getenv("DATABASE_PATH", "./data/battery_readings.db"))
    await db.init_db()
    
    # Configuration
    polling_interval = int(os.getenv("POLLING_INTERVAL_SECONDS", 60))
    confidence_threshold = float(os.getenv("POLLING_CONFIDENCE_THRESHOLD", 0.67))  # 2/3 majority
    screen_tap_enabled = os.getenv("SCREEN_TAP_ENABLED", "true").lower() == "true"
    
    # Reset startup sync flag on worker startup 
    await db.set_worker_state("startup_sync_complete", "false")
    logger.info("Worker startup: reset device sync flag - first recommendation will be forced")
    
    logger.info(f"Worker starting: interval={polling_interval}s, min_confidence={confidence_threshold}, screen_tap={screen_tap_enabled}")
    
    consecutive_screen_off_count = 0
    max_screen_off_before_tap = 2  # Tap after 2 consecutive "screen off" detections
    
    while True:
        try:
            # Check if screen is on (capture directly from ESP32 for screen analysis)
            test_image = capture_image()
            screen_analysis = analyze_screen_state(test_image)
            
            should_skip_ocr = False  # Flag to make logic clearer
            
            if screen_analysis.get("screen_state") == "off":
                consecutive_screen_off_count += 1
                logger.debug(f"Screen is off (count: {consecutive_screen_off_count})")
                
                # If screen has been off for multiple cycles and tapping is enabled, try to turn it on
                if (screen_tap_enabled and 
                    consecutive_screen_off_count >= max_screen_off_before_tap and
                    switchbot_controller.can_tap_screen()):  # Check rate limit
                    
                    logger.info(f"Screen has been off for {consecutive_screen_off_count} cycles, attempting to tap it on")
                    
                    if await switchbot_controller.tap_screen():
                        # Reset counter since we just tapped
                        consecutive_screen_off_count = 0
                        
                        # Wait a moment for screen to turn on, then retry
                        await asyncio.sleep(3)
                        
                        # Re-check screen state after tapping
                        test_image = capture_image()
                        screen_analysis = analyze_screen_state(test_image)
                        
                        if screen_analysis.get("screen_state") == "on":
                            logger.info("Successfully turned screen on with SwitchBot tap!")
                            # Screen is now on, continue to OCR processing below
                        else:
                            logger.warning("Screen tap didn't turn screen on, will retry next cycle - SKIPPING OCR")
                            should_skip_ocr = True
                    else:
                        logger.debug("Cannot tap screen (rate limited or failed), skipping OCR polling")
                        should_skip_ocr = True
                else:
                    # Screen is off but we haven't reached the tap threshold yet, or rate limited
                    if screen_tap_enabled and consecutive_screen_off_count >= max_screen_off_before_tap:
                        logger.debug("Screen tap rate limited - waiting for next allowed tap window - SKIPPING OCR")
                    else:
                        logger.debug(f"Screen off, waiting for {max_screen_off_before_tap - consecutive_screen_off_count} more cycles before tapping - SKIPPING OCR")
                    should_skip_ocr = True
            else:
                # Screen is on, reset the counter
                consecutive_screen_off_count = 0
                logger.debug("Screen detected as ON")
                
            # Check if we should skip OCR
            if should_skip_ocr:
                logger.debug("⏭️  Skipping OCR cycle as planned")
                await asyncio.sleep(polling_interval)
                continue
                
            # If we get here, screen should be on - proceed with OCR
            logger.info("🎯 Screen is ON - Starting Gemini OCR analysis with majority voting")
            start_time = time.time()
            ocr_result = await gemini_ocr_analysis_with_voting()
            processing_time = time.time() - start_time
            
            if ocr_result["success"]:
                battery_percentage = ocr_result["battery_percentage"]
                confidence = ocr_result["confidence"]
                
                if confidence >= confidence_threshold:
                    # Check plausibility against last reading
                    should_store = True
                    plausibility_msg = ""
                    
                    try:
                        last_readings = await db.get_recent_readings(limit=1)
                        if last_readings:
                            last_reading = last_readings[0]
                            time_diff_minutes = (datetime.now().timestamp() - last_reading["timestamp"]) / 60
                            percentage_diff = abs(battery_percentage - last_reading["battery_percentage"])
                            
                            # Plausibility checks
                            if time_diff_minutes < 2 and percentage_diff > 8:
                                if confidence < 0.9:
                                    should_store = False
                                    plausibility_msg = f"implausible change rejected: {last_reading['battery_percentage']}% → {battery_percentage}% in {time_diff_minutes:.1f}min (conf: {confidence})"
                            elif time_diff_minutes < 5 and percentage_diff > 15:
                                if confidence < 0.85:
                                    should_store = False  
                                    plausibility_msg = f"implausible change rejected: {last_reading['battery_percentage']}% → {battery_percentage}% in {time_diff_minutes:.1f}min (conf: {confidence})"
                    except:
                        pass
                    
                    if should_store:
                        # Store in database
                        await db.insert_reading(
                            battery_percentage=battery_percentage,
                            confidence=confidence,
                            ocr_method="worker_gemini",
                            total_attempts=ocr_result["total_attempts"],
                            raw_vote_data=ocr_result.get("raw_response")
                        )
                        logger.info(f"✅ Successfully stored reading: {battery_percentage}% (conf: {confidence}, time: {processing_time:.2f}s, method: {ocr_result.get('method', 'unknown')})")
                        
                        # Log voting details if available
                        if "vote_distribution" in ocr_result:
                            logger.info(f"Gemini voting: {ocr_result['vote_distribution']} → {ocr_result.get('message', 'majority achieved')}")
                        
                        # Check if this is the first successful reading after startup
                        startup_sync_done = await db.get_worker_state("startup_sync_complete")
                        force_devices = startup_sync_done != "true"
                        
                        # Control devices based on recommendations
                        device_control_success = await control_devices_based_on_battery(battery_percentage, force=force_devices)
                        
                        # Mark startup sync as complete after first successful device control
                        if force_devices:
                            await db.set_worker_state("startup_sync_complete", "true")
                            logger.info("Startup device synchronization complete - subsequent controls will be non-forced")
                        
                        # Log health status for debugging
                        logger.debug(f"Worker health: DB write ✅, Device control {'✅' if device_control_success else '❌'}")
                        
                    else:
                        logger.debug(f"Plausibility check failed: {plausibility_msg}")
                else:
                    logger.warning(f"❌ Low confidence reading skipped: {battery_percentage}% (confidence: {confidence}) - {ocr_result.get('message', 'no details')}")
            else:
                logger.warning(f"❌ OCR failed: {ocr_result.get('message', 'No valid OCR results')} - no database write")
                # This could be causing health check failures!
                
        except Exception as e:
            logger.error(f"Background worker error: {e}")
        
        await asyncio.sleep(polling_interval)

if __name__ == "__main__":
    logger.info("Starting Bluetti Monitor Worker")
    asyncio.run(background_worker())