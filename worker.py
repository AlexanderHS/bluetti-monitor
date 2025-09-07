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

async def parallel_ocr_analysis(image_bytes: bytes, num_captures: int = 3) -> Dict:
    """
    Perform parallelized OCR analysis across multiple captures, thresholds, and PSM modes
    
    Args:
        image_bytes: Raw image data
        num_captures: Number of image captures to process
        
    Returns:
        Dictionary with OCR results and confidence metrics
    """
    thresholds = [145, 150, 155, 160, 165, 170]
    psm_modes = [6, 7, 8, 13]
    crop_coords = get_crop_coordinates()
    
    all_results = []
    capture_details = []
    
    # Use fewer processes to avoid overwhelming the system and ESP32
    num_processes = min(mp.cpu_count(), 4)  # Cap at 4 to be more conservative
    
    for capture_num in range(num_captures):
        try:
            # Capture fresh image for each attempt
            if capture_num == 0:
                current_image = image_bytes
            else:
                # Add small delay between captures to be gentler on ESP32
                await asyncio.sleep(0.5)
                current_image = capture_image()
            
            # Prepare tasks for all threshold/PSM combinations
            tasks = []
            for threshold in thresholds:
                for psm_mode in psm_modes:
                    tasks.append((current_image, threshold, psm_mode, crop_coords))
            
            capture_start = time.time()
            
            # Execute OCR tasks in parallel using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit all tasks and collect results
                results = list(executor.map(process_single_ocr_task, tasks))
            
            capture_time = time.time() - capture_start
            
            # Filter out None results and add to all_results
            valid_results = [r for r in results if r is not None]
            all_results.extend(valid_results)
            
            capture_details.append({
                "capture": capture_num + 1,
                "results_count": len(valid_results),
                "processing_time_seconds": round(capture_time, 2),
                "tasks_executed": len(tasks),
                "processes_used": num_processes
            })
            
        except Exception as e:
            logger.error(f"Capture {capture_num + 1} failed: {e}")
            capture_details.append({
                "capture": capture_num + 1,
                "error": str(e)
            })
    
    # Enhanced voting mechanism with conflict resolution
    if all_results:
        vote_counts = Counter(all_results)
        
        # Smart conflict resolution (same logic as original)
        def are_conflicting(num1, num2):
            return (num1 * 10 == num2) or (num2 * 10 == num1)
        
        resolved_votes = Counter()
        conflict_detected = False
        
        for percentage, count in vote_counts.items():
            conflicts_with = []
            for other_percentage, other_count in vote_counts.items():
                if percentage != other_percentage and are_conflicting(percentage, other_percentage):
                    conflicts_with.append((other_percentage, other_count))
            
            if conflicts_with:
                conflict_detected = True
                # Simple resolution: prefer the one with more votes, or the 2-digit one if tied
                should_prefer_current = True
                for conflict_pct, conflict_count in conflicts_with:
                    if conflict_count > count * 1.2:  # Significantly more votes
                        should_prefer_current = False
                        break
                    elif conflict_count == count and len(str(conflict_pct)) == 2 and len(str(percentage)) == 1:
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
        
        return {
            "success": True,
            "battery_percentage": winner[0],
            "confidence": round(confidence, 3),
            "total_attempts": total_votes,
            "winning_votes": winning_votes,
            "vote_distribution": dict(vote_counts),
            "resolved_vote_distribution": dict(resolved_votes) if conflict_detected else None,
            "conflict_resolution_applied": conflict_detected,
            "captures_attempted": num_captures,
            "captures_succeeded": len([c for c in capture_details if "error" not in c]),
            "detailed_results": capture_details,
            "parallelization_used": True
        }
    else:
        return {
            "success": False,
            "message": "No valid OCR results obtained",
            "captures_attempted": num_captures,
            "detailed_results": capture_details,
            "parallelization_used": True
        }

async def control_device(device_name: str, turn_on: bool):
    """
    Control a device via the device control API
    
    Args:
        device_name: Name of the device (e.g., "input", "output_1", "output_2")
        turn_on: True to turn on, False to turn off
    """
    control_api_host = os.getenv("DEVICE_CONTROL_HOST", "10.0.0.109")
    control_api_port = os.getenv("DEVICE_CONTROL_PORT", "8084")
    control_url = f"http://{control_api_host}:{control_api_port}/device/control"
    
    payload = {
        "device_name": device_name,
        "turn_on": turn_on
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

async def get_recommendations_and_control():
    """
    Fetch recommendations from the API and control devices accordingly
    """
    api_host = os.getenv("API_HOST", "bluetti-monitor-api")
    api_port = os.getenv("API_PORT", "8000")
    recommendations_url = f"http://{api_host}:{api_port}/recommendations"
    
    try:
        response = requests.get(recommendations_url, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Failed to get recommendations: HTTP {response.status_code}")
            return False
            
        recommendations_data = response.json()
        
        if not recommendations_data.get("success"):
            logger.warning(f"Recommendations API returned error: {recommendations_data.get('message', 'Unknown error')}")
            return False
            
        recommendations = recommendations_data.get("recommendations", {})
        reasoning = recommendations_data.get("reasoning", "")
        
        logger.info(f"Recommendations received: {reasoning}")
        
        # Control each device based on recommendations
        success_count = 0
        total_devices = 0
        
        for device_name, action in recommendations.items():
            total_devices += 1
            turn_on = action == "turn_on"
            
            if await control_device(device_name, turn_on):
                success_count += 1
            else:
                logger.error(f"Failed to control {device_name}")
        
        if success_count == total_devices:
            logger.info(f"Successfully controlled all {total_devices} devices")
            return True
        else:
            logger.warning(f"Only {success_count}/{total_devices} devices controlled successfully")
            return False
            
    except Exception as e:
        logger.error(f"Error getting recommendations and controlling devices: {e}")
        return False

async def background_worker():
    """Main worker loop that performs background polling with parallelized OCR"""
    # Initialize database
    db = BatteryDatabase(os.getenv("DATABASE_PATH", "./data/battery_readings.db"))
    await db.init_db()
    
    # Configuration
    polling_interval = int(os.getenv("POLLING_INTERVAL_SECONDS", 60))
    confidence_threshold = float(os.getenv("POLLING_CONFIDENCE_THRESHOLD", 0.5))  # Lower threshold
    
    logger.info(f"Worker starting: interval={polling_interval}s, min_confidence={confidence_threshold}, processes={mp.cpu_count()}")
    
    while True:
        try:
            # Check if screen is on
            test_image = capture_image()
            screen_analysis = analyze_screen_state(test_image)
            
            if screen_analysis.get("screen_state") == "off":
                logger.debug("Screen is off, skipping OCR polling")
                await asyncio.sleep(polling_interval)
                continue
            
            # Perform parallelized OCR analysis
            start_time = time.time()
            ocr_result = await parallel_ocr_analysis(test_image, num_captures=3)
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
                            ocr_method="worker_parallel",
                            total_attempts=ocr_result["total_attempts"],
                            raw_vote_data=ocr_result["vote_distribution"]
                        )
                        logger.info(f"Stored reading: {battery_percentage}% (conf: {confidence}, time: {processing_time:.2f}s)")
                        
                        # NEW: Get recommendations and control devices
                        await get_recommendations_and_control()
                        
                    else:
                        logger.debug(f"Plausibility check failed: {plausibility_msg}")
                else:
                    logger.debug(f"Low confidence reading skipped: {battery_percentage}% (confidence: {confidence})")
            else:
                logger.debug("No valid OCR results in background polling")
                
        except Exception as e:
            logger.error(f"Background worker error: {e}")
        
        await asyncio.sleep(polling_interval)

if __name__ == "__main__":
    logger.info("Starting Bluetti Monitor Worker")
    asyncio.run(background_worker())