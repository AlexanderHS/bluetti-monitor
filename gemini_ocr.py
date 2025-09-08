"""
Gemini OCR module for Bluetti battery percentage recognition

This module uses Google's Gemini 2.0 Flash Lite model for accurate 
battery percentage reading from LCD screen images.
"""
import os
import logging
import requests
from typing import Optional, Dict
import time

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

class GeminiOCR:
    """Gemini-based OCR for battery percentage recognition"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-2.0-flash-lite"
        self.client = None
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Rate limiting: minimum 1 second between requests
        
        if not GEMINI_AVAILABLE:
            logger.error("google-genai package not installed. Run: pip install google-genai")
            return
            
        if not self.api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            return
            
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Gemini OCR client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
    
    def is_available(self) -> bool:
        """Check if Gemini OCR is available and properly configured"""
        return GEMINI_AVAILABLE and self.client is not None and self.api_key is not None
    
    def _apply_rate_limiting(self):
        """Apply rate limiting to avoid hitting API limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_battery_percentage_from_endpoint(self, endpoint_url: str) -> Optional[Dict]:
        """
        Get battery percentage by fetching image from endpoint and sending to Gemini
        
        Args:
            endpoint_url: URL to fetch the flipped/cropped LCD image from
            
        Returns:
            Dictionary with success status, percentage, and metadata
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Gemini OCR not available - check API key and package installation",
                "percentage": None
            }
        
        try:
            # Apply rate limiting
            self._apply_rate_limiting()
            
            # Fetch image from endpoint
            response = requests.get(endpoint_url, timeout=10)
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to fetch image: HTTP {response.status_code}",
                    "percentage": None
                }
            
            # Check if response is actually an image
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                # Might be JSON error response
                try:
                    error_data = response.json()
                    return {
                        "success": False,
                        "error": f"Endpoint returned error: {error_data}",
                        "percentage": None
                    }
                except:
                    return {
                        "success": False,
                        "error": f"Endpoint returned non-image content: {content_type}",
                        "percentage": None
                    }
            
            # Send image to Gemini
            return self._analyze_image_with_gemini(response.content)
            
        except Exception as e:
            logger.error(f"Error in get_battery_percentage_from_endpoint: {e}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "percentage": None
            }
    
    def _analyze_image_with_gemini(self, image_data: bytes) -> Dict:
        """
        Send image data to Gemini for battery percentage analysis
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary with success status, percentage, and metadata
        """
        try:
            # Create content for Gemini
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text="What battery % is this? Reply with the number and nothing else, don't include the percentage sign or quotes. e.g. \"12\", \"2\", \"100\", \"99\""
                        ),
                        types.Part.from_bytes(
                            data=image_data,
                            mime_type="image/jpeg"
                        )
                    ],
                ),
            ]
            
            # Configure structured response
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    required=["percentage"],
                    properties={
                        "percentage": genai.types.Schema(
                            type=genai.types.Type.NUMBER,
                        ),
                    },
                ),
            )
            
            # Generate response
            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )
            processing_time = time.time() - start_time
            
            # Parse response
            import json
            response_data = json.loads(response.text)
            percentage = response_data.get("percentage")
            
            # Validate percentage
            if not isinstance(percentage, (int, float)):
                return {
                    "success": False,
                    "error": f"Invalid response format: {response_data}",
                    "percentage": None,
                    "processing_time": processing_time
                }
            
            percentage = int(percentage)
            if not (0 <= percentage <= 100):
                return {
                    "success": False,
                    "error": f"Invalid percentage value: {percentage}",
                    "percentage": None,
                    "processing_time": processing_time
                }
            
            return {
                "success": True,
                "percentage": percentage,
                "processing_time": round(processing_time, 2),
                "error": None,
                "raw_response": response_data
            }
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return {
                "success": False,
                "error": f"Gemini API error: {str(e)}",
                "percentage": None,
                "processing_time": None
            }

# Global instance
gemini_ocr = GeminiOCR()

def get_battery_percentage_from_capture_flip(base_url: str = "http://localhost:8000") -> Dict:
    """
    Convenience function to get battery percentage from /capture/flip endpoint
    
    Args:
        base_url: Base URL of the Bluetti Monitor API
        
    Returns:
        Dictionary with success status, percentage, and metadata
    """
    endpoint_url = f"{base_url}/capture/flip"
    return gemini_ocr.get_battery_percentage_from_endpoint(endpoint_url)