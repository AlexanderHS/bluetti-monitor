"""
GROQ OCR module for Bluetti battery percentage recognition (fallback)

This module uses GROQ's Llama 4 Scout model as a fallback when Gemini is unavailable.
"""
import os
import logging
import base64
from typing import Optional, Dict
import time

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Configure logging with ISO 8601 timestamps if not already configured
if not logging.root.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )

logger = logging.getLogger(__name__)

class GroqOCR:
    """GROQ-based OCR for battery percentage recognition (fallback)"""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.client = None
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Rate limiting: minimum 1 second between requests
        
        if not GROQ_AVAILABLE:
            logger.error("groq package not installed. Run: pip install groq")
            return
            
        if not self.api_key:
            logger.error("GROQ_API_KEY environment variable not set")
            return
            
        try:
            self.client = Groq(api_key=self.api_key)
            logger.info("GROQ OCR client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GROQ client: {e}")
    
    def is_available(self) -> bool:
        """Check if GROQ OCR is available and properly configured"""
        return GROQ_AVAILABLE and self.client is not None and self.api_key is not None
    
    def _apply_rate_limiting(self):
        """Apply rate limiting to avoid hitting API limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def analyze_image_with_groq(self, image_data: bytes) -> Dict:
        """
        Send image data to GROQ for battery percentage analysis
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary with success status, percentage, and metadata
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "GROQ OCR not available - check API key and package installation",
                "percentage": None
            }
        
        try:
            # Apply rate limiting
            self._apply_rate_limiting()
            
            # Encode image as base64 data URL
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            image_data_url = f"data:image/jpeg;base64,{image_b64}"
            
            # Create messages for GROQ
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What battery % is this? Reply with the number and nothing else, don't include the percentage sign or quotes. e.g. \"12\", \"2\", \"100\", \"99\""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ]
            
            # Generate response
            start_time = time.time()
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent results
                max_completion_tokens=10,  # Only need 1-3 digits
                top_p=1,
                stream=False,  # Don't use streaming for simplicity
                stop=None
            )
            processing_time = time.time() - start_time
            
            # Extract response text
            response_text = completion.choices[0].message.content.strip()
            
            # Parse percentage from response
            digits_only = ''.join(c for c in response_text if c.isdigit())
            
            # Validate percentage
            if not digits_only or not digits_only.isdigit():
                return {
                    "success": False,
                    "error": f"Invalid response format: '{response_text}' -> '{digits_only}'",
                    "percentage": None,
                    "processing_time": processing_time,
                    "raw_response": response_text
                }
            
            percentage = int(digits_only)
            if not (0 <= percentage <= 100):
                return {
                    "success": False,
                    "error": f"Invalid percentage value: {percentage}",
                    "percentage": None,
                    "processing_time": processing_time,
                    "raw_response": response_text
                }
            
            return {
                "success": True,
                "percentage": percentage,
                "processing_time": round(processing_time, 2),
                "error": None,
                "raw_response": response_text,
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"GROQ API call failed: {e}")
            return {
                "success": False,
                "error": f"GROQ API error: {str(e)}",
                "percentage": None,
                "processing_time": None
            }

# Global instance
groq_ocr = GroqOCR()

def get_battery_percentage_with_groq(image_data: bytes) -> Dict:
    """
    Convenience function to get battery percentage using GROQ
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Dictionary with success status, percentage, and metadata
    """
    return groq_ocr.analyze_image_with_groq(image_data)