"""
Shared SwitchBot controller for Bluetti screen management

This module handles SwitchBot operations including device initialization,
rate limiting, and screen tapping functionality.
"""

import os
import time
import logging
import asyncio
from typing import Optional

# Configure logging with ISO 8601 timestamps if not already configured
if not logging.root.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )

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
        self.device_name = os.getenv("SWITCH_BOT_DEVICE_NAME")  # Optional device selector
        self.last_tap_time = 0  # Track last tap timestamp for rate limiting
        self.last_successful_tap_time = 0  # Track last successful tap for suicide logic
        self.min_tap_interval = int(os.getenv(
            "SWITCH_BOT_SECONDS_BETWEEN_TAPS", 15 * 60
        ))  # Convert to int - seconds between taps
        
        # Log configuration at startup
        logger.info(f"SwitchBot controller initialized:")
        logger.info(f"  - Token configured: {'✅' if self.token else '❌'}")
        logger.info(f"  - Secret configured: {'✅' if self.secret else '❌'}")
        logger.info(f"  - Device name filter: {self.device_name or 'None (use first Bot found)'}")
        logger.info(f"  - Min tap interval: {self.min_tap_interval} seconds ({self.min_tap_interval/60:.1f} minutes)")
        logger.info(f"  - Rate limiting: {'✅' if self.min_tap_interval > 0 else '❌ DISABLED'}")

    def _parse_error_code(self, error_message: str) -> int:
        """Extract HTTP status code from error message"""
        error_str = str(error_message).lower()
        if "401" in error_str or "unauthorized" in error_str:
            return 401
        elif "429" in error_str or "too many requests" in error_str:
            return 429
        else:
            return 0  # Unknown error

    def get_time_until_next_tap(self) -> float:
        """Get time in seconds until next tap is allowed"""
        current_time = time.time()
        time_since_last_tap = current_time - self.last_tap_time
        return max(0, self.min_tap_interval - time_since_last_tap)

    def can_tap_screen(self) -> bool:
        """Check if enough time has passed since last tap (15 minute minimum)"""
        return self.get_time_until_next_tap() == 0

    async def tap_screen(self, force: bool = False) -> dict:
        """
        Tap the screen using SwitchBot to turn it on
        Creates fresh SwitchBot object each time - no caching or state management

        Args:
            force: If True, bypass rate limiting (use with caution)

        Returns:
            Dictionary with success status and details
        """
        # Check basic configuration
        if not SWITCHBOT_AVAILABLE:
            return {
                "success": False,
                "error": "SwitchBot package not available",
                "details": "Install python-switchbot package"
            }

        if not self.token or not self.secret:
            return {
                "success": False,
                "error": "SwitchBot not configured",
                "details": "Set SWITCH_BOT_TOKEN and SWITCH_BOT_SECRET environment variables"
            }

        # Check rate limiting
        if not force and not self.can_tap_screen():
            time_remaining = self.get_time_until_next_tap()
            minutes_remaining = time_remaining / 60
            return {
                "success": False,
                "error": "Rate limited",
                "details": f"Next tap allowed in {minutes_remaining:.1f} minutes",
                "time_until_next_tap_seconds": time_remaining,
            }

        try:
            logger.info(f"Tapping screen with SwitchBot{'(FORCED)' if force else ''}...")
            
            # Create fresh SwitchBot object each time (no caching)
            switchbot = SwitchBot(token=self.token, secret=self.secret)
            
            # Find target device
            target_device = None
            devices = switchbot.devices()
            
            for device in devices:
                if hasattr(device, "press") or "Bot" in str(type(device)):
                    if self.device_name:
                        # If device name specified, match by name/ID
                        if self.device_name in str(device):
                            target_device = device
                            logger.info(f"Found specified SwitchBot device: {device}")
                            break
                    else:
                        # No device name specified, use first Bot found
                        target_device = device
                        logger.info(f"Using first SwitchBot device: {device}")
                        break
            
            if not target_device:
                device_filter = f" matching '{self.device_name}'" if self.device_name else ""
                return {
                    "success": False,
                    "error": "No SwitchBot device found",
                    "details": f"No Bot device found{device_filter} - check device name or account"
                }
            
            # Tap the device
            target_device.press()
            
            # Record successful tap
            current_time = time.time()
            self.last_tap_time = current_time
            self.last_successful_tap_time = current_time
            
            next_tap_time = current_time + self.min_tap_interval
            logger.info(f"Screen tap completed - next tap allowed at {time.strftime('%H:%M:%S', time.localtime(next_tap_time))}")

            return {
                "success": True,
                "message": "Screen tapped successfully",
                "tap_time": self.last_tap_time,
                "next_tap_allowed_at": next_tap_time,
                "forced": force,
                "device_used": str(target_device)
            }

        except Exception as e:
            logger.error(f"Failed to tap screen with SwitchBot: {e}")
            
            error_code = self._parse_error_code(str(e))
            
            # Handle rate limiting by extending the cooldown
            if error_code == 429:
                logger.warning("⏳ 429 Rate limit detected - implementing 5 minute cooldown")
                self.last_tap_time = time.time() + (5 * 60)  # Add 5 minutes
            
            return {
                "success": False, 
                "error": "Tap failed", 
                "details": str(e),
                "error_code": error_code
            }
    
    def check_suicide_condition(self) -> dict:
        """Check if container should exit due to prolonged SwitchBot failures"""
        max_failure_hours = int(os.getenv("SWITCHBOT_MAX_FAILURE_HOURS", 1))
        max_failure_seconds = max_failure_hours * 3600
        current_time = time.time()
        
        # If we've never had a successful tap, use current time as baseline
        baseline_time = self.last_successful_tap_time or current_time
        time_since_last_success = current_time - baseline_time
        
        result = {
            "should_exit": time_since_last_success >= max_failure_seconds,
            "time_since_last_success_hours": time_since_last_success / 3600,
            "max_failure_hours": max_failure_hours,
            "last_successful_tap_time": self.last_successful_tap_time,
            "baseline_time": baseline_time
        }
        
        if result["should_exit"]:
            logger.critical(f"💀 CONTAINER SUICIDE: No successful SwitchBot tap for {result['time_since_last_success_hours']:.1f} hours (limit: {max_failure_hours} hours)")
        
        return result

    async def get_status(self) -> dict:
        """Get SwitchBot controller status"""
        time_until_next = self.get_time_until_next_tap()
        configured = bool(self.token and self.secret and SWITCHBOT_AVAILABLE)

        return {
            "configured": configured,
            "package_available": SWITCHBOT_AVAILABLE,
            "token_configured": bool(self.token),
            "secret_configured": bool(self.secret),
            "device_name_filter": self.device_name,
            "can_tap": time_until_next == 0,
            "time_until_next_tap_seconds": time_until_next,
            "time_until_next_tap_minutes": round(time_until_next / 60, 1),
            "last_tap_time": self.last_tap_time if self.last_tap_time > 0 else None,
            "last_successful_tap_time": self.last_successful_tap_time if self.last_successful_tap_time > 0 else None,
            "rate_limit_minutes": self.min_tap_interval / 60,
        }


# Global instance for shared use
switchbot_controller = SwitchBotController()
