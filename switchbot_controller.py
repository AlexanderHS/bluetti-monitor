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
        self.switchbot = None
        self.bot_device = None
        self._initialized = False
        self.last_tap_time = 0  # Track last tap timestamp
        self.last_successful_tap_time = 0  # Track last successful tap for suicide logic
        self.last_object_recreation_time = 0  # Track when SwitchBot object was last recreated
        self.object_recreation_interval = 30 * 60  # 30 minutes in seconds
        self.min_tap_interval = int(os.getenv(
            "SWITCH_BOT_SECONDS_BETWEEN_TAPS", 15 * 60
        ))  # Convert to int - seconds between taps
        
        # Log configuration at startup
        logger.info(f"SwitchBot controller initialized:")
        logger.info(f"  - Token configured: {'✅' if self.token else '❌'}")
        logger.info(f"  - Secret configured: {'✅' if self.secret else '❌'}")
        logger.info(f"  - Min tap interval: {self.min_tap_interval} seconds ({self.min_tap_interval/60:.1f} minutes)")
        logger.info(f"  - Object recreation interval: {self.object_recreation_interval/60:.0f} minutes")
        logger.info(f"  - Rate limiting: {'✅' if self.min_tap_interval > 0 else '❌ DISABLED'}")

    def _needs_object_recreation(self) -> bool:
        """Check if SwitchBot object needs recreation due to time interval"""
        current_time = time.time()
        time_since_recreation = current_time - self.last_object_recreation_time
        return time_since_recreation >= self.object_recreation_interval
    
    async def initialize(self) -> bool:
        """Initialize SwitchBot connection and find the bot device"""
        current_time = time.time()
        
        # Check if we need to recreate the SwitchBot object periodically
        if self._initialized and self._needs_object_recreation():
            logger.info(f"🔄 Recreating SwitchBot object after {self.object_recreation_interval/60:.0f} minutes")
            self._initialized = False
            self.switchbot = None
            self.bot_device = None
        
        if self._initialized:
            return True

        if not SWITCHBOT_AVAILABLE:
            logger.warning("python-switchbot package not available")
            return False

        if not self.token or not self.secret:
            logger.warning(
                "SWITCH_BOT_TOKEN or SWITCH_BOT_SECRET not configured - screen tapping disabled"
            )
            return False

        try:
            logger.info("Creating new SwitchBot API object...")
            self.switchbot = SwitchBot(token=self.token, secret=self.secret)
            self.last_object_recreation_time = time.time()  # Record when object was created
            
            devices = self.switchbot.devices()

            # Find the first bot device (since user mentioned they have only one)
            for device in devices:
                if hasattr(device, "press") or "Bot" in str(type(device)):
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
    
    def _parse_error_code(self, error_message: str) -> int:
        """Extract HTTP status code from error message"""
        error_str = str(error_message).lower()
        if "401" in error_str or "unauthorized" in error_str:
            return 401
        elif "429" in error_str or "too many requests" in error_str:
            return 429
        else:
            return 0  # Unknown error
    
    async def _handle_api_error(self, error_message: str) -> bool:
        """Handle specific API errors with appropriate recovery strategies"""
        error_code = self._parse_error_code(error_message)
        
        if error_code == 401:
            # Unauthorized - force immediate object recreation
            logger.warning("🔄 401 Unauthorized detected - forcing SwitchBot object recreation")
            self._initialized = False
            self.switchbot = None
            self.bot_device = None
            
            # Try to reinitialize immediately
            return await self.initialize()
            
        elif error_code == 429:
            # Rate limited - implement exponential backoff
            logger.warning("⏳ 429 Rate limit detected - implementing 5 minute cooldown")
            # Set last_tap_time to create a 5-minute delay
            self.last_tap_time = time.time() + (5 * 60)  # Add 5 minutes to current time
            return False
            
        else:
            # Unknown error - no specific recovery
            logger.error(f"Unknown SwitchBot API error: {error_message}")
            return False

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

        Args:
            force: If True, bypass rate limiting (use with caution)

        Returns:
            Dictionary with success status and details
        """
        if not await self.initialize():
            return {
                "success": False,
                "error": "SwitchBot not initialized",
                "details": "Check token/secret configuration or device availability",
            }

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
            logger.info(
                f"Tapping screen with SwitchBot{'(FORCED)' if force else ''}..."
            )
            self.bot_device.press()
            current_time = time.time()
            self.last_tap_time = current_time  # Record the tap time
            self.last_successful_tap_time = current_time  # Record successful tap for suicide logic

            next_tap_time = current_time + self.min_tap_interval
            logger.info(
                f"Screen tap completed - next tap allowed at {time.strftime('%H:%M:%S', time.localtime(next_tap_time))}"
            )

            return {
                "success": True,
                "message": "Screen tapped successfully",
                "tap_time": self.last_tap_time,
                "next_tap_allowed_at": next_tap_time,
                "forced": force,
            }

        except Exception as e:
            logger.error(f"Failed to tap screen with SwitchBot: {e}")
            
            # Try to handle specific API errors
            error_recovery_attempted = await self._handle_api_error(str(e))
            
            return {
                "success": False, 
                "error": "Tap failed", 
                "details": str(e),
                "error_code": self._parse_error_code(str(e)),
                "recovery_attempted": error_recovery_attempted
            }
    
    def check_suicide_condition(self) -> dict:
        """Check if container should exit due to prolonged SwitchBot failures"""
        max_failure_hours = int(os.getenv("SWITCHBOT_MAX_FAILURE_HOURS", 1))
        max_failure_seconds = max_failure_hours * 3600
        current_time = time.time()
        
        # If we've never had a successful tap, use object creation time as baseline
        baseline_time = self.last_successful_tap_time or self.last_object_recreation_time or current_time
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
        if not await self.initialize():
            return {
                "initialized": False,
                "device_found": False,
                "can_tap": False,
                "error": "Not initialized",
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
        }


# Global instance for shared use
switchbot_controller = SwitchBotController()
