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
        self.min_tap_interval = os.getenv(
            "SWITCH_BOT_SECONDS_BETWEEN_TAPS", 15 * 60
        )  # 15 minutes in seconds

    async def initialize(self) -> bool:
        """Initialize SwitchBot connection and find the bot device"""
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
            self.switchbot = SwitchBot(token=self.token, secret=self.secret)
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
            self.last_tap_time = time.time()  # Record the tap time

            next_tap_time = time.time() + self.min_tap_interval
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
            return {"success": False, "error": "Tap failed", "details": str(e)}

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
