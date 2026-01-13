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

# Import InfluxDB writer (lazy import to avoid circular dependencies)
_influxdb_writer = None

def _get_influxdb_writer():
    """Lazy load InfluxDB writer to avoid circular imports"""
    global _influxdb_writer
    if _influxdb_writer is None:
        try:
            from influxdb_writer import influxdb_writer
            _influxdb_writer = influxdb_writer
        except ImportError:
            pass
    return _influxdb_writer

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
        self.last_successful_tap_time = 0  # Track last successful tap for metrics/debugging

        # Failure state tracking for suicide logic
        # None = healthy (no active failure), timestamp = when failure started
        # Only REAL failures (network, API errors) set this - intentional skips do NOT
        self.failure_since: Optional[float] = None

        # Adaptive tap intervals based on activity (in seconds)
        # Active: When input/output on OR daylight hours (rapid changes possible)
        # Idle: When both devices off AND nighttime (only slow self-discharge)
        self.active_interval = int(os.getenv("ACTIVE_TAP_INTERVAL", 300))  # 5 minutes
        self.idle_interval = int(os.getenv("IDLE_TAP_INTERVAL", 1800))  # 30 minutes

        # Daylight hours for solar activity (24-hour format)
        self.daylight_start_hour = int(os.getenv("DAYLIGHT_START_HOUR", 7))  # 7 AM
        self.daylight_end_hour = int(os.getenv("DAYLIGHT_END_HOUR", 19))  # 7 PM

        # Failure timeout for container suicide logic (must be > idle_interval)
        # Default 1 hour - only triggers after prolonged REAL failures, not idle skips
        self.failure_timeout_hours = float(os.getenv("SWITCHBOT_FAILURE_TIMEOUT_HOURS", 1.0))

        # Log configuration at startup
        logger.debug(f"SwitchBot controller initialized:")
        logger.debug(f"  - Token configured: {'✅' if self.token else '❌'}")
        logger.debug(f"  - Secret configured: {'✅' if self.secret else '❌'}")
        logger.debug(f"  - Device name filter: {self.device_name or 'None (use first Bot found)'}")
        logger.debug(f"  - Adaptive tap intervals:")
        logger.debug(f"    • Active (input ON OR output ON OR daylight): {self.active_interval}s ({self.active_interval/60:.1f} min)")
        logger.debug(f"    • Idle (all OFF + nighttime): {self.idle_interval}s ({self.idle_interval/60:.1f} min)")
        logger.debug(f"    • Daylight hours: {self.daylight_start_hour}:00 - {self.daylight_end_hour}:00")
        logger.debug(f"  - Failure timeout: {self.failure_timeout_hours} hours (suicide only on real API failures)")

    def _is_daylight_hours(self) -> bool:
        """
        Check if current time is within daylight hours (solar activity expected)

        Returns:
            bool: True if within daylight hours
        """
        from datetime import datetime
        current_hour = datetime.now().hour
        return self.daylight_start_hour <= current_hour < self.daylight_end_hour

    def get_dynamic_tap_interval(self, input_on: bool = False, output_on: bool = False) -> int:
        """
        Calculate tap interval based on device states and time of day.

        Active monitoring (frequent taps) when ANY of these are true:
        - Input device is ON (charging active)
        - Output device is ON (discharging active)
        - Daylight hours (solar could start at any moment)

        Idle monitoring (infrequent taps) only when:
        - Both devices OFF AND nighttime (only slow self-discharge)

        Args:
            input_on: True if input device is currently on
            output_on: True if output device is currently on

        Returns:
            int: Seconds until next tap is allowed
        """
        is_daylight = self._is_daylight_hours()

        # Triple-OR logic: Active if any condition is true
        if input_on or output_on or is_daylight:
            return self.active_interval
        else:
            return self.idle_interval

    def _parse_error_code(self, error_message: str) -> int:
        """Extract HTTP status code from error message"""
        error_str = str(error_message).lower()
        if "401" in error_str or "unauthorized" in error_str:
            return 401
        elif "429" in error_str or "too many requests" in error_str:
            return 429
        else:
            return 0  # Unknown error

    def get_time_until_next_tap(self, input_on: bool = False, output_on: bool = False) -> float:
        """
        Get time in seconds until next tap is allowed

        Args:
            input_on: True if input device is currently on
            output_on: True if output device is currently on

        Returns:
            float: Seconds until next tap is allowed
        """
        current_time = time.time()
        time_since_last_tap = current_time - self.last_tap_time
        required_interval = self.get_dynamic_tap_interval(input_on, output_on)
        return max(0, required_interval - time_since_last_tap)

    def can_tap_screen(self, input_on: bool = False, output_on: bool = False) -> bool:
        """
        Check if enough time has passed since last tap

        Args:
            input_on: True if input device is currently on
            output_on: True if output device is currently on

        Returns:
            bool: True if tap is allowed
        """
        return self.get_time_until_next_tap(input_on, output_on) == 0

    async def tap_screen(self, force: bool = False, input_on: bool = False, output_on: bool = False) -> dict:
        """
        Tap the screen using SwitchBot to turn it on
        Creates fresh SwitchBot object each time - no caching or state management

        Args:
            force: If True, bypass rate limiting (use with caution)
            input_on: Current state of input devices (for rate limit calculation)
            output_on: Current state of output devices (for rate limit calculation)

        Returns:
            Dictionary with success status and details

        Note: Pass current device states to ensure rate limiting uses the correct interval.
              When mode switches from IDLE to ACTIVE, the shorter ACTIVE interval is used
              immediately, allowing a tap if it's been >5 min since last tap (not >30 min).
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

        # Check rate limiting using current device states for accurate interval calculation
        # This ensures mode switches (IDLE→ACTIVE) immediately use the new interval
        if not force and not self.can_tap_screen(input_on=input_on, output_on=output_on):
            time_remaining = self.get_time_until_next_tap(input_on=input_on, output_on=output_on)
            minutes_remaining = time_remaining / 60
            # Rate limiting is NOT a failure - it means our tap interval policy is working.
            # The 'rate_limited' flag distinguishes this from actual API/network failures.
            return {
                "success": False,
                "rate_limited": True,  # Distinguish from actual failures
                "error": "Rate limited",
                "details": f"Next tap allowed in {minutes_remaining:.1f} minutes",
                "time_until_next_tap_seconds": time_remaining,
            }

        try:
            logger.debug(f"Tapping screen with SwitchBot{'(FORCED)' if force else ''}...")
            
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
                            logger.debug(f"Found specified SwitchBot device: {device}")
                            break
                    else:
                        # No device name specified, use first Bot found
                        target_device = device
                        logger.debug(f"Using first SwitchBot device: {device}")
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

            # Clear failure state - successful tap proves SwitchBot is working
            if self.failure_since is not None:
                failure_duration = (current_time - self.failure_since) / 60
                logger.info(f"SwitchBot recovered after {failure_duration:.1f} minutes of failure state")
            self.failure_since = None

            # Log next tap time (using default intervals for logging)
            next_interval = self.get_dynamic_tap_interval()
            next_tap_time = current_time + next_interval
            logger.debug(f"Screen tap completed - next tap allowed at {time.strftime('%H:%M:%S', time.localtime(next_tap_time))} ({next_interval/60:.1f} min)")

            # Write SwitchBot success metrics to InfluxDB
            writer = _get_influxdb_writer()
            if writer:
                writer.write_switchbot_status(reachable=True, rate_limited=False)

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

            # Write SwitchBot failure/rate-limit metrics to InfluxDB
            writer = _get_influxdb_writer()
            if writer:
                if error_code == 429:
                    # Rate limiting is NOT a failure - SwitchBot is reachable but throttling us
                    writer.write_switchbot_status(reachable=True, rate_limited=True)
                else:
                    # Actual failure - SwitchBot API unreachable
                    writer.write_switchbot_status(reachable=False, rate_limited=False)

            # Handle rate limiting by extending the cooldown
            if error_code == 429:
                logger.warning("429 Rate limit detected - implementing 5 minute cooldown")
                self.last_tap_time = time.time() + (5 * 60)  # Add 5 minutes
                # 429 is NOT a real failure - API is reachable, just throttling us
                # Do NOT set failure_since for this case
            else:
                # Real failure (network error, auth error, etc.) - start tracking failure state
                if self.failure_since is None:
                    self.failure_since = time.time()
                    logger.warning(f"SwitchBot entered failure state - will trigger suicide after {self.failure_timeout_hours} hours if not recovered")

            return {
                "success": False,
                "error": "Tap failed",
                "details": str(e),
                "error_code": error_code
            }
    
    def check_suicide_condition(self) -> dict:
        """
        Check if container should exit due to prolonged SwitchBot failures.

        Uses a state-machine approach:
        - failure_since = None: Healthy state (no active failure)
        - failure_since = timestamp: In failure state since that time

        State transitions:
        - Healthy -> Failure: On REAL API/network failure (not 429, not rate-limit skip)
        - Failure -> Healthy: On successful tap
        - Failure -> SUICIDE: When failure duration exceeds threshold

        IMPORTANT: Intentional rate-limit skips (idle mode, cooldowns) do NOT
        trigger or affect the failure state. They are neutral events.
        """
        failure_timeout_seconds = self.failure_timeout_hours * 3600
        current_time = time.time()

        # If not in failure state, no suicide needed
        if self.failure_since is None:
            return {
                "should_exit": False,
                "in_failure_state": False,
                "failure_duration_hours": 0.0,
                "failure_timeout_hours": self.failure_timeout_hours,
                "failure_since": None
            }

        # In failure state - check if timeout exceeded
        failure_duration = current_time - self.failure_since
        failure_duration_hours = failure_duration / 3600
        should_exit = failure_duration >= failure_timeout_seconds

        result = {
            "should_exit": should_exit,
            "in_failure_state": True,
            "failure_duration_hours": failure_duration_hours,
            "failure_timeout_hours": self.failure_timeout_hours,
            "failure_since": self.failure_since
        }

        if should_exit:
            logger.critical(f"CONTAINER SUICIDE: SwitchBot in failure state for {failure_duration_hours:.2f} hours (limit: {self.failure_timeout_hours} hours)")
            logger.critical("This indicates a REAL API/network failure, not idle mode rate limiting")
        elif failure_duration_hours > 0.1:  # Only log if in failure for > 6 minutes
            logger.warning(f"SwitchBot failure state: {failure_duration_hours:.2f}/{self.failure_timeout_hours} hours until suicide")

        return result

    async def get_status(self, input_on: bool = False, output_on: bool = False) -> dict:
        """
        Get SwitchBot controller status

        Args:
            input_on: True if input device is currently on
            output_on: True if output device is currently on
        """
        time_until_next = self.get_time_until_next_tap(input_on, output_on)
        configured = bool(self.token and self.secret and SWITCHBOT_AVAILABLE)
        is_daylight = self._is_daylight_hours()
        current_interval = self.get_dynamic_tap_interval(input_on, output_on)

        return {
            "configured": configured,
            "package_available": SWITCHBOT_AVAILABLE,
            "token_configured": bool(self.token),
            "secret_configured": bool(self.secret),
            "device_name_filter": self.device_name,
            "can_tap": time_until_next == 0,
            "time_until_next_tap_seconds": time_until_next,
            "time_until_next_tap_minutes": round(time_until_next / 60, 1),
            "current_interval_minutes": current_interval / 60,
            "is_daylight": is_daylight,
            "is_active_mode": input_on or output_on or is_daylight,
            "last_tap_time": self.last_tap_time if self.last_tap_time > 0 else None,
            "last_successful_tap_time": self.last_successful_tap_time if self.last_successful_tap_time > 0 else None,
            "in_failure_state": self.failure_since is not None,
            "failure_since": self.failure_since,
            "failure_timeout_hours": self.failure_timeout_hours,
        }


# Global instance for shared use
switchbot_controller = SwitchBotController()
