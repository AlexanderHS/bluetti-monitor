"""
InfluxDB metrics writer for Bluetti Monitor

Provides optional InfluxDB integration for time-series tracking of:
- Battery percentage and OCR confidence
- Camera connectivity status
- SwitchBot connectivity and rate limiting
- Device states (input/output on/off)

Disabled when INFLUXDB_TOKEN is not set. Gracefully handles connection failures.
"""

import os
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import InfluxDB client
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import ASYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    logger.warning("influxdb-client not installed - InfluxDB metrics disabled")
    INFLUXDB_AVAILABLE = False


class InfluxDBWriter:
    """Handle InfluxDB metric writes with graceful error handling"""

    def __init__(self):
        self.url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = os.getenv("INFLUXDB_TOKEN", "")
        self.org = os.getenv("INFLUXDB_ORG", "home")
        self.bucket = os.getenv("INFLUXDB_BUCKET", "bluetti")

        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.enabled = False

        # Only initialize if token is configured
        if self.token and INFLUXDB_AVAILABLE:
            try:
                self.client = InfluxDBClient(
                    url=self.url,
                    token=self.token,
                    org=self.org
                )
                # Use asynchronous writes for non-blocking operation
                self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)
                self.enabled = True
                logger.info(f"InfluxDB metrics enabled: {self.url}, bucket={self.bucket}")
            except Exception as e:
                logger.warning(f"Failed to initialize InfluxDB client: {e}")
                self.enabled = False
        else:
            if not self.token:
                logger.debug("InfluxDB metrics disabled: INFLUXDB_TOKEN not set")
            elif not INFLUXDB_AVAILABLE:
                logger.debug("InfluxDB metrics disabled: influxdb-client not installed")

    def _write_point(self, point: "Point"):
        """Write a point to InfluxDB with error handling"""
        if not self.enabled or not self.write_api:
            return

        try:
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        except Exception as e:
            logger.warning(f"Failed to write metric to InfluxDB: {e}")

    def write_battery_reading(
        self,
        battery_percentage: int,
        ocr_confidence: float,
        ocr_strategy: str = "unknown"
    ):
        """
        Write battery reading metrics to InfluxDB

        Args:
            battery_percentage: Current battery level (0-100)
            ocr_confidence: OCR confidence score (0-1)
            ocr_strategy: OCR method used ("template" or "llm")
        """
        if not self.enabled:
            return

        point = (
            Point("battery_reading")
            .tag("ocr_strategy", ocr_strategy)
            .field("battery_percentage", battery_percentage)
            .field("ocr_confidence", ocr_confidence)
            .time(datetime.utcnow(), WritePrecision.NS)
        )
        self._write_point(point)
        logger.debug(f"InfluxDB: battery={battery_percentage}%, confidence={ocr_confidence:.2f}, strategy={ocr_strategy}")

    def write_camera_status(self, reachable: bool):
        """
        Write camera connectivity status

        Args:
            reachable: True if camera responded successfully
        """
        if not self.enabled:
            return

        point = (
            Point("camera_status")
            .field("reachable", 1 if reachable else 0)
            .time(datetime.utcnow(), WritePrecision.NS)
        )
        self._write_point(point)
        logger.debug(f"InfluxDB: camera_reachable={1 if reachable else 0}")

    def write_switchbot_status(
        self,
        reachable: bool,
        rate_limited: bool = False
    ):
        """
        Write SwitchBot connectivity status

        Args:
            reachable: True if SwitchBot API responded successfully (not rate limited, not failed)
            rate_limited: True if SwitchBot returned 429 (not a failure, just throttled)
        """
        if not self.enabled:
            return

        point = (
            Point("switchbot_status")
            .field("reachable", 1 if reachable else 0)
            .field("rate_limited", 1 if rate_limited else 0)
            .time(datetime.utcnow(), WritePrecision.NS)
        )
        self._write_point(point)
        logger.debug(f"InfluxDB: switchbot_reachable={1 if reachable else 0}, rate_limited={1 if rate_limited else 0}")

    def write_device_state(self, device_name: str, is_on: bool):
        """
        Write device state change

        Args:
            device_name: Name of the device (e.g., "input", "output_1")
            is_on: True if device is on
        """
        if not self.enabled:
            return

        # Determine device type from name for tagging
        device_type = "unknown"
        if "input" in device_name.lower():
            device_type = "input"
        elif "output" in device_name.lower():
            device_type = "output"

        point = (
            Point("device_state")
            .tag("device_name", device_name)
            .tag("device_type", device_type)
            .field("is_on", 1 if is_on else 0)
            .time(datetime.utcnow(), WritePrecision.NS)
        )
        self._write_point(point)
        logger.debug(f"InfluxDB: device={device_name}, is_on={1 if is_on else 0}")

    def close(self):
        """Close the InfluxDB client connection"""
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error closing InfluxDB client: {e}")


# Global instance for shared use
influxdb_writer = InfluxDBWriter()
