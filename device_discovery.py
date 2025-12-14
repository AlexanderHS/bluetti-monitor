"""
Device Discovery Module

Dynamically discovers and categorizes devices as inputs or outputs based on naming.
Provides time-based segmentation for multiple devices of the same type.
"""

import os
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DeviceDiscovery:
    """Handles dynamic device discovery and time-based segmentation"""

    def __init__(self):
        self.control_api_host = os.getenv("DEVICE_CONTROL_HOST", "10.0.0.109")
        self.control_api_port = os.getenv("DEVICE_CONTROL_PORT", "8084")
        self.devices_url = f"http://{self.control_api_host}:{self.control_api_port}/devices"

        # Cached device lists (refreshed on each discovery)
        self._inputs: List[Dict] = []
        self._outputs: List[Dict] = []
        self._last_discovery_time: Optional[float] = None

    def discover_devices(self) -> Dict:
        """
        Discover all available devices and categorize them as inputs or outputs.

        Returns:
            Dictionary with 'inputs' and 'outputs' lists, plus metadata
        """
        try:
            response = requests.get(
                self.devices_url,
                headers={"accept": "application/json"},
                timeout=5
            )

            if response.status_code != 200:
                logger.warning(f"Failed to discover devices: HTTP {response.status_code}")
                return {
                    "success": False,
                    "inputs": [],
                    "outputs": [],
                    "error": f"HTTP {response.status_code}"
                }

            data = response.json()
            devices = data.get("devices", [])

            # Categorize devices based on name (case-insensitive)
            inputs = []
            outputs = []

            for device in devices:
                name = device.get("name", "")
                if "input" in name.lower():
                    inputs.append(device)
                elif "output" in name.lower():
                    outputs.append(device)

            # Cache the discovered devices
            self._inputs = inputs
            self._outputs = outputs
            self._last_discovery_time = datetime.now().timestamp()

            logger.info(f"Device discovery: found {len(inputs)} input(s), {len(outputs)} output(s)")
            if inputs:
                logger.debug(f"  Inputs: {[d.get('name') for d in inputs]}")
            if outputs:
                logger.debug(f"  Outputs: {[d.get('name') for d in outputs]}")

            return {
                "success": True,
                "inputs": inputs,
                "outputs": outputs,
                "total_devices": len(devices),
                "discovery_time": self._last_discovery_time
            }

        except Exception as e:
            logger.error(f"Device discovery failed: {e}")
            return {
                "success": False,
                "inputs": [],
                "outputs": [],
                "error": str(e)
            }

    def get_active_device_index(self, device_count: int) -> int:
        """
        Determine which device should be active based on current time.

        For N devices, divides each hour into N equal segments.
        Example with 3 devices:
        - Minutes 0-19: Device 0
        - Minutes 20-39: Device 1
        - Minutes 40-59: Device 2

        Args:
            device_count: Total number of devices

        Returns:
            Index of the device that should be active (0-based)
        """
        if device_count <= 1:
            return 0

        current_minute = datetime.now().minute
        segment_duration = 60 // device_count
        active_index = current_minute // segment_duration

        # Ensure we don't exceed bounds (edge case for 60 minutes / N devices)
        return min(active_index, device_count - 1)

    def get_active_input(self) -> Optional[Dict]:
        """
        Get the currently active input device based on time segmentation.

        Returns:
            Active input device dict, or None if no inputs exist
        """
        if not self._inputs:
            return None

        if len(self._inputs) == 1:
            return self._inputs[0]

        active_index = self.get_active_device_index(len(self._inputs))
        return self._inputs[active_index]

    def get_active_output(self) -> Optional[Dict]:
        """
        Get the currently active output device based on time segmentation.

        Returns:
            Active output device dict, or None if no outputs exist
        """
        if not self._outputs:
            return None

        if len(self._outputs) == 1:
            return self._outputs[0]

        active_index = self.get_active_device_index(len(self._outputs))
        return self._outputs[active_index]

    def get_device_states(self) -> Dict[str, bool]:
        """
        Get the current state of all discovered devices.

        Returns:
            Dictionary mapping device names to their states (True=on, False=off)
        """
        try:
            response = requests.get(
                self.devices_url,
                headers={"accept": "application/json"},
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                devices = data.get("devices", [])

                states = {}
                for device in devices:
                    name = device.get("name")
                    if name:
                        states[name] = device.get("known_state", False)

                return states
            else:
                logger.warning(f"Failed to get device states: HTTP {response.status_code}")
                return {}

        except Exception as e:
            logger.warning(f"Error getting device states: {e}")
            return {}

    def is_any_input_on(self) -> bool:
        """Check if any input device is currently on"""
        states = self.get_device_states()
        return any(
            states.get(device.get("name"), False)
            for device in self._inputs
        )

    def is_any_output_on(self) -> bool:
        """Check if any output device is currently on"""
        states = self.get_device_states()
        return any(
            states.get(device.get("name"), False)
            for device in self._outputs
        )

    def get_segmentation_info(self) -> Dict:
        """
        Get information about current time-based segmentation.

        Returns:
            Dictionary with segmentation details for debugging
        """
        current_minute = datetime.now().minute

        input_info = None
        if self._inputs:
            input_count = len(self._inputs)
            input_segment_duration = 60 // input_count if input_count > 1 else 60
            input_active_index = self.get_active_device_index(input_count)
            input_info = {
                "count": input_count,
                "segment_duration_minutes": input_segment_duration,
                "active_index": input_active_index,
                "active_device": self._inputs[input_active_index].get("name") if input_count > 0 else None
            }

        output_info = None
        if self._outputs:
            output_count = len(self._outputs)
            output_segment_duration = 60 // output_count if output_count > 1 else 60
            output_active_index = self.get_active_device_index(output_count)
            output_info = {
                "count": output_count,
                "segment_duration_minutes": output_segment_duration,
                "active_index": output_active_index,
                "active_device": self._outputs[output_active_index].get("name") if output_count > 0 else None
            }

        return {
            "current_minute": current_minute,
            "inputs": input_info,
            "outputs": output_info,
            "last_discovery": self._last_discovery_time
        }


# Global instance for convenience
device_discovery = DeviceDiscovery()
