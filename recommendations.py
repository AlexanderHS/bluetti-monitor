"""
Shared recommendation logic for Bluetti power management

This module contains the core logic for determining device control recommendations
based on battery status. It's used by both the API endpoint and the worker.
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging
from device_discovery import device_discovery

logger = logging.getLogger(__name__)


def calculate_device_recommendations(battery_percentage: int) -> Dict:
    """
    Calculate device control recommendations based on battery percentage with dynamic device discovery.
    Uses time-based segmentation when multiple inputs or outputs exist.

    Args:
        battery_percentage: Current battery percentage (0-100)

    Returns:
        Dictionary with device recommendations and reasoning
    """
    # Discover devices dynamically
    discovery_result = device_discovery.discover_devices()

    if not discovery_result["success"]:
        logger.warning(f"Device discovery failed: {discovery_result.get('error', 'unknown')}")
        return {
            "recommendations": {},
            "reasoning": "Device discovery failed - no recommendations available"
        }

    inputs = discovery_result["inputs"]
    outputs = discovery_result["outputs"]

    # Build recommendations based on battery percentage
    recommendations = {}

    if battery_percentage < 20:
        # Below 20% - turn on active input, turn off all outputs
        reasoning = f"Battery at {battery_percentage}% - critical low, charging needed"

        # Handle inputs
        if len(inputs) == 1:
            recommendations[inputs[0]["name"]] = "turn_on"
        elif len(inputs) > 1:
            # Multiple inputs: only turn on the active one, turn off others
            active_input = device_discovery.get_active_input()
            for inp in inputs:
                if inp["name"] == active_input["name"]:
                    recommendations[inp["name"]] = "turn_on"
                else:
                    recommendations[inp["name"]] = "turn_off"

        # Turn off all outputs
        for out in outputs:
            recommendations[out["name"]] = "turn_off"

    elif 20 <= battery_percentage < 60:
        # Between 20-60% - turn off everything (conservation)
        reasoning = f"Battery at {battery_percentage}% - low, conserving power"

        for inp in inputs:
            recommendations[inp["name"]] = "turn_off"
        for out in outputs:
            recommendations[out["name"]] = "turn_off"

    else:  # battery_percentage >= 60
        # Above 60% - turn off inputs, turn on active output
        reasoning = f"Battery at {battery_percentage}% - good level, using outputs"

        # Turn off all inputs
        for inp in inputs:
            recommendations[inp["name"]] = "turn_off"

        # Handle outputs
        if len(outputs) == 1:
            recommendations[outputs[0]["name"]] = "turn_on"
        elif len(outputs) > 1:
            # Multiple outputs: only turn on the active one, turn off others
            active_output = device_discovery.get_active_output()
            for out in outputs:
                if out["name"] == active_output["name"]:
                    recommendations[out["name"]] = "turn_on"
                else:
                    recommendations[out["name"]] = "turn_off"

    return {"recommendations": recommendations, "reasoning": reasoning}


def analyze_recent_readings_for_recommendations(
    readings_last_30min: List[Dict],
) -> Dict:
    """
    Analyze recent readings and generate recommendations

    Args:
        readings_last_30min: List of battery readings from the last 30 minutes

    Returns:
        Dictionary with recommendation result including status, recommendations, and metadata
    """
    # Case 1: No readings in the last 30 minutes - we're blind
    if not readings_last_30min:
        # Discover devices to turn everything off
        discovery_result = device_discovery.discover_devices()
        recommendations = {}

        if discovery_result["success"]:
            # Turn off all discovered devices
            for inp in discovery_result["inputs"]:
                recommendations[inp["name"]] = "turn_off"
            for out in discovery_result["outputs"]:
                recommendations[out["name"]] = "turn_off"

        return {
            "success": True,
            "status": "blind",
            "message": "No recent readings in the last 30 minutes",
            "recommendations": recommendations,
            "reasoning": "No recent battery data available - turning off all devices for safety",
            "last_reading_age_minutes": None,
            "battery_percentage": None,
        }

    # Case 2: We have recent readings - analyze battery percentage
    latest_reading = readings_last_30min[0]  # Most recent reading
    battery_percentage = latest_reading["battery_percentage"]
    last_reading_age_minutes = (
        datetime.now().timestamp() - latest_reading["timestamp"]
    ) / 60

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
        "last_reading_timestamp": latest_reading["timestamp"],
    }
