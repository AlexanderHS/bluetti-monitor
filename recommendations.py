"""
Shared recommendation logic for Bluetti power management

This module contains the core logic for determining device control recommendations
based on battery status. It's used by both the API endpoint and the worker.

Hysteresis Logic (with input and output hysteresis):
- Below INPUT_LOW_THRESHOLD (20%): Critical low - charge (inputs on, outputs off)
- INPUT_LOW_THRESHOLD to INPUT_HIGH_THRESHOLD (20-40%): Input hysteresis zone (maintain input state, outputs off)
- INPUT_HIGH_THRESHOLD to OUTPUT_LOW_THRESHOLD (40-60%): Conservation mode (everything off)
- OUTPUT_LOW_THRESHOLD to OUTPUT_HIGH_THRESHOLD (60-80%): Output hysteresis zone (inputs off, maintain output state)
- Above OUTPUT_HIGH_THRESHOLD (80%): Use outputs (inputs off, outputs on)

This prevents oscillation when battery hovers around either the charging or discharging thresholds.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
import logging
from device_discovery import device_discovery

logger = logging.getLogger(__name__)

# Load hysteresis thresholds from environment
OUTPUT_HIGH_THRESHOLD = int(os.getenv("OUTPUT_HIGH_THRESHOLD", 80))
OUTPUT_LOW_THRESHOLD = int(os.getenv("OUTPUT_LOW_THRESHOLD", 60))
INPUT_LOW_THRESHOLD = int(os.getenv("INPUT_LOW_THRESHOLD", 20))
INPUT_HIGH_THRESHOLD = int(os.getenv("INPUT_HIGH_THRESHOLD", 40))


def calculate_device_recommendations(
    battery_percentage: int,
    current_output_on: Optional[bool] = None,
    current_input_on: Optional[bool] = None
) -> Dict:
    """
    Calculate device control recommendations based on battery percentage with dynamic device discovery.
    Uses time-based segmentation when multiple inputs or outputs exist.

    Implements hysteresis for both inputs and outputs to prevent oscillation:
    - Below INPUT_LOW_THRESHOLD: Critical - charge (inputs on, outputs off)
    - INPUT_LOW_THRESHOLD to INPUT_HIGH_THRESHOLD: Input hysteresis zone (maintain input state, outputs off)
    - INPUT_HIGH_THRESHOLD to OUTPUT_LOW_THRESHOLD: Conservation (everything off)
    - OUTPUT_LOW_THRESHOLD to OUTPUT_HIGH_THRESHOLD: Output hysteresis zone (inputs off, maintain output state)
    - Above OUTPUT_HIGH_THRESHOLD: Use outputs (inputs off, outputs on)

    Args:
        battery_percentage: Current battery percentage (0-100)
        current_output_on: Current state of outputs (True=on, False=off, None=unknown)
                          Used for output hysteresis zone decisions
        current_input_on: Current state of inputs (True=on, False=off, None=unknown)
                         Used for input hysteresis zone decisions

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

    # Build recommendations based on battery percentage with input and output hysteresis
    recommendations = {}

    if battery_percentage < INPUT_LOW_THRESHOLD:
        # Below INPUT_LOW_THRESHOLD - turn on active input, turn off all outputs (critical low)
        reasoning = f"Battery at {battery_percentage}% - below {INPUT_LOW_THRESHOLD}%, critical low, charging needed"

        # Handle inputs - turn on active input
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

    elif INPUT_LOW_THRESHOLD <= battery_percentage < INPUT_HIGH_THRESHOLD:
        # Input hysteresis zone - maintain current input state, outputs always off
        # Turn off all outputs first
        for out in outputs:
            recommendations[out["name"]] = "turn_off"

        # Determine input action based on current state
        if current_input_on is None:
            # Unknown state - query current device states
            device_states = device_discovery.get_device_states()
            current_input_on = device_discovery.is_any_input_on(device_states)
            logger.debug(f"Input hysteresis zone: queried input state = {current_input_on}")

        if current_input_on:
            # Inputs are currently on - keep them on (maintain state, continue charging)
            reasoning = f"Battery at {battery_percentage}% - input hysteresis zone ({INPUT_LOW_THRESHOLD}-{INPUT_HIGH_THRESHOLD}%), maintaining inputs ON"
            if len(inputs) == 1:
                recommendations[inputs[0]["name"]] = "turn_on"
            elif len(inputs) > 1:
                active_input = device_discovery.get_active_input()
                for inp in inputs:
                    if inp["name"] == active_input["name"]:
                        recommendations[inp["name"]] = "turn_on"
                    else:
                        recommendations[inp["name"]] = "turn_off"
        else:
            # Inputs are currently off - keep them off (maintain state)
            reasoning = f"Battery at {battery_percentage}% - input hysteresis zone ({INPUT_LOW_THRESHOLD}-{INPUT_HIGH_THRESHOLD}%), maintaining inputs OFF"
            for inp in inputs:
                recommendations[inp["name"]] = "turn_off"

    elif INPUT_HIGH_THRESHOLD <= battery_percentage < OUTPUT_LOW_THRESHOLD:
        # Between input high threshold and output low threshold - turn off everything (conservation)
        reasoning = f"Battery at {battery_percentage}% - conservation zone ({INPUT_HIGH_THRESHOLD}-{OUTPUT_LOW_THRESHOLD}%), all devices off"

        for inp in inputs:
            recommendations[inp["name"]] = "turn_off"
        for out in outputs:
            recommendations[out["name"]] = "turn_off"

    elif OUTPUT_LOW_THRESHOLD <= battery_percentage < OUTPUT_HIGH_THRESHOLD:
        # Output hysteresis zone - maintain current output state, inputs always off
        for inp in inputs:
            recommendations[inp["name"]] = "turn_off"

        # Determine output action based on current state
        if current_output_on is None:
            # Unknown state - query current device states
            device_states = device_discovery.get_device_states()
            current_output_on = device_discovery.is_any_output_on(device_states)
            logger.debug(f"Output hysteresis zone: queried output state = {current_output_on}")

        if current_output_on:
            # Outputs are currently on - keep them on (maintain state)
            reasoning = f"Battery at {battery_percentage}% - output hysteresis zone ({OUTPUT_LOW_THRESHOLD}-{OUTPUT_HIGH_THRESHOLD}%), maintaining outputs ON"
            if len(outputs) == 1:
                recommendations[outputs[0]["name"]] = "turn_on"
            elif len(outputs) > 1:
                active_output = device_discovery.get_active_output()
                for out in outputs:
                    if out["name"] == active_output["name"]:
                        recommendations[out["name"]] = "turn_on"
                    else:
                        recommendations[out["name"]] = "turn_off"
        else:
            # Outputs are currently off - keep them off (maintain state)
            reasoning = f"Battery at {battery_percentage}% - output hysteresis zone ({OUTPUT_LOW_THRESHOLD}-{OUTPUT_HIGH_THRESHOLD}%), maintaining outputs OFF"
            for out in outputs:
                recommendations[out["name"]] = "turn_off"

    else:  # battery_percentage >= OUTPUT_HIGH_THRESHOLD
        # Above high threshold - turn off inputs, turn on active output
        reasoning = f"Battery at {battery_percentage}% - above {OUTPUT_HIGH_THRESHOLD}%, using outputs"

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
    current_output_on: Optional[bool] = None,
    current_input_on: Optional[bool] = None,
) -> Dict:
    """
    Analyze recent readings and generate recommendations

    Args:
        readings_last_30min: List of battery readings from the last 30 minutes
        current_output_on: Current state of outputs (True=on, False=off, None=unknown)
                          Used for output hysteresis zone decisions
        current_input_on: Current state of inputs (True=on, False=off, None=unknown)
                         Used for input hysteresis zone decisions

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

    # Get recommendations based on battery percentage (passing current states for hysteresis)
    recommendation_result = calculate_device_recommendations(battery_percentage, current_output_on, current_input_on)

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
        "hysteresis_thresholds": {
            "input_low": INPUT_LOW_THRESHOLD,
            "input_high": INPUT_HIGH_THRESHOLD,
            "output_low": OUTPUT_LOW_THRESHOLD,
            "output_high": OUTPUT_HIGH_THRESHOLD,
        },
    }
