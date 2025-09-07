"""
Shared recommendation logic for Bluetti power management

This module contains the core logic for determining device control recommendations
based on battery status. It's used by both the API endpoint and the worker.
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_device_recommendations(battery_percentage: int) -> Dict:
    """
    Calculate device control recommendations based on battery percentage

    Args:
        battery_percentage: Current battery percentage (0-100)

    Returns:
        Dictionary with device recommendations and reasoning
    """
    if battery_percentage < 20:
        # Below 20% - turn off outputs, turn on input (charge)
        recommendations = {
            "input": "turn_on",
            "output_1": "turn_off",
            "output_2": "turn_off",
        }
        reasoning = f"Battery at {battery_percentage}% - critical low, charging needed"

    elif 20 <= battery_percentage < 60:
        # Between 20-60% - turn off input, turn off outputs (conservation)
        recommendations = {
            "input": "turn_off",
            "output_1": "turn_off",
            "output_2": "turn_off",
        }
        reasoning = f"Battery at {battery_percentage}% - low, conserving power"

    elif 60 <= battery_percentage < 80:
        # Between 60-80% - turn off input, turn on one output (moderate drain)
        recommendations = {
            "input": "turn_off",
            "output_1": "turn_on",
            "output_2": "turn_off",
        }
        reasoning = f"Battery at {battery_percentage}% - moderate level, using one output for moderate drain"

    else:  # battery_percentage >= 80
        # Above 80% - turn off input, turn on both outputs (rapid drain)
        recommendations = {
            "input": "turn_off",
            "output_1": "turn_off",
            "output_2": "turn_on",
        }
        reasoning = f"Battery at {battery_percentage}% - high level, using both outputs for rapid drain"

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
        return {
            "success": True,
            "status": "blind",
            "message": "No recent readings in the last 30 minutes",
            "recommendations": {
                "input": "turn_off",
                "output_1": "turn_off",
                "output_2": "turn_off",
            },
            "reasoning": "No recent battery data available - turning off all outputs and input for safety",
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
