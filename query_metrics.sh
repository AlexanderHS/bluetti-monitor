#!/bin/bash
#
# Query InfluxDB metrics for Bluetti Monitor
#
# This wrapper abstracts away the SSH and docker exec complexity.
# Runs query_metrics.py inside the Docker container on the production server.
#
# Usage:
#   ./query_metrics.sh readings [--count N]      # Last N readings
#   ./query_metrics.sh recent [--hours N]        # Readings from last N hours
#   ./query_metrics.sh stats [--hours N]         # Statistics for time range
#   ./query_metrics.sh anomalies [--hours N]     # Detect suspicious values
#   ./query_metrics.sh status [--hours N]        # Camera/SwitchBot status
#   ./query_metrics.sh devices [--hours N]       # Device state changes
#
# Examples:
#   ./query_metrics.sh readings --count 20
#   ./query_metrics.sh anomalies --hours 6
#   ./query_metrics.sh stats --hours 12
#

set -e

# Server configuration
SERVER="ahs@blu"
CONTAINER="bluetti-monitor-bluetti-monitor-worker-1"
SCRIPT="query_metrics.py"

# Show help if no arguments
if [ $# -eq 0 ]; then
    echo "Bluetti Monitor Metrics Query Tool"
    echo ""
    echo "Usage: ./query_metrics.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  readings    Get last N battery readings (--count N, default 10)"
    echo "  recent      Get readings from last N hours (--hours N, default 1)"
    echo "  stats       Get summary statistics (--hours N, default 24)"
    echo "  anomalies   Detect anomalies in readings (--hours N, default 24)"
    echo "  status      Get camera/SwitchBot status (--hours N, default 4)"
    echo "  devices     Get device state changes (--hours N, default 4)"
    echo ""
    echo "Examples:"
    echo "  ./query_metrics.sh readings --count 20"
    echo "  ./query_metrics.sh anomalies --hours 6"
    echo "  ./query_metrics.sh stats --hours 12"
    echo ""
    exit 0
fi

# Pass all arguments to the Python script inside the container
ssh "$SERVER" "docker exec $CONTAINER python $SCRIPT $*"
