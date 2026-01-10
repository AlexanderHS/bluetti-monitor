#!/usr/bin/env python3
"""
InfluxDB Metrics Query Tool for Bluetti Monitor

Query battery readings, status history, and anomalies from InfluxDB.
Designed for quick diagnostics by humans and agents.

Usage:
    python query_metrics.py readings [--count N]      # Last N readings
    python query_metrics.py recent [--hours N]        # Readings from last N hours
    python query_metrics.py stats [--hours N]         # Statistics for time range
    python query_metrics.py anomalies [--hours N]     # Detect suspicious values
    python query_metrics.py status [--hours N]        # Camera/SwitchBot status
    python query_metrics.py devices [--hours N]       # Device state changes
"""

import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

# Load .env from script's directory
script_dir = Path(__file__).parent.absolute()
env_path = script_dir / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(env_path)
except ImportError:
    print("Warning: python-dotenv not installed, using environment variables only")

try:
    from influxdb_client import InfluxDBClient
except ImportError:
    print("Error: influxdb-client not installed. Run: pip install influxdb-client")
    sys.exit(1)


class MetricsQuery:
    """Query InfluxDB metrics for Bluetti Monitor"""

    def __init__(self):
        self.url = os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = os.getenv("INFLUXDB_TOKEN", "")
        self.org = os.getenv("INFLUXDB_ORG", "home")
        self.bucket = os.getenv("INFLUXDB_BUCKET", "bluetti")

        if not self.token:
            print("Error: INFLUXDB_TOKEN not set in .env file")
            print(f"Looked for .env at: {env_path}")
            sys.exit(1)

        self.client = InfluxDBClient(
            url=self.url,
            token=self.token,
            org=self.org
        )
        self.query_api = self.client.query_api()

    def _format_time(self, time_val) -> str:
        """Format timestamp to local time string"""
        if time_val is None:
            return "N/A"
        if isinstance(time_val, datetime):
            # Convert to local time
            local_time = time_val.astimezone()
            return local_time.strftime("%Y-%m-%d %H:%M:%S")
        return str(time_val)

    def _execute_query(self, flux_query: str) -> List[Dict[str, Any]]:
        """Execute a Flux query and return results as list of dicts"""
        try:
            tables = self.query_api.query(flux_query)
            results = []
            for table in tables:
                for record in table.records:
                    results.append(record.values)
            return results
        except Exception as e:
            print(f"Query error: {e}")
            return []

    def get_readings(self, count: int = 10) -> None:
        """Get the last N battery readings"""
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -7d)
            |> filter(fn: (r) => r._measurement == "battery_reading")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"], desc: true)
            |> limit(n: {count})
        '''

        results = self._execute_query(query)

        if not results:
            print("No battery readings found in the last 7 days")
            return

        print(f"\n{'='*70}")
        print(f"Last {count} Battery Readings")
        print(f"{'='*70}")
        print(f"{'Timestamp':<22} {'Battery %':<12} {'Confidence':<12} {'Strategy':<12}")
        print(f"{'-'*70}")

        for r in results:
            time_str = self._format_time(r.get("_time"))
            battery = r.get("battery_percentage", "N/A")
            confidence = r.get("ocr_confidence", 0)
            strategy = r.get("ocr_strategy", "unknown")

            conf_str = f"{confidence:.2f}" if isinstance(confidence, float) else str(confidence)
            print(f"{time_str:<22} {str(battery):<12} {conf_str:<12} {strategy:<12}")

        print()

    def get_recent(self, hours: int = 1) -> None:
        """Get readings from the last N hours"""
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r._measurement == "battery_reading")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"], desc: false)
        '''

        results = self._execute_query(query)

        if not results:
            print(f"No battery readings found in the last {hours} hour(s)")
            return

        print(f"\n{'='*70}")
        print(f"Battery Readings - Last {hours} Hour(s) ({len(results)} readings)")
        print(f"{'='*70}")
        print(f"{'Timestamp':<22} {'Battery %':<12} {'Confidence':<12} {'Strategy':<12}")
        print(f"{'-'*70}")

        for r in results:
            time_str = self._format_time(r.get("_time"))
            battery = r.get("battery_percentage", "N/A")
            confidence = r.get("ocr_confidence", 0)
            strategy = r.get("ocr_strategy", "unknown")

            conf_str = f"{confidence:.2f}" if isinstance(confidence, float) else str(confidence)
            print(f"{time_str:<22} {str(battery):<12} {conf_str:<12} {strategy:<12}")

        print()

    def get_stats(self, hours: int = 24) -> None:
        """Get summary statistics for a time range"""
        # Get battery stats
        battery_query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r._measurement == "battery_reading")
            |> filter(fn: (r) => r._field == "battery_percentage")
        '''

        confidence_query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r._measurement == "battery_reading")
            |> filter(fn: (r) => r._field == "ocr_confidence")
        '''

        battery_results = self._execute_query(battery_query)
        confidence_results = self._execute_query(confidence_query)

        print(f"\n{'='*70}")
        print(f"Statistics - Last {hours} Hour(s)")
        print(f"{'='*70}")

        if battery_results:
            values = [r.get("_value") for r in battery_results if r.get("_value") is not None]
            if values:
                print(f"\nBattery Percentage:")
                print(f"  Count:   {len(values)}")
                print(f"  Min:     {min(values)}%")
                print(f"  Max:     {max(values)}%")
                print(f"  Average: {sum(values)/len(values):.1f}%")
                print(f"  Current: {values[-1]}%")
        else:
            print("\nNo battery readings found")

        if confidence_results:
            values = [r.get("_value") for r in confidence_results if r.get("_value") is not None]
            if values:
                print(f"\nOCR Confidence:")
                print(f"  Min:     {min(values):.2f}")
                print(f"  Max:     {max(values):.2f}")
                print(f"  Average: {sum(values)/len(values):.2f}")

        # Count by strategy
        strategy_query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r._measurement == "battery_reading")
            |> filter(fn: (r) => r._field == "battery_percentage")
            |> group(columns: ["ocr_strategy"])
            |> count()
        '''

        strategy_results = self._execute_query(strategy_query)
        if strategy_results:
            print(f"\nReadings by Strategy:")
            for r in strategy_results:
                strategy = r.get("ocr_strategy", "unknown")
                count = r.get("_value", 0)
                print(f"  {strategy}: {count}")

        print()

    def get_anomalies(self, hours: int = 24) -> None:
        """Detect anomalies in battery readings"""
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r._measurement == "battery_reading")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"], desc: false)
        '''

        results = self._execute_query(query)

        print(f"\n{'='*70}")
        print(f"Anomaly Detection - Last {hours} Hour(s)")
        print(f"{'='*70}")

        if not results:
            print("No battery readings found")
            return

        anomalies = []

        # Check for various anomaly types
        prev_battery = None
        prev_time = None

        for i, r in enumerate(results):
            battery = r.get("battery_percentage")
            confidence = r.get("ocr_confidence", 0)
            time_val = r.get("_time")
            strategy = r.get("ocr_strategy", "unknown")

            issues = []

            # Check for out-of-range values
            if battery is not None:
                if battery < 0 or battery > 100:
                    issues.append(f"OUT OF RANGE: {battery}%")

                # Check for single-digit anomalies (common OCR error: 7 instead of 70)
                if battery < 10 and prev_battery is not None and prev_battery >= 10:
                    issues.append(f"SUSPICIOUS DROP: {prev_battery}% -> {battery}% (possible OCR error)")

                # Check for large jumps (> 5% change in one reading)
                if prev_battery is not None:
                    delta = abs(battery - prev_battery)
                    if delta > 5:
                        direction = "UP" if battery > prev_battery else "DOWN"
                        issues.append(f"LARGE JUMP {direction}: {prev_battery}% -> {battery}% (delta: {delta}%)")

            # Check for low confidence readings
            if confidence is not None and confidence < 0.5:
                issues.append(f"LOW CONFIDENCE: {confidence:.2f}")

            if issues:
                anomalies.append({
                    "time": time_val,
                    "battery": battery,
                    "confidence": confidence,
                    "strategy": strategy,
                    "issues": issues
                })

            prev_battery = battery
            prev_time = time_val

        if not anomalies:
            print("\nNo anomalies detected")
            print(f"Analyzed {len(results)} readings")
        else:
            print(f"\nFound {len(anomalies)} anomalies in {len(results)} readings:\n")
            print(f"{'Timestamp':<22} {'Battery %':<12} {'Confidence':<12} {'Issues'}")
            print(f"{'-'*80}")

            for a in anomalies:
                time_str = self._format_time(a["time"])
                battery = a["battery"]
                confidence = a["confidence"]
                conf_str = f"{confidence:.2f}" if isinstance(confidence, float) else str(confidence)

                for j, issue in enumerate(a["issues"]):
                    if j == 0:
                        print(f"{time_str:<22} {str(battery):<12} {conf_str:<12} *** {issue}")
                    else:
                        print(f"{'':<22} {'':<12} {'':<12} *** {issue}")

        print()

    def get_status(self, hours: int = 4) -> None:
        """Get camera and SwitchBot status history"""
        print(f"\n{'='*70}")
        print(f"Status History - Last {hours} Hour(s)")
        print(f"{'='*70}")

        # Camera status
        camera_query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r._measurement == "camera_status")
            |> filter(fn: (r) => r._field == "reachable")
            |> sort(columns: ["_time"], desc: false)
        '''

        camera_results = self._execute_query(camera_query)

        print(f"\nCamera Status:")
        if camera_results:
            # Count reachable vs unreachable
            reachable = sum(1 for r in camera_results if r.get("_value") == 1)
            unreachable = len(camera_results) - reachable

            print(f"  Total checks: {len(camera_results)}")
            print(f"  Reachable:    {reachable} ({100*reachable/len(camera_results):.1f}%)")
            print(f"  Unreachable:  {unreachable} ({100*unreachable/len(camera_results):.1f}%)")

            # Show recent failures
            failures = [r for r in camera_results if r.get("_value") == 0]
            if failures:
                print(f"\n  Recent camera failures:")
                for f in failures[-5:]:
                    print(f"    {self._format_time(f.get('_time'))}")
        else:
            print("  No camera status data found")

        # SwitchBot status
        switchbot_query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r._measurement == "switchbot_status")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"], desc: false)
        '''

        switchbot_results = self._execute_query(switchbot_query)

        print(f"\nSwitchBot Status:")
        if switchbot_results:
            reachable = sum(1 for r in switchbot_results if r.get("reachable") == 1)
            rate_limited = sum(1 for r in switchbot_results if r.get("rate_limited") == 1)
            failed = sum(1 for r in switchbot_results if r.get("reachable") == 0 and r.get("rate_limited") == 0)

            print(f"  Total checks:  {len(switchbot_results)}")
            print(f"  Reachable:     {reachable} ({100*reachable/len(switchbot_results):.1f}%)")
            print(f"  Rate limited:  {rate_limited} ({100*rate_limited/len(switchbot_results):.1f}%)")
            print(f"  Failed:        {failed} ({100*failed/len(switchbot_results):.1f}%)")
        else:
            print("  No SwitchBot status data found")

        print()

    def get_devices(self, hours: int = 4) -> None:
        """Get device state changes"""
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: -{hours}h)
            |> filter(fn: (r) => r._measurement == "device_state")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"], desc: false)
        '''

        results = self._execute_query(query)

        print(f"\n{'='*70}")
        print(f"Device State Changes - Last {hours} Hour(s)")
        print(f"{'='*70}")

        if not results:
            print("\nNo device state changes found")
            return

        print(f"\n{'Timestamp':<22} {'Device':<15} {'Type':<10} {'State'}")
        print(f"{'-'*60}")

        for r in results:
            time_str = self._format_time(r.get("_time"))
            device = r.get("device_name", "unknown")
            device_type = r.get("device_type", "unknown")
            is_on = r.get("is_on", 0)
            state = "ON" if is_on == 1 else "OFF"

            print(f"{time_str:<22} {device:<15} {device_type:<10} {state}")

        print()

    def close(self):
        """Close the InfluxDB client"""
        if self.client:
            self.client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Query InfluxDB metrics for Bluetti Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_metrics.py readings --count 20    # Last 20 readings
  python query_metrics.py recent --hours 2       # Readings from last 2 hours
  python query_metrics.py stats --hours 12       # Statistics for last 12 hours
  python query_metrics.py anomalies --hours 6    # Anomalies in last 6 hours
  python query_metrics.py status --hours 4       # Camera/SwitchBot status
  python query_metrics.py devices --hours 2      # Device state changes
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # readings command
    readings_parser = subparsers.add_parser("readings", help="Get last N battery readings")
    readings_parser.add_argument("--count", "-n", type=int, default=10,
                                 help="Number of readings to show (default: 10)")

    # recent command
    recent_parser = subparsers.add_parser("recent", help="Get readings from last N hours")
    recent_parser.add_argument("--hours", "-H", type=int, default=1,
                               help="Number of hours to query (default: 1)")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Get summary statistics")
    stats_parser.add_argument("--hours", "-H", type=int, default=24,
                              help="Number of hours to analyze (default: 24)")

    # anomalies command
    anomalies_parser = subparsers.add_parser("anomalies", help="Detect anomalies in readings")
    anomalies_parser.add_argument("--hours", "-H", type=int, default=24,
                                  help="Number of hours to analyze (default: 24)")

    # status command
    status_parser = subparsers.add_parser("status", help="Get camera/SwitchBot status history")
    status_parser.add_argument("--hours", "-H", type=int, default=4,
                               help="Number of hours to query (default: 4)")

    # devices command
    devices_parser = subparsers.add_parser("devices", help="Get device state changes")
    devices_parser.add_argument("--hours", "-H", type=int, default=4,
                                help="Number of hours to query (default: 4)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    query = MetricsQuery()

    try:
        if args.command == "readings":
            query.get_readings(args.count)
        elif args.command == "recent":
            query.get_recent(args.hours)
        elif args.command == "stats":
            query.get_stats(args.hours)
        elif args.command == "anomalies":
            query.get_anomalies(args.hours)
        elif args.command == "status":
            query.get_status(args.hours)
        elif args.command == "devices":
            query.get_devices(args.hours)
    finally:
        query.close()


if __name__ == "__main__":
    main()
