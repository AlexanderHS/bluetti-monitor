<objective>
Create a Python script on the production server that queries InfluxDB for battery readings and other metrics. This script will be used by future agents and humans to quickly inspect recent data, identify patterns, and diagnose issues like false OCR readings.
</objective>

<context>
Server: ssh ahs@blu
Script location: /home/ahs/bluetti-monitor/query_metrics.py
InfluxDB: http://localhost:8086 (or use INFLUXDB_URL from .env)
- Organization: home
- Bucket: bluetti
- Token: Read from INFLUXDB_TOKEN in /home/ahs/bluetti-monitor/.env

Measurements in InfluxDB:
- `battery_reading`: fields `battery_percentage` (int), `ocr_confidence` (float), tag `ocr_strategy`
- `camera_status`: field `reachable` (0 or 1)
- `switchbot_status`: fields `reachable`, `rate_limited` (0 or 1)
- `device_state`: field `is_on` (0 or 1), tags `device_name`, `device_type`
</context>

<requirements>
1. Create a Python script `/home/ahs/bluetti-monitor/query_metrics.py` that:
   - Loads InfluxDB credentials from .env file (same directory)
   - Provides CLI commands for common queries
   - Outputs clean, readable results

2. Required CLI commands:
   ```bash
   # Get last N battery readings (default 10)
   python query_metrics.py readings [--count N]

   # Get readings from last N hours (default 1)
   python query_metrics.py recent [--hours N]

   # Get summary statistics for a time range
   python query_metrics.py stats [--hours N]

   # Check for anomalies (sudden jumps, impossible values)
   python query_metrics.py anomalies [--hours N]

   # Get camera/switchbot status history
   python query_metrics.py status [--hours N]
   ```

3. Output format should be human-readable but also parseable:
   - Include timestamps in local time
   - Show battery %, confidence, and strategy for readings
   - For anomalies, highlight suspicious values

4. The script should:
   - Handle missing/empty data gracefully
   - Work when run from any directory (use absolute paths for .env)
   - Include helpful --help output
</requirements>

<implementation>
Use the influxdb-client library (already installed in requirements.txt).
Use argparse for CLI argument parsing.
Use python-dotenv to load .env file.

For anomaly detection, consider:
- Battery jumps > 5% between consecutive readings
- Values outside 0-100 range
- Single-digit readings that break a pattern (like 7 instead of 70)
- Low confidence readings

Make the script executable and include a shebang line.
</implementation>

<output>
Create on server via SSH:
- `/home/ahs/bluetti-monitor/query_metrics.py` - The query script

Test the script works by running a few commands and showing output.
</output>

<verification>
Before declaring complete:
1. Script exists and is executable: `ls -la /home/ahs/bluetti-monitor/query_metrics.py`
2. Help works: `python query_metrics.py --help`
3. Readings command works: `python query_metrics.py readings --count 5`
4. Show sample output from at least one command
</verification>

<success_criteria>
- Script created on production server
- All CLI commands work without errors
- Output is readable and includes relevant data
- Can be used by future agents to quickly check metrics
</success_criteria>
