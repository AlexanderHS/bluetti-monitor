<objective>
Create a Python script that queries InfluxDB for battery readings and other metrics. This script will be used by future agents and humans to quickly inspect recent data, identify patterns, and diagnose issues like false OCR readings.

The script runs inside the Docker container (not on the host) to avoid polluting the host environment.
</objective>

<context>
Server: ssh ahs@blu
Local repo: /Users/alexhamiltonsmith/repos/bluetti-monitor
Script location in repo: ./query_metrics.py

The script runs via docker exec:
```bash
ssh ahs@blu "docker exec bluetti-monitor-bluetti-monitor-worker-1 python query_metrics.py readings --count 10"
```

InfluxDB connection (from inside container):
- URL: http://10.0.0.142:8086 (use INFLUXDB_URL from container's env)
- Organization: home
- Bucket: bluetti
- Token: From INFLUXDB_TOKEN env var (already configured in container)

Measurements in InfluxDB:
- `battery_reading`: fields `battery_percentage` (int), `ocr_confidence` (float), tag `ocr_strategy`
- `camera_status`: field `reachable` (0 or 1)
- `switchbot_status`: fields `reachable`, `rate_limited` (0 or 1)
- `device_state`: field `is_on` (0 or 1), tags `device_name`, `device_type`
</context>

<requirements>
1. Create a Python script `./query_metrics.py` in the local repo that:
   - Reads InfluxDB credentials from environment variables (already set in container)
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
Read credentials from environment variables (os.environ):
- INFLUXDB_URL
- INFLUXDB_TOKEN
- INFLUXDB_ORG
- INFLUXDB_BUCKET

For anomaly detection, consider:
- Battery jumps > 5% between consecutive readings
- Values outside 0-100 range
- Single-digit readings that break a pattern (like 7 instead of 70)
- Low confidence readings

Include a shebang line: #!/usr/bin/env python3
</implementation>

<output>
Create in local repo:
- `./query_metrics.py` - The Python script that queries InfluxDB (runs inside container)
- `./query_metrics.sh` - Local wrapper script that SSHs and runs docker exec

The wrapper script abstracts away all complexity. Future agents just run:
```bash
./query_metrics.sh readings --count 10
./query_metrics.sh anomalies --hours 24
```

Then deploy to production:
1. Commit and push to git
2. SSH to server, pull, and rebuild container:
   ```bash
   ssh ahs@blu "cd /home/ahs/bluetti-monitor && git pull && docker compose up -d --build"
   ```
3. Test locally using the wrapper script
</output>

<verification>
Before declaring complete:
1. Python script compiles: `python -m py_compile query_metrics.py`
2. Shell wrapper is executable: `chmod +x query_metrics.sh`
3. Changes committed and pushed to git
4. Container rebuilt: `ssh ahs@blu "cd /home/ahs/bluetti-monitor && git pull && docker compose up -d --build"`
5. Wrapper script works locally:
   ```bash
   ./query_metrics.sh --help
   ./query_metrics.sh readings --count 5
   ```
6. Show sample output from at least one command
</verification>

<success_criteria>
- Python script created in repo (query_metrics.py)
- Shell wrapper created in repo (query_metrics.sh)
- Deployed to production via git pull + docker compose rebuild
- Wrapper script works from local machine
- Output is readable and includes relevant data
- Future agents can simply run `./query_metrics.sh [command] [args]`
</success_criteria>
