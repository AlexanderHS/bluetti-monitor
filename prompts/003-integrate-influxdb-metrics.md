<objective>
Update the bluetti-monitor codebase to optionally write metrics to InfluxDB when configured.

This enables time-series tracking of battery percentage, camera connectivity, SwitchBot connectivity, OCR confidence, and device states for visualization in Grafana and alerting on failures.
</objective>

<context>
Local repo: /Users/alexhamiltonsmith/repos/bluetti-monitor
Production server: ssh ahs@blu at /home/ahs/bluetti-monitor

InfluxDB is running at: http://localhost:8086 (on the server)
- Organization: home
- Bucket: bluetti
- Token: (configured via env var)

Key files to modify:
- .env.example - Add InfluxDB configuration variables
- worker.py - Main monitoring loop, write metrics here
- switchbot_controller.py - SwitchBot API interactions, track failures here
- requirements.txt - Add influxdb-client dependency
</context>

<requirements>
1. Update .env.example with new InfluxDB settings:
   - INFLUXDB_URL (default: http://localhost:8086)
   - INFLUXDB_TOKEN (required for metrics, empty = disabled)
   - INFLUXDB_ORG (default: home)
   - INFLUXDB_BUCKET (default: bluetti)

2. Add influxdb-client to requirements.txt

3. Create a simple InfluxDB writer module or add to existing code:
   - Initialize client only if INFLUXDB_TOKEN is set
   - Graceful handling if InfluxDB is unreachable (log warning, don't crash)
   - Non-blocking writes (don't slow down main loop)

4. Write these metrics to InfluxDB on each successful reading:
   - battery_percentage (int, 0-100)
   - ocr_confidence (float, 0-1)
   - ocr_strategy (tag: "template" or "llm")
   - camera_reachable (bool, 1 or 0)

5. Write these metrics on device control actions:
   - input_state (bool, 1=on, 0=off)
   - output_state (bool, 1=on, 0=off)
   - device_name (tag)

6. Track camera connectivity:
   - On successful capture: camera_reachable=1
   - On connection error: camera_reachable=0
   - This is a key metric for alerting on ESP32 failures

7. Track SwitchBot connectivity:
   - On successful SwitchBot API call: switchbot_reachable=1
   - On SwitchBot API failure (not rate limiting): switchbot_reachable=0
   - Note: Rate limiting (429) should NOT be treated as unreachable - the recent fix in switchbot_controller.py distinguishes these
   - Include switchbot_rate_limited (bool) as separate metric when 429 occurs
   - This enables alerting when SwitchBot device is offline (battery dead, network down, etc.)

8. After code changes:
   - Commit and push to git
   - Deploy to production (pull, rebuild, restart)
   - Verify logs show InfluxDB writes (if configured) or graceful skip (if not)
</requirements>

<implementation>
Read worker.py and switchbot_controller.py to understand the current structure before making changes.

The InfluxDB integration should be:
- Optional: Only active when INFLUXDB_TOKEN is set
- Resilient: Connection failures log a warning but don't stop monitoring
- Efficient: Use batching or async writes if possible
- Simple: Don't over-engineer, just write the key metrics

Use the influxdb-client-python library Point class for building measurements.

For SwitchBot tracking:
- The existing code already distinguishes rate limiting (429) from real failures
- Hook into the existing error handling to emit the appropriate metric
- switchbot_reachable=0 should only be set for actual failures (network errors, 5xx, etc.)
- switchbot_rate_limited=1 when rate limited (temporary, not an alert condition)
</implementation>

<output>
Modified files:
- .env.example (add InfluxDB vars)
- requirements.txt (add influxdb-client)
- worker.py (add metrics writing)
- switchbot_controller.py (add SwitchBot connectivity metrics)

Deployment:
- Commit and push to git
- Deploy to production server
</output>

<verification>
Before declaring complete:
1. Code compiles: python -m py_compile worker.py switchbot_controller.py
2. Changes committed and pushed
3. Deployed to production
4. Check logs: docker compose logs --tail=20 bluetti-monitor-worker
   - Should see normal operation (InfluxDB writes if configured, or graceful skip if not)
</verification>

<success_criteria>
- InfluxDB integration is optional (disabled by default)
- When enabled, writes battery_percentage, ocr_confidence, camera_reachable metrics
- When enabled, writes switchbot_reachable and switchbot_rate_limited metrics
- Camera connectivity failures are tracked (key for alerting on ESP32 issues)
- SwitchBot connectivity failures are tracked (key for alerting on device/network issues)
- Rate limiting is tracked separately and not confused with actual failures
- Production deployment successful
- No crashes if InfluxDB is unreachable
</success_criteria>
