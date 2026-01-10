<objective>
Implement rate-of-change validation for battery readings to reject implausible OCR errors like "71" being misread as "7".

The validation should be time-aware: a 60% drop over 2 days is fine, but a 60% drop in 5 minutes is impossible and should be rejected.
</objective>

<context>
Local repo: /Users/alexhamiltonsmith/repos/bluetti-monitor
Query script: ./query_metrics.sh

The problem: OCR sometimes drops digits (71 → 7), and these readings pass through because:
- Template OCR always returns confidence = 1.0
- Plausibility checks only trigger when confidence < 0.9

We need rate-of-change validation that applies regardless of confidence.

Key files:
- worker.py - Main loop where readings are stored (validation should go here)
- template_classifier.py - OCR implementation (returns confidence 1.0)
</context>

<requirements>
**Phase 1: Analyze historical data**

Use the query script to determine realistic rate-of-change limits:
```bash
./query_metrics.sh readings --count 200
```

Calculate from the data:
- Maximum observed drain rate (% per hour) during discharge
- Maximum observed charge rate (% per hour) during charging
- Typical time between readings

**Phase 2: Implement validation**

Add rate-of-change validation in worker.py that:

1. Calculates the rate of change:
   ```python
   time_diff_hours = (current_time - last_reading_time).total_seconds() / 3600
   rate_of_change = abs(new_percentage - last_percentage) / time_diff_hours
   ```

2. Compares against configurable limits:
   ```python
   # From .env with sensible defaults based on analysis
   MAX_DRAIN_RATE = float(os.getenv("MAX_DRAIN_RATE_PER_HOUR", "15"))  # %/hour
   MAX_CHARGE_RATE = float(os.getenv("MAX_CHARGE_RATE_PER_HOUR", "20"))  # %/hour
   RATE_MULTIPLIER = float(os.getenv("RATE_VALIDATION_MULTIPLIER", "1.5"))
   ```

3. Rejects readings that exceed limits:
   ```python
   if is_draining and rate_of_change > MAX_DRAIN_RATE * RATE_MULTIPLIER:
       logger.warning(f"Rejected implausible drain: {last}% → {new}% in {time_diff:.1f}h ({rate_of_change:.1f}%/h > {MAX_DRAIN_RATE * RATE_MULTIPLIER:.1f}%/h limit)")
       return False  # Don't store this reading
   ```

4. Has a minimum time threshold to avoid division issues:
   ```python
   MIN_TIME_FOR_RATE_CHECK = 0.01  # 36 seconds - below this, use absolute change limit
   if time_diff_hours < MIN_TIME_FOR_RATE_CHECK:
       # For very short intervals, just check absolute change
       if abs(new_percentage - last_percentage) > 5:
           logger.warning(f"Rejected large instant change: {last}% → {new}%")
           return False
   ```

**Phase 3: Update configuration**

Add to .env.example:
```
# Rate-of-change validation (rejects implausible OCR errors)
MAX_DRAIN_RATE_PER_HOUR=15
MAX_CHARGE_RATE_PER_HOUR=20
RATE_VALIDATION_MULTIPLIER=1.5
```
</requirements>

<implementation>
1. First, run the query script and analyze the data to determine appropriate defaults
2. Read worker.py to find where readings are validated/stored
3. Add the rate-of-change validation logic
4. Update .env.example with new config options
5. Test by reviewing what WOULD have been rejected in recent data

The validation should:
- Log warnings for rejected readings (don't silently drop)
- Not affect device control when readings are rejected (maintain last known state)
- Be configurable via environment variables
- Apply regardless of OCR confidence score
</implementation>

<output>
Modified files:
- worker.py - Add rate-of-change validation
- .env.example - Add configuration options

Then deploy:
1. Commit and push
2. Deploy to server: `ssh ahs@blu "cd /home/ahs/bluetti-monitor && git pull && docker compose up -d --build"`
3. Verify logs show validation is active
</output>

<verification>
Before declaring complete:
1. Run analysis to determine rate limits from historical data
2. Code compiles: `python -m py_compile worker.py`
3. Changes committed and pushed
4. Deployed to production
5. Check logs show rate validation is working:
   ```bash
   ssh ahs@blu "cd /home/ahs/bluetti-monitor && docker compose logs --tail=50 bluetti-monitor-worker"
   ```
6. Demonstrate that the 71→7 false reading WOULD have been rejected with this logic
</verification>

<success_criteria>
- Rate limits based on actual historical data analysis
- Validation rejects implausible readings regardless of confidence
- Configuration is flexible via environment variables
- The 71→7 false reading scenario would be caught
- Normal readings continue to pass through
</success_criteria>
