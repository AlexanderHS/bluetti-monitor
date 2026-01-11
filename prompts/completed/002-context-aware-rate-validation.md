<objective>
Implement context-aware rate-of-change validation with two simple levels based on output state.

The current static 150%/h limit catches extreme errors but misses OCR digit-drops over longer intervals (51→5 over 30 min = 92%/h passed validation).
</objective>

<context>
Local repo: /Users/alexhamiltonsmith/repos/bluetti-monitor
Query script: ./query_metrics.sh

**The problem:**
Our static rate validation (150%/h limit) caught extreme cases like 71→7 in 5 minutes, but missed 51→5 over 30 minutes because 92%/h is under the limit.

**Recent false reading that passed:**
```
2026-01-11 21:13:17    51%    template
2026-01-11 21:43:27    5%     template  ← PASSED (92%/h < 150%/h limit)
```

**The simple fix - two rate limits based on output state:**

| Output State | Max Rate | Reasoning |
|--------------|----------|-----------|
| Output ON | 150%/h | 2000W discharge = 100%/h theoretical, with 1.5x margin |
| Output OFF | 55%/h | Solar 700W = 35%/h, AC input 500W = 25%/h, with margin |

When output is OFF, the only way battery % can change significantly is:
- Solar charging: 700W / 2000Wh = 35%/h max
- AC input charging: 500W / 2000Wh = 25%/h max
- Standby drain: ~1-3%/h

A 55%/h limit when output is OFF covers all legitimate scenarios with margin.

**The 51→5 false reading would now be rejected:**
- Rate: 46% / 0.5h = 92%/h
- Output was OFF at 21:43 (nighttime)
- Limit: 55%/h
- 92 > 55 → **REJECTED**

**Existing context already available in worker.py:**
```python
device_states = get_device_states()
output_on = device_discovery.is_any_output_on(device_states)
```

Key files:
- worker.py - validate_rate_of_change() at line ~219, main loop has device state
- .env.example - rate config at lines 75-83
</context>

<requirements>
**Phase 1: Update validate_rate_of_change()**

Modify the function in worker.py to accept output state:

```python
def validate_rate_of_change(
    new_percentage: int,
    last_percentage: int,
    time_diff_seconds: float,
    output_on: bool = False
) -> Tuple[bool, str]:
    """
    Context-aware validation based on output state.

    Two rate limits:
    - Output ON: 150%/h (2000W discharge possible)
    - Output OFF: 55%/h (solar/AC charging only, ~35%/h max + margin)
    """
    time_diff_hours = time_diff_seconds / 3600
    percentage_change = new_percentage - last_percentage
    abs_change = abs(percentage_change)

    # For very short intervals, just check absolute change
    if time_diff_hours < MIN_TIME_FOR_RATE_CHECK_HOURS:
        if abs_change > MAX_INSTANT_CHANGE:
            return False, f"instant change too large: {last_percentage}% -> {new_percentage}% ({abs_change}% in {time_diff_seconds:.0f}s)"
        return True, ""

    # Calculate rate of change in %/hour
    rate_per_hour = abs_change / time_diff_hours

    # Simple two-level rate limit based on output state
    if output_on:
        max_rate = 150.0  # 2000W discharge = 100%/h, with 1.5x margin
    else:
        max_rate = 55.0   # Solar 35%/h + AC 25%/h max, with margin

    if rate_per_hour > max_rate:
        context = "output ON" if output_on else "output OFF"
        return False, f"implausible rate ({context}): {last_percentage}% -> {new_percentage}% in {time_diff_seconds/60:.1f}min ({rate_per_hour:.0f}%/h > {max_rate:.0f}%/h limit)"

    return True, ""
```

**Phase 2: Update the call site**

In background_worker(), update the validation call to pass output state:

```python
# Rate-of-change validation (context-aware based on output state)
rate_valid, rate_reason = validate_rate_of_change(
    primary_percentage,
    last_reading["battery_percentage"],
    time_diff_seconds,
    output_on=output_on  # Already available in scope
)
```

**Phase 3: Update configuration**

Update .env.example comments to document the new behavior:
```
# Rate-of-Change Validation (context-aware)
# Two rate limits based on output state:
# - Output ON: 150%/h (2000W discharge = 100%/h + margin)
# - Output OFF: 55%/h (solar 35%/h or AC 25%/h + margin)
# This catches OCR errors like 51->5 (92%/h) when output is off
```

**Phase 4: Clean up old config**

Remove or comment out the old static config variables that are no longer used:
- MAX_DRAIN_RATE_PER_HOUR (now hardcoded based on output state)
- MAX_CHARGE_RATE_PER_HOUR (no longer separate)
- Keep RATE_VALIDATION_MULTIPLIER only if still useful, otherwise remove
</requirements>

<implementation>
1. Read worker.py to find validate_rate_of_change() and its call site
2. Update the function signature to accept output_on parameter
3. Replace the drain/charge logic with simple two-level check
4. Update the call site in background_worker() to pass output_on
5. Update .env.example documentation
6. Remove unused config variables
7. Syntax check: `python -m py_compile worker.py`
</implementation>

<output>
Modified files:
- worker.py - Simplified context-aware rate validation
- .env.example - Updated documentation, removed unused config

After changes:
1. Commit and push
2. Deploy: `ssh ahs@blu "cd /home/ahs/bluetti-monitor && git pull && docker compose up -d --build"`
3. Verify logs show output context in rate validation
</output>

<verification>
Before declaring complete:
1. Code compiles: `python -m py_compile worker.py`
2. Test cases:
   | Scenario | Rate | Limit | Result |
   |----------|------|-------|--------|
   | 51→5, 30min, output OFF | 92%/h | 55%/h | REJECT |
   | 51→45, 30min, output ON | 12%/h | 150%/h | PASS |
   | 51→45, 30min, output OFF | 12%/h | 55%/h | PASS |
   | 50→80, 30min, output OFF (charging) | 60%/h | 55%/h | REJECT (borderline - may need 60%/h limit?) |
   | 50→75, 30min, output OFF (charging) | 50%/h | 55%/h | PASS |
3. Committed and pushed
4. Deployed to production
5. Logs: `ssh ahs@blu "cd /home/ahs/bluetti-monitor && docker compose logs --tail=50 bluetti-monitor-worker"`
</verification>

<success_criteria>
- Two-level rate limit: 150%/h (output ON) vs 55%/h (output OFF)
- The 51→5 false reading (output OFF) is rejected
- Active discharge allows high rates
- Charging scenarios (solar/AC) pass within the 55%/h limit
- Simple, maintainable implementation
</success_criteria>
