<objective>
Fix bug where SwitchBot tap rate limiter doesn't sync with monitoring mode changes, causing ACTIVE mode to use IDLE tap intervals.
</objective>

<context>
The bluetti-monitor system has two monitoring modes that determine screen tap frequency:
- **ACTIVE mode** (5 min taps): When input OR output devices are ON (battery state changing rapidly)
- **IDLE mode** (30 min taps): Nighttime AND all devices OFF (battery state stable)

The problem from logs:
```
19:29:43 - 💤 Switched to IDLE monitoring (30 min taps): nighttime, all devices OFF
19:30:29 - 🔄 Startup calibration: Battery at 7% - input ON, output OFF
19:30:29 - → Sending: input - mini ON (force=True)
19:30:39 - 🔄 Switched to ACTIVE monitoring (5 min taps): input(s) ON
19:34:55 - Screen tap skipped (rate limited) - 25.0 min until next tap allowed  ← BUG!
```

The mode switched to ACTIVE (5 min taps) at 19:30:39, but 4 minutes later at 19:34:55, the rate limiter still says "25 min until next tap" - it's using the old IDLE interval (30 min) instead of the new ACTIVE interval (5 min).

This matters because when devices are ON, the battery is actively charging or discharging, so we need frequent screen taps to keep it awake for accurate OCR readings.
</context>

<research>
Examine the code to understand:
1. Where are ACTIVE_TAP_INTERVAL and IDLE_TAP_INTERVAL defined?
2. How does the SwitchBot controller track `last_tap_time` and enforce rate limits?
3. Where does monitoring mode switching happen (the 💤/🔄 log messages)?
4. How is the current tap interval communicated to/used by the SwitchBot rate limiter?

Key files: worker.py, switchbot_controller.py
</research>

<requirements>
1. **Immediate interval sync**: When monitoring mode changes from IDLE to ACTIVE, the rate limiter must immediately start using the ACTIVE interval (5 min)
2. **Allow immediate tap on mode change**: When switching to ACTIVE mode, if the last tap was within the IDLE interval but outside the ACTIVE interval, allow a tap immediately
3. **Both directions**: Handle both IDLE→ACTIVE and ACTIVE→IDLE transitions correctly
4. **Preserve rate limiting**: The rate limiter should still prevent too-frequent taps within the current mode's interval
</requirements>

<implementation>
The likely fix involves one of these approaches:

**Option A - Pass current interval to tap function**: The SwitchBot tap function should receive the current required interval as a parameter, rather than using a cached/stale value.

**Option B - Notify SwitchBot of mode changes**: When monitoring mode changes, call a method on the SwitchBot controller to update its interval.

**Option C - Reset rate limit on mode change**: When switching to ACTIVE mode, reset or adjust `last_tap_time` so the new interval is used.

Choose the simplest approach that fits the existing code structure.
</implementation>

<constraints>
- Do not change the ACTIVE_TAP_INTERVAL (5 min) or IDLE_TAP_INTERVAL (30 min) values
- Preserve backward compatibility - the rate limiter should still work correctly
- Keep the fix minimal - focus on syncing the interval, not refactoring the entire system
</constraints>

<output>
Modify the relevant files to fix this bug. Expected changes in:
- `./switchbot_controller.py` - rate limiting logic
- `./worker.py` - mode switching and tap interval passing

Add a brief comment explaining the mode-sync behavior where appropriate.
</output>

<verification>
After implementing:
1. Run syntax check: `python -m py_compile worker.py switchbot_controller.py`
2. Trace through the code mentally:
   - When mode switches IDLE→ACTIVE, does the rate limiter immediately allow taps if >5 min since last tap?
   - When mode switches ACTIVE→IDLE, does the rate limiter extend the interval to 30 min?
3. The log message should show the correct remaining time based on current mode

The fix is successful when:
- Mode switch to ACTIVE allows a tap if it's been >5 min (not >30 min) since last tap
- Rate limiting messages show times consistent with the current mode's interval
- Transitions in both directions work correctly
</verification>
