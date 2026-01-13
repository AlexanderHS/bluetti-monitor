<objective>
Fix the container suicide logic in the SwitchBot controller to only trigger on REAL network or API failures, not during intentional idle/rate-limited periods.

The current bug: During nighttime/idle mode, the system intentionally reduces SwitchBot tap frequency (e.g., 30-minute intervals). However, the suicide logic incorrectly interprets this intentional rate-limiting as a SwitchBot failure, triggering container restart when there's no actual problem.
</objective>

<context>
This is a Python monitoring agent that uses SwitchBot to tap a Bluetti solar generator's LCD screen to keep it awake for OCR reading.

Key behavior modes:
- **Active mode**: Frequent screen taps (default 300s intervals)
- **Idle mode**: Infrequent taps (default 1800s/30min intervals) during nighttime when all devices are OFF

The suicide logic exists to restart the container when the SwitchBot API is genuinely unreachable (network issues, API failures) so Docker can attempt recovery. It should NOT trigger when taps are intentionally skipped due to idle mode rate limiting.

Read CLAUDE.md for project conventions and deployment workflow.

Examine these files to understand the current implementation:
- switchbot_controller.py - Contains the suicide logic and tap tracking
- main.py - Contains idle mode switching logic ("Switched to IDLE monitoring")
</context>

<evidence>
From the logs, the bug sequence is:
1. `💀 CONTAINER SUICIDE: No successful SwitchBot tap for 0.3 hours (limit: 0.25 hours)` - Suicide triggered
2. Container restarts, then immediately: `💤 Switched to IDLE monitoring (30 min taps): nighttime, all devices OFF`
3. Shortly after: `📊 51% (conf: 1.00, template)` - OCR working fine, system is healthy

The suicide triggered because 0.3 hours passed without a successful tap, but the system was in idle mode where 30-minute gaps are EXPECTED behavior.
</evidence>

<requirements>
1. **Identify the suicide logic**: Find where the "no successful tap for X hours" check occurs

2. **Distinguish failure types**: The system needs to track:
   - Intentional skips (idle mode rate limiting) - Should NOT count toward suicide timer
   - Actual failures (network errors, API errors, timeouts) - SHOULD count toward suicide timer

3. **Fix the logic**: Modify the suicide check to only trigger when there have been genuine API/network failures, not when taps were intentionally skipped

4. **Consider edge cases**:
   - What if idle mode lasts longer than the suicide threshold? (This is normal, don't suicide)
   - What if there are real failures during idle mode? (Should still track those)
   - Transition from active to idle mode shouldn't reset failure tracking incorrectly
</requirements>

<implementation_approach>
Use a state machine approach similar to monitoring tools (Grafana, Prometheus alerting):

**States:**
```
OKAY ──(real failure)──► PENDING ──(timeout)──► SUICIDE
  ▲                          │
  └───────(success)──────────┘
```

**Implementation (simplified with timestamp):**
- Track `switchbot_failure_since: Optional[datetime]`
- If `switchbot_failure_since` is None → SwitchBot is healthy
- If `switchbot_failure_since` is set → SwitchBot is in failure state
- Suicide only when: `switchbot_failure_since is not None and (now - switchbot_failure_since) > timeout`

**State transitions:**
- **Healthy → Failure**: On a REAL failure (network error, API error, timeout, rate limit error) - set `switchbot_failure_since = now`
- **Failure → Healthy**: On a successful tap - reset `switchbot_failure_since = None`
- **Failure → SUICIDE**: When `switchbot_failure_since` exceeds timeout threshold

**Critical: What does NOT change state:**
- Intentional skips (idle mode rate limiting, cooldown periods)
- These are neutral - they don't prove success OR failure

**Timeout value:**
- Must be longer than idle mode interval (30 minutes)
- Suggest 45-60 minutes minimum, or make configurable via env var: `SWITCHBOT_FAILURE_TIMEOUT_HOURS` (default: 1.0)

**Startup:**
- Start in OKAY state (`pending_since = None`) - innocent until proven guilty

Check recent git commits for any related changes that might inform the fix.
</implementation_approach>

<output>
Modify the existing files (do NOT create new files unless absolutely necessary):
- `./switchbot_controller.py` - Fix the suicide logic
- `./main.py` - If changes needed to pass idle state context

After fixing, verify the logic by tracing through these scenarios mentally:
1. Idle mode for 2 hours with no taps (all skips) → State stays OKAY, no suicide
2. Network error, then 1+ hour passes with no recovery → PENDING times out, suicide
3. Network error, then successful tap 10 min later → PENDING → OKAY, no suicide
4. Successful tap, then enter idle mode for 2 hours → State stays OKAY, no suicide
5. Network error during idle mode, then successful tap 35 min later → PENDING → OKAY, no suicide
</output>

<verification>
Before declaring complete:
1. Run `python -m py_compile switchbot_controller.py main.py` to verify syntax
2. Trace through the code logic to confirm:
   - Intentional idle skips do NOT trigger suicide
   - Real network failures DO trigger suicide after threshold
3. Check that logging clearly distinguishes between "skipped (idle)" and "failed (error)"
</verification>

<success_criteria>
- Container no longer suicides during normal idle mode operation
- Container still suicides on genuine prolonged SwitchBot API failures
- Logging makes it clear why a tap was skipped vs why it failed
- No regression to active mode behavior
</success_criteria>
