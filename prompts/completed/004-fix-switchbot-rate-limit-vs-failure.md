<objective>
Fix bug where SwitchBot rate limiting is incorrectly treated as a failure, triggering false CRITICAL errors and unnecessary safe shutdown mode.
</objective>

<context>
The bluetti-monitor system uses SwitchBot to tap the Bluetti screen to keep it awake. The system has:
- **ACTIVE monitoring mode**: Taps every 5 minutes when outputs are ON
- **IDLE monitoring mode**: Taps every 30 minutes during nighttime or when all devices are OFF

The problem: When the system tries to tap but the rate limit says "Next tap allowed in X minutes", this is logged as a FAILURE and counts toward the 5-failure threshold that triggers safe shutdown. But rate limiting is **expected behavior** - it means the tap interval policy is working correctly.

From the logs, you can see the problematic pattern:
```
19:03:01 - ERROR - SwitchBot tap failed (1/5): Rate limited - Next tap allowed in 24.9 minutes
19:03:11 - ERROR - SwitchBot tap failed (2/5): Rate limited - Next tap allowed in 24.8 minutes
...
19:05:04 - ERROR - SwitchBot tap failed (5/5): Rate limited - Next tap allowed in 22.9 minutes
19:05:14 - CRITICAL - SwitchBot failure threshold reached - 5 consecutive failures
19:05:14 - CRITICAL - Cannot reliably tap screen - entering SAFE SHUTDOWN mode
```

The safe shutdown mode is designed to protect against actual API/network failures where we can't wake the screen. Rate limiting is NOT such a failure - it means the screen was tapped recently and doesn't need another tap yet.
</context>

<research>
First, examine the relevant code to understand the current implementation:
1. Find where SwitchBot tapping is implemented
2. Find where rate limiting is checked/returned
3. Find where the failure counter is incremented
4. Find where SAFE SHUTDOWN mode is triggered
5. Understand how ACTIVE vs IDLE tap intervals are determined

Key files to examine: worker.py, main.py, and any SwitchBot-related modules.
</research>

<requirements>
1. **Distinguish rate limiting from actual failures**: Rate limiting should NOT increment the failure counter or trigger safe shutdown
2. **Only count actual failures**: Network errors, API errors, authentication failures, timeouts - these are real failures
3. **Graceful rate limit handling**: When rate limited, log at INFO or DEBUG level (not ERROR) and skip the tap attempt without counting it as a failure
4. **Preserve safe shutdown for real failures**: The 5-failure threshold and safe shutdown logic should still work for genuine connectivity/API issues
</requirements>

<implementation>
Likely approaches (choose based on code structure):

**Option A - Pre-check approach**: Before attempting a tap, check if we're within the rate limit window. If so, skip silently (or log at INFO level) without calling the API at all.

**Option B - Return type differentiation**: Have the tap function return different values for "success", "rate_limited", and "failed". Only increment failure counter on "failed".

**Option C - Exception handling**: Use a specific exception type for rate limiting that doesn't count as a failure.

The fix should:
- NOT cause spurious ERROR/CRITICAL logs when rate limiting is working as expected
- NOT trigger safe shutdown when rate limiting is the only "issue"
- Preserve the protective behavior when actual network/API failures occur
- Be minimal - don't refactor more than necessary to fix the bug
</implementation>

<constraints>
- Do not change the tap interval settings (ACTIVE_TAP_INTERVAL, IDLE_TAP_INTERVAL)
- Do not remove the safe shutdown feature - it's important for real failures
- Keep the failure counter mechanism for genuine API/network failures
- Log rate limit status at INFO level rather than ERROR
</constraints>

<output>
Modify the relevant files to fix this bug. Expected changes likely in:
- `./worker.py` or wherever SwitchBot tap logic lives
- Possibly the SwitchBot client module

After fixing, add a brief comment explaining the rate-limit vs failure distinction where appropriate.
</output>

<verification>
After implementing:
1. Run syntax check: `python -m py_compile worker.py` (and any other modified files)
2. Verify the logic by tracing through the code mentally - rate limited responses should NOT increment failure counters
3. Check that actual failures (network error, API error) WOULD still increment the counter

The fix is successful when:
- Rate limiting logs at INFO level, not ERROR
- Rate limiting does NOT increment the consecutive failure counter
- Rate limiting does NOT trigger safe shutdown
- Actual API/network failures STILL trigger the existing protective behavior
</verification>
