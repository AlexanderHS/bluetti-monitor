<objective>
Fix the startup calibration routine to only touch devices that are safe to control.

Currently, the calibration routine turns everything off and gradually turns on only what's needed, but it touches devices it shouldn't (like desktop computers and servers). Modify the calibration logic to ONLY touch devices with "light", "input", or "output" in their names (case-insensitive matching).
</objective>

<context>
This is part of a Bluetti solar generator monitoring system that controls smart home devices based on battery levels. The calibration routine ensures a known starting state, but must not touch critical infrastructure like computers and servers.

Read CLAUDE.md first for project conventions, then examine the codebase to understand:
- Where the calibration routine is implemented
- How devices are discovered and controlled
- The current device filtering logic (if any)
</context>

<research>
Before implementing, find and understand:
1. The startup/calibration routine code
2. How devices are enumerated and stored
3. The device control functions (turn on/off)
4. Any existing device filtering or categorization logic
</research>

<requirements>
1. Modify the calibration routine to filter devices before touching them
2. Only include devices where the name contains (case-insensitive):
   - "light"
   - "input"
   - "output"
3. Skip all other devices during calibration (computers, servers, etc.)
4. Log which devices are being calibrated vs skipped for debugging
5. Preserve existing calibration behavior for filtered devices
</requirements>

<implementation>
Create a device filtering function that can be reused:

```python
def is_controllable_device(device_name: str) -> bool:
    """Check if device should be controlled during calibration."""
    name_lower = device_name.lower()
    return any(keyword in name_lower for keyword in ['light', 'input', 'output'])
```

Apply this filter in the calibration routine before any device control operations.

Avoid:
- Hardcoding specific device names (use pattern matching instead)
- Modifying device control functions themselves (only filter at calibration level)
- Breaking existing non-calibration device control flows
</implementation>

<output>
Modify existing files as needed. If creating new utility functions, place them in an appropriate existing module rather than creating new files.
</output>

<deployment>
After implementing changes:

1. **Commit and push**:
   ```bash
   git add -A
   git commit -m "Filter calibration to only touch light/input/output devices"
   git push
   ```

2. **Deploy to VM**:
   ```bash
   ssh 192.168.0.109
   cd /home/ahs/bluetti-monitor
   git pull
   docker-compose down && docker-compose up -d
   ```

3. **Verify live**:
   - Check container logs for calibration output
   - Confirm only light/input/output devices are touched
   - Verify other devices (computers, servers) are NOT affected
</deployment>

<verification>
Before declaring complete:
1. Trace through the calibration code path to confirm filtering is applied
2. Verify the filter is case-insensitive (matches "Light", "INPUT", "Output_1", etc.)
3. Confirm non-matching devices like "desktop_computer" or "server_rack" would be skipped
4. Check that logging shows which devices are included/excluded
</verification>

<success_criteria>
- Calibration routine only touches devices with light/input/output in their names
- Other devices (computers, servers, etc.) are completely untouched during startup
- Filtering is case-insensitive
- Debug logging shows filtered device list
</success_criteria>
