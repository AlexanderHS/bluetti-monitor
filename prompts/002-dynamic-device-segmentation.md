<objective>
Implement dynamic device discovery with time-based segmentation for multiple inputs/outputs.

The system needs flexibility to handle varying configurations (1 input + 1 output, 1 input + 2 outputs, etc.). When multiple inputs or outputs exist, implement time-based segmentation so only one is active at a time within each hour.
</objective>

<context>
This is part of a Bluetti solar generator monitoring system. The configuration changes over time as hardware is added/removed. Rather than hardcoding device counts, the system should discover devices dynamically and adapt its behavior.

The existing threshold-based logic determines WHEN to turn devices on/off based on battery readings. This change affects WHICH devices get controlled and HOW they're rotated.

Read CLAUDE.md first for project conventions, then examine the codebase to understand:
- How devices are currently discovered and categorized
- The existing threshold logic that triggers device control
- Current input/output handling code
</context>

<research>
Before implementing, find and understand:
1. Device discovery mechanism
2. How inputs and outputs are currently identified/stored
3. The threshold logic that triggers on/off decisions
4. Where device control commands are issued
</research>

<requirements>
1. **Device Discovery**:
   - Search for all available devices at startup
   - Categorize devices as "input" or "output" based on names (case-insensitive matching for "input" or "output" in device name)
   - Store discovered inputs and outputs separately

2. **Single Device Mode**:
   - If only ONE input exists: control it directly when threshold logic says to
   - If only ONE output exists: control it directly when threshold logic says to
   - This should work regardless of device naming (e.g., "InpUt_123x" counts as an input)

3. **Multi-Device Segmentation Mode** (when multiple inputs OR outputs exist):
   - Divide each hour into equal segments based on device count
   - For N devices, each device gets 60/N minutes per hour
   - Example with 3 outputs:
     - Minutes 0-19: Output 1 active, Outputs 2-3 OFF
     - Minutes 20-39: Output 2 active, Outputs 1,3 OFF
     - Minutes 40-59: Output 3 active, Outputs 1-2 OFF
   - When threshold logic says "turn on output", only turn on the currently-active-segment device
   - Explicitly turn OFF devices not in their time segment

4. **Integration**:
   - Integrate with existing threshold logic (don't modify threshold calculations)
   - When threshold says "turn on", apply to current segment's device
   - When threshold says "turn off", turn off all devices of that type
</requirements>

<implementation>
Suggested approach:

```python
def get_active_device_index(device_count: int) -> int:
    """Determine which device should be active based on current time."""
    if device_count <= 1:
        return 0

    current_minute = datetime.now().minute
    segment_duration = 60 // device_count
    return current_minute // segment_duration

def control_outputs(should_be_on: bool, outputs: list):
    """Control outputs with segmentation if multiple exist."""
    if not outputs:
        return

    if len(outputs) == 1:
        # Single output mode - direct control
        set_device_state(outputs[0], should_be_on)
    else:
        # Multi-output segmentation mode
        active_index = get_active_device_index(len(outputs))
        for i, output in enumerate(outputs):
            if i == active_index and should_be_on:
                set_device_state(output, True)
            else:
                set_device_state(output, False)  # Explicitly OFF
```

Apply similar logic for inputs if needed.

Avoid:
- Hardcoding device counts or names
- Modifying the threshold calculation logic
- Creating race conditions between segment transitions
</implementation>

<output>
Modify existing files as needed. If creating new utility functions, place them in an appropriate existing module.
</output>

<deployment>
After implementing changes:

1. **Commit and push**:
   ```bash
   git add -A
   git commit -m "Add dynamic device discovery with time-based segmentation"
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
   - Check container logs for device discovery output
   - Confirm inputs/outputs are detected correctly
   - If multiple outputs exist, verify time-based segmentation is working
   - Monitor that only the active segment's device turns on when threshold triggers
</deployment>

<verification>
Before declaring complete:
1. Test device discovery finds inputs/outputs by name pattern (case-insensitive)
2. Verify single-device mode works (one input, one output)
3. Verify segmentation math: 3 devices = 20-minute segments, 2 devices = 30-minute segments
4. Confirm devices outside their time segment are explicitly turned OFF
5. Verify integration with existing threshold logic still works
</verification>

<success_criteria>
- Devices are dynamically discovered by searching for "input"/"output" in names (case-insensitive)
- Single input/output configurations work directly
- Multiple inputs/outputs use time-based segmentation within each hour
- Only the current segment's device is ON when threshold logic activates
- Devices outside their segment are explicitly turned OFF
- Existing threshold logic continues to determine WHEN to activate
</success_criteria>
