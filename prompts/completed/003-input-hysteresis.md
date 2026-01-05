<objective>
Add hysteresis-based input control to prevent oscillation around the charging threshold, mirroring the output hysteresis already implemented.
</objective>

<context>
We just implemented output hysteresis with OUTPUT_HIGH_THRESHOLD (80%) and OUTPUT_LOW_THRESHOLD (60%). The same oscillation problem exists for inputs - if battery hovers around 20%, inputs could toggle on/off rapidly.

Current input logic in `recommendations.py`:
- < 20%: Turn input ON (critical low, charging needed)
- >= 20%: Turn input OFF

With hysteresis:
- Below INPUT_LOW_THRESHOLD (e.g., 20%): Turn inputs ON
- Above INPUT_HIGH_THRESHOLD (e.g., 40%): Turn inputs OFF
- Between thresholds: Maintain current input state

Production server: `ssh ahs@10.0.0.109` at `/home/ahs/bluetti-monitor`
</context>

<requirements>
Add new .env variables to `.env.example`:
```
# Input Control Hysteresis Settings
INPUT_LOW_THRESHOLD=20      # Inputs turn ON below this percentage (start charging)
INPUT_HIGH_THRESHOLD=40     # Inputs turn OFF above this percentage (stop charging)
```

Update `recommendations.py`:
1. Load INPUT_LOW_THRESHOLD and INPUT_HIGH_THRESHOLD from environment
2. Add `current_input_on: Optional[bool] = None` parameter to `calculate_device_recommendations()`
3. Implement input hysteresis logic:
   - Below INPUT_LOW_THRESHOLD: Inputs ON, outputs OFF (critical)
   - INPUT_LOW_THRESHOLD to INPUT_HIGH_THRESHOLD: Maintain current input state, outputs OFF
   - Above INPUT_HIGH_THRESHOLD: Use existing output hysteresis logic (inputs OFF)
4. Update `analyze_recent_readings_for_recommendations()` to accept and pass through `current_input_on`

Update `worker.py`:
1. Update `control_devices_based_on_battery()` to accept `current_input_on` parameter
2. Pass `input_on` (already queried) to the function call

Update documentation:
1. Add input hysteresis settings to CLAUDE.md Configuration section
2. Add to README.md Key Settings
</requirements>

<implementation_steps>
1. Add input hysteresis env vars to `.env.example`
2. Update `recommendations.py` with input hysteresis logic
3. Update `worker.py` to pass current input state
4. Run `python -m py_compile` on modified files
5. Update CLAUDE.md and README.md with new settings
6. Commit and push
7. SSH to production, pull, rebuild, verify logs
</implementation_steps>

<verification>
1. All Python files pass syntax check
2. Changes committed and pushed
3. Production deployed and logs show input hysteresis working
4. Example log: "Battery at 25% - input hysteresis zone (20-40%), maintaining inputs OFF"
</verification>

<success_criteria>
- Input thresholds configurable via .env
- No oscillation when battery hovers around 20%
- Logs show hysteresis zone messages for inputs
- Production deployment successful
</success_criteria>
