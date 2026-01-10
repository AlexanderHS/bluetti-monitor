<objective>
Implement three related improvements to the Bluetti Monitor system (in this order):

1. **Deployment Workflow**: Add standard deployment workflow to CLAUDE.md first (so we can follow it for subsequent steps)
2. **Hysteresis Threshold Feature**: Add configurable high/low battery thresholds to prevent output oscillation
3. **Documentation Update**: Update CLAUDE.md and README.md to reflect current .env configuration (including new hysteresis vars) and OCR approach (Tesseract, not LLM APIs)

The system is deployed at `ssh ahs@10.0.0.142` in `/home/ahs/bluetti-monitor`.
</objective>

<context>
This is a Python-based monitoring agent that:
- Captures images from an ESP32 webcam pointed at a Bluetti solar generator's LCD screen
- Extracts battery status using **Tesseract OCR** (recently migrated from LLM APIs)
- Exposes data via FastAPI endpoints
- Controls power outlets via a device control API based on battery percentage

Current problem: The output control oscillates too frequently around threshold boundaries. For example, at 59% battery under heavy sun, it hits 61%, turns output ON, then quickly drops back to 59% and turns OFF again, causing strain on controllers.

The solution is hysteresis: instead of a single 60% threshold, use two thresholds:
- Output turns ON when battery rises ABOVE the high threshold (e.g., 80%)
- Output turns OFF when battery falls BELOW the low threshold (e.g., 60%)
- Between thresholds, maintain current state
</context>

<requirements>

<task_1_deployment_workflow>
Add this deployment workflow section to CLAUDE.md so future sessions know the standard process:

```markdown
## Standard Deployment Workflow

When making changes to this project, follow this workflow:

1. **Make changes locally** (in the repo at the current working directory)
2. **Syntax check**: `python -m py_compile <changed_files>` (optional but recommended)
3. **Commit and push**: `git add . && git commit -m "message" && git push`
4. **SSH to production**: `ssh ahs@10.0.0.142`
5. **Pull changes**: `cd /home/ahs/bluetti-monitor && git pull`
6. **Rebuild and restart**: `docker compose down && docker compose up -d --build`
7. **Verify logs**: `docker compose logs -f` (wait ~30 seconds, check for errors)
8. **Exit SSH**: `exit`

Production server: `ssh ahs@10.0.0.142` at `/home/ahs/bluetti-monitor`
```

Commit and deploy this change first using the workflow above, so subsequent changes follow the documented process.
</task_1_deployment_workflow>

<task_2_hysteresis_implementation>
Add hysteresis-based output control with these new .env variables:

```
# Output Control Hysteresis Settings
OUTPUT_HIGH_THRESHOLD=80    # Output turns ON above this percentage
OUTPUT_LOW_THRESHOLD=60     # Output turns OFF below this percentage
```

Modify `recommendations.py` to implement hysteresis logic:
- When battery >= OUTPUT_HIGH_THRESHOLD: recommend output ON
- When battery <= OUTPUT_LOW_THRESHOLD: recommend output OFF
- Between thresholds: maintain current output state (requires knowing current state)

The current logic in `recommendations.py` uses fixed thresholds:
- < 20%: critical low, charge
- 20-60%: conservation mode
- >= 60%: use outputs

New logic should be:
- < 20%: critical low (charge, outputs off)
- 20 to OUTPUT_LOW_THRESHOLD: conservation (everything off)
- OUTPUT_LOW_THRESHOLD to OUTPUT_HIGH_THRESHOLD: maintain current state
- >= OUTPUT_HIGH_THRESHOLD: outputs on, inputs off

You'll need to:
1. Add new env vars to `.env.example`
2. Load them in the appropriate module (likely recommendations.py or a config module)
3. Modify `calculate_device_recommendations()` to accept current output state
4. Update callers to pass current output state
5. Implement the hysteresis logic
</task_2_hysteresis_implementation>

<task_3_documentation_update>
After hysteresis is implemented, update documentation to reflect current state:

**CLAUDE.md updates needed:**
- Update Configuration section with all current .env vars from `.env.example` (including new hysteresis vars)
- Remove references to LLM APIs (Gemini/GROQ) for primary OCR - we now use Tesseract
- Update Technology Stack to reflect Tesseract as primary OCR

**README.md updates needed:**
- Update the Configuration section to document key .env variables
- Update Technology Stack to mention Tesseract OCR
- Keep it concise - README should be overview, not exhaustive
- Update any outdated API response examples if needed

Reference `.env.example` for the authoritative list of environment variables.
</task_3_documentation_update>

</requirements>

<implementation_steps>
**Phase 1: Deployment Workflow (commit and deploy separately)**
1. Add deployment workflow section to `CLAUDE.md`
2. Commit and push this change
3. SSH to production, pull, rebuild, verify logs (following the new workflow)

**Phase 2: Hysteresis Implementation (commit and deploy separately)**
4. Add hysteresis env vars to `.env.example`
5. Implement hysteresis logic in `recommendations.py`
6. Update any callers of `calculate_device_recommendations()` to pass current output state
7. Run `python -m py_compile` on modified Python files
8. Commit and push
9. SSH to production, pull, rebuild, verify logs

**Phase 3: Documentation Update (commit and deploy separately)**
10. Update `CLAUDE.md` Configuration section with all current .env vars
11. Update `README.md` with accurate but concise information
12. Commit and push
13. SSH to production, pull, rebuild, verify logs
</implementation_steps>

<verification>
Before declaring complete:
1. All modified Python files pass `python -m py_compile`
2. Changes are committed and pushed to git
3. Production server has been updated via git pull
4. Docker containers have been rebuilt and restarted
5. Logs show no errors after ~30 seconds of running
6. CLAUDE.md now contains the deployment workflow for future reference
</verification>

<success_criteria>
- CLAUDE.md accurately reflects current .env vars and Tesseract OCR approach
- CLAUDE.md includes standard deployment workflow
- README.md is updated and accurate
- Hysteresis thresholds are configurable via .env
- Output control no longer oscillates near single threshold
- Production deployment is successful with no errors in logs
</success_criteria>
