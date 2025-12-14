<objective>
Add an "invalid" category to the training image review system for handling garbled, black, or otherwise unreadable images.

This is needed because the screen-off detection sometimes fails, resulting in images that get falsely classified under a percentage number when they should be marked as unreadable. The classifier should skip these images silently rather than updating battery status with incorrect data.
</objective>

<context>
This is a Python-based monitoring agent that captures images from an ESP32 webcam pointed at a Bluetti solar generator's LCD screen. It uses template matching (OpenCV) to classify battery percentage from captured images.

Key files to examine:
- `./main.py` - FastAPI endpoints including `/training/review`
- `./template_classifier.py` - OpenCV template matching classifier
- `./training_data/` - Directory containing training images organized by percentage

Read CLAUDE.md for project conventions.
</context>

<requirements>
1. **Training Review UI Changes:**
   - Add "invalid" as a new reclassification option in the web UI at `/training/review`
   - The "invalid" option should appear alongside percentage options (0-100)
   - Moving an image to "invalid" should relocate it to `./training_data/invalid/`

2. **Template Classifier Changes:**
   - Load templates from `./training_data/invalid/` along with percentage templates
   - When the best match is an "invalid" template:
     - Log at DEBUG level: "Image matched invalid template, skipping cycle"
     - Return a value that indicates "skip this cycle" (do not update battery status)
   - The calling code should handle this gracefully and continue to the next capture cycle

3. **Directory Structure:**
   - Create `./training_data/invalid/` directory if it doesn't exist during template loading
   - Handle the case where no invalid templates exist yet (classifier works normally)
</requirements>

<implementation>
1. Start by reading `./main.py` to understand the current training review endpoint structure
2. Read `./template_classifier.py` to understand how templates are loaded and matched
3. Modify the reclassify endpoint to accept "invalid" as a valid target category
4. Update the web UI to include "invalid" in the dropdown/options
5. Modify template loading to include invalid templates
6. Update classification logic to detect invalid matches and return appropriately
7. Ensure the main processing loop handles the "skip" response correctly

Use None or a sentinel value to indicate "invalid/skip" - the calling code should check for this and continue without updating status.
</implementation>

<verification>
Before declaring complete, verify:
1. The `/training/review` UI shows "invalid" as an option
2. Reclassifying an image to "invalid" moves it to `./training_data/invalid/`
3. The template classifier loads invalid templates without error
4. When an image matches an invalid template, it logs at DEBUG and returns skip indicator
5. Run `python -c "from template_classifier import TemplateClassifier; tc = TemplateClassifier(); print(tc)"` to verify no import errors
6. Test the Docker build: `docker-compose build` should succeed
</verification>

<success_criteria>
- "invalid" category appears in training review UI
- Images can be reclassified to "invalid" and moved to correct directory
- Classifier loads and matches invalid templates
- Invalid matches cause silent skip (DEBUG log only, no status update)
- No regressions in normal percentage classification
</success_criteria>
