<objective>
Build a local template matching image classifier to replace Gemini API for battery percentage recognition (0-100%).

The system needs two modes:
1. **Collection mode** (always active): Save every captured image with its Gemini-classified label
2. **Comparison mode** (activates at 50% coverage): Log what template matching WOULD classify, alongside Gemini's classification, for accuracy observation
</objective>

<context>
Current system uses Gemini API (`gemini_ocr.py`) to classify LCD screen images showing battery percentages 0-100%. This is overkill for 101 discrete classes.

Key files to examine:
- `./gemini_ocr.py` - Current Gemini-based classification
- `./main.py` - FastAPI endpoints, image capture flow
- `./worker.py` - Background processing that calls Gemini

The battery percentage is displayed on an LCD screen, captured by ESP32 webcam. Images have minor variations in lighting/position but the digits are consistent.
</context>

<requirements>
1. **Image Storage Structure**
   - Create `./training_data/` directory
   - Subdirectories per percentage: `./training_data/0/`, `./training_data/1/`, ... `./training_data/100/`
   - Save images as timestamped JPEGs: `{timestamp}.jpg`
   - Target: 10 images per percentage value
   - **FIFO rotation**: If a percentage already has 10 images, delete the oldest before adding new one (keeps storage bounded while maintaining fresh examples)

2. **Collection Integration**
   - Hook into existing capture flow (after Gemini successfully classifies)
   - Only save when Gemini returns a valid percentage (0-100)
   - Save the exact image that was classified
   - Log collection progress periodically

3. **Template Matching Classifier**
   - Use OpenCV for image similarity (normalized cross-correlation or structural similarity)
   - Load reference images from `./training_data/` on startup
   - For each percentage with images, compute average/representative template
   - Match new images against all available templates, return best match + confidence score

4. **Coverage Tracking**
   - Track which percentages have at least 1 example
   - Calculate coverage percentage: (values with examples / 101) * 100
   - Expose via API endpoint: `GET /training/status`

5. **Comparison Mode** (activates when coverage >= 50%)
   - When coverage threshold met, enable parallel classification
   - Log both Gemini result AND template matching result for every image
   - Format: `[COMPARE] Gemini: {gemini_pct}% | Template: {template_pct}% (confidence: {conf}) | Match: {yes/no}`
   - Continue using Gemini as source of truth (don't change actual system behavior)

6. **API Endpoints**
   - `GET /training/status` - Returns coverage stats, images per percentage, total images
   - `GET /training/enable` - Manually enable/disable collection (default: enabled)
   - `POST /training/label/{percentage}` - Manually label/relabel last captured image (for corrections)
</requirements>

<implementation>
Create a new module `./template_classifier.py` containing:
- `TemplateClassifier` class with methods for loading templates, matching, and saving images
- Coverage tracking logic
- Comparison logging

Integration points:
- Modify the capture/classification flow to save images after successful Gemini classification
- Add comparison logging when coverage threshold is met
- Add new API endpoints to `main.py`

Use OpenCV functions already available (project uses cv2 for image processing).
</implementation>

<output>
Create/modify files:
- `./template_classifier.py` - New module with TemplateClassifier class
- `./main.py` - Add training status endpoints, integrate collection
- `./worker.py` - Hook collection into existing Gemini classification flow (if classification happens there)

Do NOT modify `./gemini_ocr.py` - keep it as-is for comparison baseline.
</output>

<verification>
Before declaring complete:
1. Verify `./training_data/` directory structure is created correctly
2. Verify images are saved after successful Gemini classification
3. Verify FIFO rotation works (oldest image deleted when count exceeds 10)
4. Verify `/training/status` endpoint returns accurate coverage data
5. Verify comparison logging activates only when coverage >= 50%
6. Test template matching returns reasonable results with a few sample images
</verification>

<success_criteria>
- Images automatically saved to correct percentage subdirectory after Gemini classification
- FIFO rotation keeps max 10 images per percentage (oldest deleted when adding 11th)
- Coverage tracking accurately reflects stored images
- Template matching classifier produces percentage predictions with confidence scores
- Comparison logs appear when coverage >= 50%
- No changes to existing Gemini-based system behavior (it remains source of truth)
</success_criteria>
