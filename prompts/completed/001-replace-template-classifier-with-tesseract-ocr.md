<objective>
Replace the internal classification logic in `template_classifier.py` with the proven Tesseract OCR approach documented in the research findings.

The research demonstrates 96.6% accuracy with OCR vs 53.8% for template matching - a dramatic improvement that maintains the existing interface for seamless integration.
</objective>

<context>
This Bluetti Monitor system captures LCD screen images from an ESP32 webcam and extracts battery percentages. Currently, `template_classifier.py` uses whole-image correlation matching which suffers from systematic digit confusion (9→3, 7→2, 4→8).

The research in `@docs/ocr-classifier-research.md` has validated an optimal OCR pipeline:
- Preprocessing: Otsu threshold + invert + 5px padding
- Tesseract config: PSM 6, digits-only whitelist
- No scaling (original 75x70 resolution is optimal)
- Dependencies already available in Docker container

Examine these files for full context:
- `@template_classifier.py` - Current implementation to modify
- `@worker.py` - Consumer of the classifier (no changes needed)
- `@docs/ocr-classifier-research.md` - Research with proven code snippets
- `@ocr_test.py` - Tests 10 preprocessing strategies (found v5_invert_threshold best)
- `@ocr_tune.py` - Tunes PSM/padding/scaling/OEM modes (found PSM 6 + 5px padding best)
</context>

<requirements>
1. **Replace `_preprocess_image()` method** with the OCR preprocessing pipeline:
   - Otsu threshold → invert → 5px white padding
   - Remove histogram equalization (it hurts OCR accuracy)

2. **Replace `classify_image()` internals** with Tesseract OCR:
   - Use config: `--psm 6 -c tessedit_char_whitelist=0123456789`
   - Extract digits with regex, validate 0-100 range
   - Keep existing black screen detection BEFORE preprocessing
   - Return None for invalid/failed OCR (existing interface)
   - Return dict with success/percentage/confidence on success

3. **Maintain existing interface** - worker.py should require NO changes:
   - `classify_image(image_data: bytes) -> Optional[Dict]`
   - Return format: `{"success": True, "percentage": int, "confidence": float, ...}`
   - Return None for black screens and invalid images

4. **Remove template matching code**:
   - Delete `_load_templates()`, `_compute_similarity()` methods
   - Delete `self.templates` dictionary
   - Keep training data directory structure (used by collection mode)

5. **Keep collection mode active** for OCR accuracy monitoring:
   - `save_labeled_image()` continues saving LLM-labeled images
   - The `/training/review` endpoint can be repurposed for verifying OCR accuracy
   - This enables ongoing accuracy tracking and future improvements

6. **Update `get_preprocessed_image()` method** to use new preprocessing
</requirements>

<implementation>
Use the exact preprocessing and OCR code from the research document - it's been validated on 199 samples.

Key code from research (preserve exactly):
```python
# Preprocessing
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
inverted = cv2.bitwise_not(binary)
padded = cv2.copyMakeBorder(inverted, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)

# OCR config
TESSERACT_CONFIG = '--psm 6 -c tessedit_char_whitelist=0123456789'
```

**WHY these specific parameters:**
- PSM 6 (uniform block) outperformed PSM 7 (single line) by 6.8 percentage points
- 5px padding improves edge digit recognition (+1.4pp)
- Invert is required because Tesseract expects black text on white background
- No scaling - the research showed 2x/3x scaling DECREASED accuracy

Avoid:
- Adding new dependencies (pytesseract/cv2/numpy already available)
- Changing the external interface
- Creating new files - edit template_classifier.py only
</implementation>

<output>
Modify: `./template_classifier.py`
- Replace preprocessing with OCR pipeline
- Replace classify_image internals with Tesseract
- Remove unused template matching methods and data structures
- Update docstrings to reflect new OCR approach
</output>

<verification>
**Use the existing test scripts to verify accuracy:**

1. Run preprocessing strategy test (should confirm v5_invert_threshold wins):
```bash
docker cp ocr_test.py bluetti-monitor-bluetti-monitor-api-1:/app/
docker exec bluetti-monitor-bluetti-monitor-api-1 python /app/ocr_test.py
```
Expected: v5_invert_threshold ~88.4% accuracy (baseline before tuning)

2. Run the tuning script (tests PSM modes, padding, scaling):
```bash
docker cp ocr_tune.py bluetti-monitor-bluetti-monitor-api-1:/app/
docker exec bluetti-monitor-bluetti-monitor-api-1 python /app/ocr_tune.py
```
Expected: PSM 6 + 5px padding achieves ~96.6% accuracy

**Verify interface compatibility:**
- `template_classifier.classify_image(image_bytes)` still works
- Returns None for black screens and invalid images
- Returns dict with success/percentage/confidence for valid readings
- worker.py requires NO code changes
</verification>

<success_criteria>
- Template classifier uses Tesseract OCR internally
- Existing worker.py code works without modification
- Black screen detection still functions
- Collection mode (save_labeled_image) still works for accuracy monitoring
- /training/review endpoint remains functional
- No new dependencies required
- ocr_tune.py reports ~96.6% accuracy when run against classifier
</success_criteria>
