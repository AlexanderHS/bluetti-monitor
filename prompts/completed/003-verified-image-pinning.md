<objective>
Implement a verified/pinned image system for training data that protects manually-confirmed images from FIFO deletion while always accepting new samples.

Currently, the FIFO rotation treats all images equally, so manually verified training samples get deleted over time. This causes:
1. Re-reviewing the same images without knowing they were already confirmed
2. Loss of high-quality, human-verified training data

The solution: Add a "verified" flag that protects images from deletion (unless all images are verified), with UI controls to verify images and filter the review view.
</objective>

<context>
This is a Python-based monitoring agent using FastAPI that captures battery percentage images and classifies them using template matching (OpenCV).

Key files to examine:
- `./main.py` - FastAPI endpoints including `/training/review` web UI
- `./template_classifier.py` - Template loading and FIFO rotation logic
- `./docker-compose.yaml` - Environment variable configuration
- `./.env.example` - Environment variable documentation

Read CLAUDE.md for project conventions.
</context>

<requirements>

1. **Verified Flag Storage**
   - Use filename suffix convention: `image.jpg` (unverified) vs `image.verified.jpg` (verified)
   - This keeps verification state with the file and survives moves between categories
   - No separate manifest or database needed

2. **Smart FIFO Deletion Logic**
   - When at capacity (max images per category):
     - First, delete oldest UNVERIFIED image
     - Only delete oldest VERIFIED image if ALL images in category are verified
   - Always accept new incoming images (to capture lighting/camera changes)
   - New auto-labeled images from Gemini are always unverified

3. **Automatic Verification on Reclassify**
   - When user reclassifies an image (moves from one category to another), automatically mark as verified
   - Why: Reclassification implies human review and correction

4. **Manual Verify Button**
   - Add "Verify" button on each image card for images already in correct category
   - Clicking verify renames the file to add `.verified.` suffix
   - Add "Verify All Visible" bulk action button to verify all currently displayed images

5. **Filter Toggle in UI**
   - Add filter control: "All" | "Unverified only"
   - Default to "Unverified only" so reviewer focuses on images needing attention
   - Filter state should persist in URL query param or localStorage

6. **Visual Indicator**
   - Show small checkmark or "verified" badge on verified images when viewing "All"

7. **Environment Variables**
   - Add to `.env.example` with sensible defaults:
     ```
     # Template classifier settings
     MAX_IMAGES_PER_CATEGORY=10
     COMPARISON_MODE_THRESHOLD=0.25
     ```
   - Update `template_classifier.py` to read these from environment
   - Update `docker-compose.yaml` to pass environment variables to container

</requirements>

<implementation>

1. **Start with template_classifier.py**:
   - Update `_load_templates()` to recognize `.verified.` suffix
   - Modify FIFO rotation in `save_labeled_image()` to prefer deleting unverified
   - Add helper methods: `is_verified(filename)`, `mark_verified(filepath)`
   - Read MAX_IMAGES_PER_CATEGORY and COMPARISON_MODE_THRESHOLD from os.getenv()

2. **Update main.py endpoints**:
   - Modify `/training/reclassify` to mark destination file as verified
   - Add `/training/verify/{category}/{filename}` POST endpoint
   - Add `/training/verify-all` POST endpoint (verifies visible images by category)
   - Update `/training/images` to include `verified` boolean in response
   - Add `filter` query param to `/training/review` (all | unverified)

3. **Update review UI HTML**:
   - Add filter toggle buttons (All / Unverified)
   - Add "Verify" button on each image card
   - Add "Verify All Visible" button in header
   - Show verified badge on verified images
   - Wire up JavaScript for verify actions and filter state

4. **Update configuration files**:
   - Add variables to `.env.example`
   - Update `docker-compose.yaml` to pass env vars

</implementation>

<verification>
Before declaring complete, verify:
1. Run `python -c "from template_classifier import TemplateClassifier; tc = TemplateClassifier(); print(tc.max_images_per_percentage)"` - should read from env
2. Create a test verified file: `touch training_data/50/test.verified.jpg` and verify it loads
3. Test the review UI shows filter toggle and verify buttons
4. Test reclassifying an image marks it as verified
5. Test FIFO deletes unverified before verified
6. `docker-compose config` should show the new env vars
</verification>

<success_criteria>
- Verified images use `.verified.` filename suffix
- FIFO deletes unverified images first, verified only when necessary
- Reclassifying auto-verifies the image
- "Verify" button works on individual images
- "Verify All Visible" bulk action works
- Filter toggle shows All / Unverified only
- Verified images show visual badge
- Config values come from environment variables
- Docker compose passes env vars correctly
</success_criteria>
