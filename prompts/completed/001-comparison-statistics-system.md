<objective>
Build a systematic comparison statistics system for the template vs LLM (Gemini/Groq) classification strategies.

The goal is to enable data-driven analysis of where and when the strategies agree or disagree, allowing identification of patterns (e.g., "always disagrees on 21%, reliable on 40-49%") and case-by-case review with full context.
</objective>

<context>
This is a Bluetti solar generator monitoring system that uses OCR to read battery percentage from an LCD screen.

**Current Architecture:**
- Primary: Gemini LLM OCR with Groq fallback
- Secondary: Template matching classifier (local, trained on Gemini-labeled data)
- Both strategies process images, but only Gemini results are used for logic
- Comparison logging exists via `log_comparison()` but only to stdout logs
- Training review UI at `/training/review` shows individual training images

**Key Files to Examine:**
- `main.py` - FastAPI endpoints, /training/review UI
- `worker.py` - Main capture/classification loop, calls both strategies
- `template_classifier.py` - Template matching, `log_comparison()` function
- `gemini_ocr.py` and `groq_ocr.py` - LLM-based OCR
- `.env.example` - Environment configuration

**Problem Being Solved:**
No systematic way to review comparison data. Logs are ephemeral. Cannot answer "does template always fail on value X?" or review specific disagreements with the actual image.
</context>

<requirements>

<storage>
Create a SQLite-based comparison statistics storage system:

1. **New Table: `comparison_records`**
   - timestamp (REAL)
   - gemini_percentage (INTEGER, nullable - may be None if LLM failed)
   - groq_percentage (INTEGER, nullable - captured if Groq was used as fallback)
   - template_percentage (INTEGER, nullable - may be None if template failed)
   - template_confidence (REAL, nullable)
   - human_verified_percentage (INTEGER, nullable - populated if later verified via UI)
   - image_filename (TEXT - reference to saved image)
   - agreement (BOOLEAN - did strategies agree?)
   - llm_source (TEXT - "gemini", "groq", or "none")

2. **FIFO Management:**
   - Keep maximum 1000 comparison records
   - When inserting record 1001, delete oldest record
   - Also delete associated image file when purging record
   - Store images in `./data/comparison_images/` directory

3. **Image Storage:**
   - Save a copy of each compared image (JPEG, reasonable quality)
   - Filename format: `{timestamp}_{gemini}_{template}.jpg`
   - Must respect FIFO - delete image when record is purged
</storage>

<env_configuration>
Add new environment variables to control strategy behavior:

```
# Strategy Control
ENABLE_GEMINI_STRATEGY=true          # Enable Gemini/Groq LLM OCR
ENABLE_TEMPLATE_STRATEGY=true        # Enable template matching classifier
PRIMARY_STRATEGY=llm                 # Which strategy to use for logic: "llm" or "template"

# Comparison Storage
MAX_COMPARISON_RECORDS=1000          # FIFO limit for comparison statistics
```

**Behavior Rules:**
- If both strategies disabled: error on startup
- If only one enabled: use that strategy, no comparison logging
- If both enabled: run both, log comparison, use PRIMARY_STRATEGY for logic
- Default (no env vars set): current behavior - LLM primary, template logged
- `PRIMARY_STRATEGY=template` means template result goes to database/logic
</env_configuration>

<statistics_api>
Create new API endpoints for comparison statistics:

1. **GET /comparisons/stats**
   Returns aggregated statistics:
   ```json
   {
     "total_comparisons": 847,
     "agreement_rate": 0.923,
     "by_value": {
       "0": {"count": 12, "agreements": 12, "agreement_rate": 1.0},
       "21": {"count": 8, "agreements": 3, "agreement_rate": 0.375},
       ...
     },
     "by_llm_source": {
       "gemini": {"count": 800, "agreement_rate": 0.95},
       "groq": {"count": 47, "agreement_rate": 0.72}
     },
     "recent_disagreements": 5
   }
   ```

2. **GET /comparisons/records**
   Query parameters: `?limit=50&offset=0&filter=disagreements|all&value=21`
   Returns paginated comparison records for review.

3. **GET /comparisons/image/{filename}**
   Serve comparison images from storage.

4. **POST /comparisons/verify/{record_id}**
   Body: `{"human_percentage": 45}`
   Update record with human-verified ground truth.
</statistics_api>

<ui_extension>
Extend the existing `/training/review` page with a new "Comparisons" tab:

**Tab Structure:**
- Tab 1: "Training Images" (existing functionality)
- Tab 2: "Comparison Stats" (new)

**Comparison Stats Tab Contents:**

1. **Summary Cards Row:**
   - Total comparisons (with date range)
   - Overall agreement rate (percentage with color: green >90%, yellow 70-90%, red <70%)
   - Recent disagreements count (last 24h)
   - Primary strategy indicator

2. **Per-Value Agreement Table:**
   - Sortable table with columns: Value (0-100), Count, Agreements, Disagreements, Agreement Rate
   - Color-coded rows: red for low agreement (<70%), yellow for moderate (70-90%)
   - Click row to filter records list below

3. **Comparison Records List:**
   - Shows individual comparison records
   - Each record displays:
     - Timestamp
     - Gemini result (or Groq with indicator)
     - Template result + confidence
     - Human verified result (if exists) with edit button
     - Agreement status (✓ or ✗)
     - Thumbnail of image (click to enlarge)
   - Filter controls: All / Disagreements only / By specific value
   - Pagination

4. **Record Detail Modal:**
   - Large image display
   - Side-by-side comparison: Gemini | Template | Human
   - Buttons to set human-verified value (0-100 grid like existing reclassify)
   - Shows confidence scores and metadata
</ui_extension>

</requirements>

<implementation>

**Order of Implementation:**
1. Database schema changes (add comparison_records table)
2. Environment variable handling
3. Comparison storage class/functions with FIFO
4. Modify worker.py to save comparisons (respecting env vars)
5. API endpoints for statistics
6. UI tab extension

**Key Integration Points:**
- In `worker.py` after both strategies run, call comparison storage
- Respect `ENABLE_*` flags before running each strategy
- Use `PRIMARY_STRATEGY` to determine which result goes to battery_readings table
- Groq fallback should be captured in comparison records when it occurs

**Error Handling:**
- If comparison storage fails, log error but don't fail the main capture cycle
- Handle missing images gracefully in UI (show placeholder)
- Validate human_percentage input (0-100 integer only)

</implementation>

<output>
Modify existing files:
- `main.py` - Add comparison endpoints and extend /training/review UI
- `worker.py` - Integrate comparison storage, respect env flags
- `template_classifier.py` - May need adjustments for new flow
- `.env.example` - Document new environment variables

Create if needed:
- `comparison_storage.py` - Dedicated module for comparison FIFO storage (optional, can be in main.py)

Update:
- `./data/` directory structure to include `comparison_images/`
</output>

<verification>
Before declaring complete, verify:

1. **Storage works:** Insert test comparison records, verify FIFO purges at 1001
2. **Images managed:** Confirm image files are created and deleted with records
3. **Env vars work:** Test with different combinations:
   - Both enabled (default) - should compare and log
   - Only LLM enabled - should skip template, no comparison
   - PRIMARY_STRATEGY=template - should use template for logic
4. **API endpoints:** Test /comparisons/stats returns valid aggregations
5. **UI renders:** New tab appears, shows data, filtering works
6. **Human verification:** Can mark a record with human ground truth
</verification>

<success_criteria>
- Comparison data persists across restarts (SQLite storage)
- FIFO correctly limits to 1000 records with image cleanup
- Per-value statistics clearly show which values have agreement problems
- Individual records can be reviewed with actual image
- Human can provide ground truth for any comparison
- Strategy selection via .env works without code changes
- Existing functionality unchanged when no new env vars set
</success_criteria>
