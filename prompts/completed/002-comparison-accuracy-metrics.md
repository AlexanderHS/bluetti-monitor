<objective>
Enhance the comparison statistics system to track who was right/wrong when human verification is available, enabling assessment of each method's accuracy and identification of systematic error patterns.

The goal is to answer: "Which method should I trust?" and "What specific errors does each method make?"
</objective>

<context>
The comparison statistics system (just implemented) tracks disagreements between LLM (Gemini/Groq) and template strategies, but doesn't leverage human verification to determine accuracy.

**Current Gap:**
- User reclassifies training images (e.g., marks "100%" as "invalid")
- This ground truth doesn't flow to comparison records
- Comparison stats show "disagreements at 100%" but not "LLM was wrong 12 times at 100%"
- No way to assess which method is more reliable overall

**Key Use Case:**
Gemini sometimes classifies black/invalid screens as 100%. User catches this in training review and reclassifies to "invalid". The comparison system should:
1. Link this verification to the comparison record (by timestamp)
2. Show that LLM was wrong (not just "disagreement")
3. Track this as an "invalid→100%" error pattern for LLM

**Key Files:**
- `main.py` - Training reclassification endpoints, comparison UI
- `comparison_storage.py` - Comparison records storage
- `template_classifier.py` - Training image management
</context>

<requirements>

<ground_truth_linking>
When a training image is reclassified, update the corresponding comparison record:

1. **Find matching comparison record:**
   - Training images have timestamp in filename (e.g., `1702656789.123_cropped.jpg`)
   - Comparison records have timestamp field
   - Match within ±2 seconds tolerance (same capture cycle)

2. **Update comparison record:**
   - Set `human_verified_percentage` to the reclassified value
   - For "invalid" category, use special value (e.g., -1 or null with separate `is_invalid` flag)
   - Recalculate `agreement` field based on ground truth:
     - If human says X, and LLM said X → LLM was correct
     - If human says X, and template said X → Template was correct

3. **Modify `/training/reclassify` endpoint:**
   - After moving training image, find and update matching comparison record
   - Log the linkage for debugging
</ground_truth_linking>

<accuracy_metrics>
Extend `/comparisons/stats` to include accuracy metrics based on human-verified records:

```json
{
  "total_comparisons": 847,
  "agreement_rate": 0.923,
  "human_verified_count": 156,

  "accuracy": {
    "llm": {
      "total_verified": 156,
      "correct": 142,
      "accuracy_rate": 0.910,
      "errors_by_type": {
        "invalid_as_100": 8,
        "off_by_one": 4,
        "off_by_more": 2
      }
    },
    "template": {
      "total_verified": 156,
      "correct": 134,
      "accuracy_rate": 0.859,
      "errors_by_type": {
        "invalid_as_value": 3,
        "off_by_one": 12,
        "off_by_more": 7
      }
    }
  },

  "error_patterns": [
    {"method": "llm", "predicted": 100, "actual": "invalid", "count": 8},
    {"method": "template", "predicted": 21, "actual": 27, "count": 3},
    ...
  ],

  "by_value": { ... },
  "by_llm_source": { ... }
}
```

**Error Type Definitions:**
- `invalid_as_X`: Method said X% but image was invalid/unreadable
- `X_as_invalid`: Method said invalid but image was actually X%
- `off_by_one`: Method was ±1% from correct value
- `off_by_more`: Method was >1% from correct value
</accuracy_metrics>

<ui_enhancements>
Enhance the Comparison Stats tab to show accuracy insights:

1. **Summary Cards Row (update):**
   - Add: "LLM Accuracy: 91%" (green/yellow/red based on rate)
   - Add: "Template Accuracy: 86%" (green/yellow/red based on rate)
   - Show "(based on N verified)" under each

2. **New Section: Error Patterns**
   - Table showing most common errors by each method
   - Columns: Method, Predicted, Actual, Count
   - Sorted by count descending
   - Color-coded by method (distinguish LLM vs Template errors)

3. **Per-Value Table (update):**
   - Add columns: "LLM Errors", "Template Errors" (when human-verified)
   - Highlight values where one method consistently fails

4. **Comparison Records List (update):**
   - When human-verified, show verdict:
     - "✓ LLM correct" (green)
     - "✗ LLM wrong" (red)
     - "✓ Template correct" (green)
     - "✗ Template wrong" (red)
   - Show the ground truth value prominently

5. **Record Detail Modal (update):**
   - Three-column comparison: LLM | Template | Ground Truth
   - Visual indicator showing which method(s) were correct
   - If not yet verified, show "Verify" button prominently
</ui_enhancements>

<invalid_handling>
Special handling for "invalid" category in comparisons:

1. **Storage:**
   - Add `human_verified_invalid` BOOLEAN column to comparison_records
   - When human marks as invalid: set `human_verified_percentage = NULL`, `human_verified_invalid = TRUE`

2. **Agreement calculation with invalid:**
   - If human says invalid AND template said invalid → template correct
   - If human says invalid AND LLM said any number → LLM wrong
   - If human says X% AND template said invalid → template wrong

3. **Display:**
   - Show "Invalid" clearly in UI (not as a percentage)
   - Error patterns should distinguish "invalid→100%" from "47→48"
</invalid_handling>

</requirements>

<implementation>

**Database Changes:**
- Add `human_verified_invalid` BOOLEAN column to `comparison_records`
- Add index on `human_verified_percentage` for faster accuracy queries

**Order of Implementation:**
1. Add `human_verified_invalid` column with migration
2. Update `comparison_storage.py` with accuracy calculation methods
3. Modify training reclassify endpoint to link to comparison records
4. Extend `/comparisons/stats` with accuracy metrics
5. Update UI with accuracy displays and error patterns

**Key Queries:**
```sql
-- LLM accuracy
SELECT
  COUNT(*) as total,
  SUM(CASE WHEN
    (human_verified_percentage = gemini_percentage OR human_verified_percentage = groq_percentage)
    OR (human_verified_invalid = 1 AND template_percentage IS NULL)
  THEN 1 ELSE 0 END) as llm_correct
FROM comparison_records
WHERE human_verified_percentage IS NOT NULL OR human_verified_invalid = 1;

-- Error patterns
SELECT
  'llm' as method,
  COALESCE(gemini_percentage, groq_percentage) as predicted,
  CASE WHEN human_verified_invalid THEN 'invalid' ELSE human_verified_percentage END as actual,
  COUNT(*) as count
FROM comparison_records
WHERE (human_verified_percentage IS NOT NULL OR human_verified_invalid = 1)
  AND COALESCE(gemini_percentage, groq_percentage) != human_verified_percentage
GROUP BY predicted, actual
ORDER BY count DESC;
```
</implementation>

<output>
Modify:
- `comparison_storage.py` - Add accuracy methods, invalid handling
- `main.py` - Update reclassify endpoint, extend stats endpoint, enhance UI
- Possibly `template_classifier.py` - If needed for timestamp extraction

No new files needed.
</output>

<verification>
1. **Ground truth linking works:**
   - Reclassify a training image
   - Check that matching comparison record was updated
   - Verify timestamp matching logic (±2 second tolerance)

2. **Accuracy metrics calculate correctly:**
   - With some human-verified records, `/comparisons/stats` shows accuracy
   - LLM accuracy and Template accuracy both calculated
   - Error patterns list populated

3. **Invalid handling works:**
   - Mark a training image as "invalid"
   - Comparison record shows `human_verified_invalid = TRUE`
   - If LLM said 100% and human said invalid → counted as LLM error

4. **UI shows accuracy:**
   - Summary cards show LLM/Template accuracy rates
   - Error patterns section visible with data
   - Records list shows who was right/wrong when verified
</verification>

<success_criteria>
- Human verification from training flows to comparison records
- Per-method accuracy rates clearly displayed
- Error patterns identify systematic failures (e.g., "LLM: invalid→100%")
- UI clearly shows who was right when ground truth available
- Invalid category properly handled throughout
- Existing functionality unchanged
</success_criteria>
