<objective>
Use the query_metrics.py script to analyze recent battery readings and identify false readings or OCR errors. The user has observed a pattern where readings went 74→73→72→71→7→69, suggesting "70" is being misread as "7".

This analysis will help identify:
1. How often false readings occur
2. Which values are most commonly misread
3. What might be causing the errors
4. Potential fixes to improve OCR accuracy
</objective>

<context>
Local repo: /Users/alexhamiltonsmith/repos/bluetti-monitor
Query script: ./query_metrics.sh (wrapper script created by previous prompt)

The bluetti-monitor system uses OCR to read battery percentage from an LCD screen. The OCR can fail in various ways:
- Missing digits (70 → 7)
- Misread digits (8 → 0, 6 → 0, etc.)
- Low confidence readings

Related code files:
- template_classifier.py - OCR implementation
- worker.py - Main loop that stores readings
</context>

<analysis_requirements>
1. Use the query wrapper script to get recent readings:
   ```bash
   ./query_metrics.sh readings --count 50
   ./query_metrics.sh anomalies --hours 24
   ```

2. Analyze the data for:
   - Sudden jumps in battery percentage (especially drops > 10%)
   - Single or double digit readings that break sequential patterns
   - Correlation between OCR confidence and false readings
   - Which specific values seem to be misread (e.g., 70→7, 80→8)

3. Review the OCR code to understand:
   - How confidence is calculated
   - What thresholds are used
   - How the template matching works

4. Propose solutions based on findings:
   - Could confidence thresholds be adjusted?
   - Should certain readings be rejected?
   - Would smoothing/filtering help?
   - Are there patterns that could be caught programmatically?
</analysis_requirements>

<output_format>
Provide a summary report including:

1. **Data Overview**: How many readings analyzed, time range covered
2. **Anomalies Found**: List of suspicious readings with timestamps
3. **Pattern Analysis**: Which values are commonly misread
4. **Root Cause Hypothesis**: Why these errors might be happening
5. **Recommendations**: Specific, actionable fixes ranked by impact/effort

If code changes are recommended, describe them but don't implement yet - let the user decide.
</output_format>

<verification>
Before completing:
- Confirm ./query_metrics.sh worked and returned data
- Ensure analysis is based on actual data, not assumptions
- Recommendations should be specific and actionable
</verification>

<success_criteria>
- Clear identification of false reading patterns
- Data-backed analysis (not speculation)
- Actionable recommendations for improving OCR accuracy
</success_criteria>
