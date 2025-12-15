# Template Classifier Analysis

A deep-dive into the current template matching implementation, its suitability for battery percentage recognition, environmental challenges, alternatives, and architectural recommendations.

---

## 1. Current Implementation Overview

### Algorithm: Normalized Cross-Correlation (NCC)

The classifier uses OpenCV's `TM_CCORR_NORMED` template matching:

```python
correlation = cv2.matchTemplate(img1_norm, img2_norm, cv2.TM_CCORR_NORMED)
```

**How it works:**
- Computes normalized cross-correlation between test image and each template
- Returns similarity score 0.0-1.0 (higher = more similar)
- For each category (0-100%), averages similarity across all templates in that category
- Selects category with highest average similarity

**Is this k-NN with k=1?**

Not exactly. It's more accurately described as:
- **k=ALL within category**: Averages across all templates per category (up to 10)
- **k=1 across categories**: Takes single best-matching category

This is closer to a **nearest centroid classifier** where each category's "centroid" is the average similarity across its exemplars, rather than true k-NN which would consider the k nearest individual exemplars regardless of category.

### Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max images per category | 10 | FIFO rotation |
| Comparison threshold | 25% | Coverage needed to enable comparison mode |
| Image format | Grayscale JPEG | No color information used |
| Confidence metric | Average NCC similarity | 0.0-1.0 scale |

---

## 2. Suitability Assessment

### Strengths

1. **Simplicity**: Easy to understand, debug, and maintain
2. **No training phase**: Works immediately with labeled data
3. **Interpretable**: Can inspect templates directly, understand why classification happened
4. **Low memory**: Templates are just images, no model weights
5. **Deterministic**: Same input always produces same output

### Weaknesses

1. **Pixel-level sensitivity**: Small shifts cause significant score drops
2. **No abstraction**: Learns exact pixel patterns, not "the concept of digits"
3. **Linear scaling**: Computation grows with number of templates
4. **No uncertainty quantification**: Always returns a "best match" even if confidence is low
5. **Vulnerable to systematic bias**: If all training examples have same lighting, model fails under different lighting

### Is it well-suited to this scenario?

**Partially yes, partially no.**

**Why it can work:**
- LCD displays are highly consistent (same font, same position, same colors)
- Camera is fixed (same angle, same distance)
- Display is backlit (somewhat resistant to ambient lighting)
- Only 101 classes (0-100) + invalid - manageable scale

**Why it struggles:**
- The "21" vs "27" problem: visually similar digits
- Real-world variance: lighting, camera bumps, LCD artifacts
- No mechanism to say "I don't know"
- Doesn't leverage digit structure (treating "47" as unrelated to "4" or "7")

---

## 3. Does 10 Examples Per Category Work Well?

### Theoretical Analysis

With 10 examples per category:
- **Total training images**: Up to 1,010 (101 categories × 10)
- **Templates checked per classification**: All loaded templates
- **Variance captured**: Limited - 10 examples can't capture full distribution of lighting/position variance

### Empirical Considerations

**10 is likely insufficient for robust classification because:**

1. **Lighting variance**: Throughout a day, ambient light changes significantly. 10 images captured over a few hours may all have similar lighting.

2. **Temporal clustering**: FIFO means recent images dominate. If battery stays at 50% for hours, all 10 images are from similar conditions.

3. **Edge cases uncovered**: Rare conditions (very bright sun, night, display glitch) may never be in training set.

**Recommendations:**
- Increase to 20-30 per category, OR
- Implement intelligent sampling (keep diverse examples, not just recent)
- Track capture conditions (time of day, brightness) to ensure diversity

---

## 4. Computational Efficiency

### Current Complexity

```
Per classification:
  For each category (101 + invalid ≈ 102):
    For each template (up to 10):
      cv2.matchTemplate() - O(N²) for N×N images

Total: O(102 × 10 × N²) = O(1020 × N²)
```

For a 100×100 pixel image: ~10 million operations per classification.

### Is this efficient?

**For this use case: Yes, acceptable.**
- Classification happens every 30 seconds
- Single-threaded is fine
- Raspberry Pi can handle this easily

**But scaling concerns exist:**
- More templates = slower
- Higher resolution = much slower (quadratic)
- Real-time applications would need optimization

### Optimization Options

1. **Reduce image resolution**: Downsample to 50×50 or smaller
2. **Pre-compute template features**: Store normalized templates
3. **Early termination**: Stop if confidence exceeds threshold
4. **Hierarchical matching**: First match tens digit, then ones digit
5. **GPU acceleration**: OpenCV CUDA support available

---

## 5. Environmental Factors

### Camera Physical Displacement

**Problem**: If camera gets bumped, entire image shifts left/right/up/down.

**Impact on NCC**:
- NCC is **NOT** translation invariant when comparing same-size images
- A 5-pixel shift can drop similarity from 0.95 to 0.70
- Current implementation resizes template to match test image, but doesn't handle translation

**Current behavior**: Will likely misclassify after camera bump until new templates collected.

**Solutions:**
1. **Multi-scale template matching**: Search for template at different positions
2. **Feature-based matching**: Use SIFT/ORB features instead of raw pixels
3. **Region of Interest (ROI) detection**: Find LCD display bounds first, then classify
4. **Data augmentation**: Generate translated versions of templates

### Ambient Lighting Changes

**Problem**: Sunlight through window changes overall brightness throughout day.

**Impact on NCC**:
- NCC is somewhat normalized for brightness (it's "normalized" cross-correlation)
- However, uneven lighting (shadows) still causes issues
- LCD backlight helps, but doesn't eliminate ambient light influence

**Current behavior**: May work reasonably due to normalization, but contrast changes can still affect results.

**Solutions:**
1. **Histogram equalization**: Normalize image contrast before matching
2. **Adaptive thresholding**: Extract digits as binary before comparison
3. **Local contrast normalization**: Normalize each region independently
4. **Time-of-day stratified templates**: Keep morning/afternoon/night examples

### LCD Display Artifacts

**Problem**: LCD can have:
- Ghosting (previous digits visible)
- Partial refresh (digits half-changed)
- Pixel defects
- Condensation/fog on camera lens

**Current behavior**: May classify incorrectly or match "invalid" template.

**Solutions:**
1. **Temporal filtering**: Require consistent readings before accepting
2. **Multiple capture voting**: Current Gemini approach does this
3. **Anomaly detection**: Detect unusual images before classification

---

## 6. Human Labeling Strategy Analysis

### Current Flow

1. Gemini classifies image → auto-labeled with Gemini's answer
2. Human reviews in `/training/review` UI
3. Human can reclassify or verify
4. Reclassification marks as "verified"

### Potential Flaws

**Flaw 1: Trusting the wrong teacher**
- Templates are labeled by Gemini
- If Gemini systematically misclassifies "21" as "27", template learns this error
- Human review is optional and may not catch all errors
- **Solution**: Require human verification for a percentage of each category

**Flaw 2: Verification doesn't mean correctness**
- "Verified" only means a human looked at it
- Human might verify incorrect labels (cognitive fatigue, mistakes)
- **Solution**: Add confidence display during verification, require active label selection

**Flaw 3: Selection bias in review**
- Humans tend to review disagreements
- Agreements are assumed correct but may have systematic errors
- **Solution**: Random sampling for review, not just disagreements

**Flaw 4: No ground truth validation**
- No mechanism to measure true accuracy
- Relying on agreement between two flawed systems
- **Solution**: Periodic manual audit of random samples with fresh eyes

**Flaw 5: Invalid category under-labeled**
- Black screens, glitches may be labeled as 100% by Gemini
- Only discovered when human notices
- **Solution**: Pre-filter suspected invalid images using heuristics (variance, brightness)

### Leveraging Human Labels Better

**Currently underutilized:**

1. **Verified vs unverified distinction**: Used for FIFO rotation priority, but not for confidence weighting
2. **Reclassification patterns**: Not analyzed to identify systematic Gemini errors
3. **Human verification rate**: Not tracked per category

**Recommendations:**
- Weight verified templates higher in classification
- Track which categories have most reclassifications (indicates Gemini weakness)
- Alert when a category has no verified examples

---

## 7. Alternative Approaches

### Option A: Digit-Level Template Matching

Instead of matching full display, match individual digits.

**Approach:**
1. Segment display into digit regions (tens, ones)
2. Build templates for digits 0-9 only (not 0-100)
3. Classify each digit independently
4. Combine: tens × 10 + ones = percentage

**Pros:**
- Only 10 templates needed (not 101)
- More training data per class
- "21" vs "27" becomes "1" vs "7" (still hard, but more data)

**Cons:**
- Requires reliable segmentation
- Digit position must be consistent
- Extra complexity

### Option B: Feature-Based Matching (SIFT/ORB)

Use keypoint detectors instead of raw pixels.

**Pros:**
- Translation and rotation invariant
- More robust to lighting changes
- Can handle partial occlusion

**Cons:**
- LCD digits may not have enough features (smooth surfaces)
- More complex implementation
- May be overkill for this constrained scenario

### Option C: Simple Neural Network

Small CNN or even MLP trained on the images.

**Pros:**
- Can learn abstract features ("digit shapes")
- Potentially more robust to variations
- Can output uncertainty/confidence

**Cons:**
- Requires more training data
- Training infrastructure needed
- Less interpretable
- Risk of overfitting

### Option D: OCR-Focused Approach (Tesseract)

Use traditional OCR with preprocessing.

**Pros:**
- Designed for text recognition
- Well-tested on digits
- No training needed

**Cons:**
- Tesseract struggles with LCD fonts
- May require significant preprocessing
- Less controllable

### Option E: Ensemble / Voting System

Combine multiple methods.

**Pros:**
- Redundancy improves reliability
- Can detect disagreements (uncertainty signal)
- Best of multiple worlds

**Cons:**
- More complexity
- Slower
- Need to handle disagreements

### Recommendation

**Short term**: Improve current template matching with preprocessing and confidence thresholds.

**Medium term**: Implement digit-level matching for 10x data efficiency.

**Long term**: Consider lightweight neural network if data volume justifies it.

---

## 8. Small Tweaks to Current Implementation

### Tweak 1: Add Histogram Equalization

```python
def preprocess_image(img):
    """Normalize lighting variations"""
    return cv2.equalizeHist(img)
```

**Impact**: Reduces lighting sensitivity, ~5% accuracy improvement expected.

### Tweak 2: Add Translation Search

```python
def _compute_similarity_with_translation(self, img1, img2, search_range=10):
    """Find best match within translation range"""
    best_score = 0
    for dx in range(-search_range, search_range + 1, 2):
        for dy in range(-search_range, search_range + 1, 2):
            shifted = np.roll(np.roll(img2, dx, axis=1), dy, axis=0)
            score = self._compute_similarity(img1, shifted)
            best_score = max(best_score, score)
    return best_score
```

**Impact**: Handles small camera displacements, ~10% robustness improvement.

### Tweak 3: Confidence Threshold for "Not Sure"

```python
def classify_image(self, image_data: bytes) -> Optional[Dict]:
    # ... existing code ...

    # Add uncertainty detection
    MIN_CONFIDENCE = 0.75
    MARGIN_THRESHOLD = 0.05

    if best_confidence < MIN_CONFIDENCE:
        return {
            "success": True,
            "percentage": None,
            "confidence": best_confidence,
            "status": "uncertain_low_confidence"
        }

    # Check margin between top 2
    scores = sorted(all_scores.values(), reverse=True)
    if len(scores) > 1 and (scores[0] - scores[1]) < MARGIN_THRESHOLD:
        return {
            "success": True,
            "percentage": None,
            "confidence": best_confidence,
            "status": "uncertain_close_match",
            "candidates": top_3
        }
```

**Impact**: Enables "I don't know" response, prevents confident wrong answers.

### Tweak 4: Weight Verified Templates Higher

```python
def classify_image(self, image_data: bytes) -> Optional[Dict]:
    for category, templates in self.templates.items():
        similarities = []
        for template, is_verified in templates:  # Store verification status
            similarity = self._compute_similarity(test_image, template)
            weight = 1.5 if is_verified else 1.0  # Verified templates count more
            similarities.append(similarity * weight)
```

**Impact**: Human knowledge contributes more to classification.

### Tweak 5: Per-Category Confidence Calibration

Track historical accuracy per category and adjust confidence:

```python
# If category 21 historically has 60% accuracy, multiply confidence by 0.6
calibrated_confidence = raw_confidence * category_accuracy[category]
```

**Impact**: Reflects true reliability, not just similarity score.

---

## 9. Architectural Recommendations

### Goal: Prefer "Not Sure" Over Confidently Wrong

**Current problem**: System always returns best match, even if terrible.

**Proposed architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      Classification Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Pre-check  │───▶│   Template   │───▶│  Confidence  │       │
│  │  (invalid?)  │    │   Matching   │    │   Gating     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│    [INVALID]           [candidate]        [UNCERTAIN]           │
│    skip cycle          percentage         defer to LLM          │
│                                           or skip                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key changes:**

1. **Pre-check for invalid**: Before full classification, check for black screens, extreme brightness, low variance (solid color)

2. **Confidence gating**: Return "uncertain" if:
   - Best confidence < 0.75
   - Margin between top 2 < 0.05
   - Top match is historically unreliable category

3. **Fallback strategy**: When uncertain:
   - Use LLM (current behavior)
   - Or skip cycle entirely (conservative)
   - Or return last known good value (temporal smoothing)

### Goal: Minimize Long-Term Cost

**Cost factors:**
| Method | Per-query cost | Power | Reliability |
|--------|---------------|-------|-------------|
| Template (local) | $0 | ~0.5W | Moderate |
| Gemini API | ~$0.0001 | ~0.1W | High |
| Groq API | ~$0.00005 | ~0.1W | High |

**For 30-second intervals**: 2,880 queries/day
- Template only: $0/day, ~1.4 kWh/day
- LLM only: $0.29-$0.14/day, minimal local power

**Optimal strategy:**
1. Use template as primary when confident (>85% of queries)
2. Fallback to LLM only for uncertain cases (15% of queries)
3. Cost: ~$0.04-0.02/day + local power

**Architecture for cost optimization:**

```python
def classify_with_fallback(image_data):
    # Try template first (free)
    template_result = template_classifier.classify(image_data)

    if template_result.is_confident():
        return template_result

    # Uncertain - use LLM
    llm_result = gemini_ocr.classify(image_data)

    # Use LLM result for this query AND to improve template
    template_classifier.save_labeled_image(image_data, llm_result.percentage)

    return llm_result
```

### Goal: Maximize Accuracy

**Recommended architecture:**

```
┌────────────────────────────────────────────────────────────────────┐
│                    Multi-Strategy Classifier                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                    │
│  │  Template  │  │   Digit    │  │    LLM     │                    │
│  │  Matching  │  │  Segment   │  │  (Gemini)  │                    │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                    │
│        │               │               │                            │
│        ▼               ▼               ▼                            │
│  ┌─────────────────────────────────────────────────────┐           │
│  │               Voting / Arbitration                    │           │
│  │                                                       │           │
│  │  - If all agree → high confidence                    │           │
│  │  - If 2/3 agree → medium confidence                  │           │
│  │  - If all disagree → "uncertain"                     │           │
│  └─────────────────────────────────────────────────────┘           │
│                           │                                         │
│                           ▼                                         │
│                    [Final Result]                                   │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 10. Summary of Recommendations

### Immediate (Low Effort, High Impact)

1. **Add confidence thresholds** - return "uncertain" when confidence < 0.75 or margin < 0.05
2. **Add histogram equalization** - normalize lighting in preprocessing
3. **Pre-check for invalid** - detect black screens before full classification
4. **Track calibrated accuracy** - adjust confidence by historical category accuracy

### Short-Term (Medium Effort)

5. **Implement translation search** - handle small camera displacements
6. **Weight verified templates** - leverage human verification
7. **Stratified template storage** - keep diverse examples (morning/afternoon/night)
8. **Increase to 20 templates per category** - more variance coverage

### Medium-Term (Higher Effort)

9. **Digit-level segmentation** - match individual digits for 10x data efficiency
10. **Ensemble voting** - combine template + LLM with confidence weighting
11. **Automatic LLM fallback** - use template when confident, LLM when uncertain

### Long-Term (Requires Evaluation)

12. **Lightweight CNN** - if data volume justifies, train small neural network
13. **Active learning** - automatically request human review for uncertain cases
14. **Continuous calibration** - adjust thresholds based on rolling accuracy metrics

---

## Appendix: Quick Reference

### Current Algorithm Pseudocode

```
function classify(test_image):
    best_category = None
    best_score = 0

    for category in [0..100, "invalid"]:
        scores = []
        for template in templates[category]:
            score = NCC(test_image, template)
            scores.append(score)

        avg_score = mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_category = category

    return (best_category, best_score)
```

### Key Metrics to Track

| Metric | Purpose | Current Status |
|--------|---------|----------------|
| Agreement rate | LLM vs Template | ✓ Tracked |
| Per-category accuracy | Identify weak spots | ✓ Tracked |
| Confidence distribution | Calibration | ✗ Not tracked |
| Time-of-day accuracy | Environmental factors | ✗ Not tracked |
| Translation sensitivity | Camera stability | ✗ Not tracked |

### Environment Variables

```bash
MAX_IMAGES_PER_CATEGORY=10        # Templates per percentage
COMPARISON_MODE_THRESHOLD=0.25    # Coverage to enable comparison
TEMPLATE_CONFIDENCE_MIN=0.75      # Proposed: minimum confidence
TEMPLATE_MARGIN_MIN=0.05          # Proposed: minimum margin
```
