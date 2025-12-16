#!/usr/bin/env python3
"""
OCR Test: Tesseract with various preprocessing strategies

Tests different preprocessing approaches on verified samples
to find the best combination for reading serif digits on blue background.
"""

import sqlite3
import numpy as np
from pathlib import Path
import cv2
import pytesseract
import re
from collections import defaultdict

# Configuration
DB_PATH = "./data/battery_readings.db"
IMAGES_DIR = "./data/comparison_images"

# Baselines
LLM_ACCURACY = 0.60
TEMPLATE_ACCURACY = 0.538


def load_verified_data():
    """Load verified samples from database"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT image_filename, human_verified_percentage, human_verified_invalid
        FROM comparison_records
        WHERE human_verified_percentage IS NOT NULL OR human_verified_invalid = 1
    """)

    records = []
    for filename, percentage, is_invalid in cur.fetchall():
        img_path = Path(IMAGES_DIR) / filename
        if img_path.exists():
            records.append({
                'filename': filename,
                'path': img_path,
                'label': -1 if is_invalid else int(percentage),
                'is_invalid': bool(is_invalid)
            })

    conn.close()
    print(f"Loaded {len(records)} verified samples")
    return records


def preprocess_v1_histogram(img):
    """Basic histogram equalization"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.equalizeHist(gray)


def preprocess_v2_clahe(img):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def preprocess_v3_threshold(img):
    """Otsu's thresholding for binarization"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_v4_adaptive(img):
    """Adaptive thresholding"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)


def preprocess_v5_invert_threshold(img):
    """Threshold then invert (black digits on white)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary)


def preprocess_v6_clahe_threshold(img):
    """CLAHE then Otsu threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def preprocess_v7_clahe_invert(img):
    """CLAHE, threshold, then invert"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary)


def preprocess_v8_morphology(img):
    """CLAHE + threshold + morphological cleanup"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological opening to remove noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(cleaned)


def preprocess_v9_scale_up(img):
    """Scale up 2x then CLAHE + threshold + invert"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # Scale up 2x
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(scaled)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary)


def preprocess_v10_blue_channel(img):
    """Extract blue channel (should have best contrast for white-on-blue)"""
    if len(img.shape) == 3:
        blue = img[:, :, 0]  # BGR format
    else:
        blue = img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(blue)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary)


PREPROCESSORS = {
    'v1_histogram': preprocess_v1_histogram,
    'v2_clahe': preprocess_v2_clahe,
    'v3_threshold': preprocess_v3_threshold,
    'v4_adaptive': preprocess_v4_adaptive,
    'v5_invert_threshold': preprocess_v5_invert_threshold,
    'v6_clahe_threshold': preprocess_v6_clahe_threshold,
    'v7_clahe_invert': preprocess_v7_clahe_invert,
    'v8_morphology': preprocess_v8_morphology,
    'v9_scale_up': preprocess_v9_scale_up,
    'v10_blue_channel': preprocess_v10_blue_channel,
}


def extract_number(text):
    """Extract number from OCR output"""
    # Remove whitespace and newlines
    text = text.strip()

    # Try to find a number (1-3 digits)
    match = re.search(r'\b(\d{1,3})\b', text)
    if match:
        num = int(match.group(1))
        if 0 <= num <= 100:
            return num

    # Try without word boundaries (sometimes OCR adds extra chars)
    digits = re.findall(r'\d', text)
    if digits:
        num = int(''.join(digits[:3]))  # Take first 3 digits max
        if 0 <= num <= 100:
            return num

    return None


def run_ocr(img, config='--psm 7 -c tessedit_char_whitelist=0123456789'):
    """Run Tesseract OCR with digit-only config"""
    try:
        text = pytesseract.image_to_string(img, config=config)
        return extract_number(text)
    except Exception as e:
        return None


def evaluate_preprocessor(records, preprocess_fn, name):
    """Evaluate a preprocessing strategy"""
    correct = 0
    total_valid = 0
    invalid_detected = 0
    total_invalid = 0
    errors = []

    for record in records:
        img = cv2.imread(str(record['path']))
        if img is None:
            continue

        if record['is_invalid']:
            total_invalid += 1
            # For invalid images, OCR should fail or return None
            processed = preprocess_fn(img)
            result = run_ocr(processed)
            if result is None:
                invalid_detected += 1
        else:
            total_valid += 1
            processed = preprocess_fn(img)
            result = run_ocr(processed)

            if result == record['label']:
                correct += 1
            else:
                errors.append({
                    'filename': record['filename'],
                    'expected': record['label'],
                    'got': result
                })

    accuracy = correct / total_valid if total_valid > 0 else 0
    invalid_rate = invalid_detected / total_invalid if total_invalid > 0 else 0

    return {
        'name': name,
        'accuracy': accuracy,
        'correct': correct,
        'total_valid': total_valid,
        'invalid_detection_rate': invalid_rate,
        'invalid_detected': invalid_detected,
        'total_invalid': total_invalid,
        'errors': errors[:10]  # Keep first 10 errors for analysis
    }


def main():
    print("=" * 60)
    print("OCR Test: Tesseract with Preprocessing Strategies")
    print("=" * 60)

    records = load_verified_data()

    valid_count = sum(1 for r in records if not r['is_invalid'])
    invalid_count = sum(1 for r in records if r['is_invalid'])
    print(f"  Valid samples: {valid_count}")
    print(f"  Invalid samples: {invalid_count}")

    print("\nTesting preprocessing strategies...")
    print("-" * 60)

    results = []
    for name, preprocess_fn in PREPROCESSORS.items():
        print(f"Testing {name}...", end=" ", flush=True)
        result = evaluate_preprocessor(records, preprocess_fn, name)
        results.append(result)
        print(f"Accuracy: {result['accuracy']*100:.1f}%")

    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    print("\n" + "=" * 60)
    print("RESULTS (sorted by accuracy)")
    print("=" * 60)

    print(f"\n{'Strategy':<25} {'Accuracy':>10} {'Correct':>10} {'Invalid Det':>12}")
    print("-" * 60)

    for r in results:
        print(f"{r['name']:<25} {r['accuracy']*100:>9.1f}% {r['correct']:>7}/{r['total_valid']:<3} {r['invalid_detection_rate']*100:>10.1f}%")

    # Best result details
    best = results[0]
    print(f"\n{'='*60}")
    print(f"BEST: {best['name']}")
    print(f"{'='*60}")
    print(f"Accuracy: {best['accuracy']*100:.1f}% ({best['correct']}/{best['total_valid']})")
    print(f"Invalid detection: {best['invalid_detection_rate']*100:.1f}% ({best['invalid_detected']}/{best['total_invalid']})")

    # Compare to baselines
    print(f"\n{'-'*40}")
    print("COMPARISON TO BASELINES")
    print(f"{'-'*40}")
    print(f"Best OCR:  {best['accuracy']*100:.1f}%")
    print(f"LLM:       {LLM_ACCURACY*100:.1f}%")
    print(f"Template:  {TEMPLATE_ACCURACY*100:.1f}%")

    if best['accuracy'] > LLM_ACCURACY:
        print(f"\n✓ BEATS LLM by {(best['accuracy'] - LLM_ACCURACY)*100:.1f}pp")
    else:
        print(f"\n✗ Below LLM by {(LLM_ACCURACY - best['accuracy'])*100:.1f}pp")

    if best['accuracy'] > TEMPLATE_ACCURACY:
        print(f"✓ BEATS Template by {(best['accuracy'] - TEMPLATE_ACCURACY)*100:.1f}pp")
    else:
        print(f"✗ Below Template by {(TEMPLATE_ACCURACY - best['accuracy'])*100:.1f}pp")

    # Show error patterns for best
    if best['errors']:
        print(f"\nTop errors for {best['name']}:")
        error_patterns = defaultdict(int)
        for e in best['errors']:
            pattern = f"{e['expected']} -> {e['got']}"
            error_patterns[pattern] += 1

        for pattern, count in sorted(error_patterns.items(), key=lambda x: -x[1])[:10]:
            print(f"  {pattern}: {count}")


if __name__ == "__main__":
    main()
