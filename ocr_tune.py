#!/usr/bin/env python3
"""
OCR Tuning: Test different Tesseract configurations

Focus on the best preprocessing (threshold + invert) and tune:
- PSM modes (page segmentation)
- Padding around image
- OEM modes (OCR engine)
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
BEST_SO_FAR = 0.884  # v5_invert_threshold


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
    return records


def preprocess_base(img):
    """Best preprocessing: Otsu threshold + invert"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary)


def add_padding(img, padding, color=255):
    """Add padding around image"""
    return cv2.copyMakeBorder(img, padding, padding, padding, padding,
                              cv2.BORDER_CONSTANT, value=color)


def scale_image(img, scale):
    """Scale image by factor"""
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def extract_number(text):
    """Extract number from OCR output"""
    text = text.strip()

    # Try to find a number (1-3 digits)
    match = re.search(r'\b(\d{1,3})\b', text)
    if match:
        num = int(match.group(1))
        if 0 <= num <= 100:
            return num

    # Try without word boundaries
    digits = re.findall(r'\d', text)
    if digits:
        num = int(''.join(digits[:3]))
        if 0 <= num <= 100:
            return num

    return None


def run_ocr(img, config):
    """Run Tesseract OCR"""
    try:
        text = pytesseract.image_to_string(img, config=config)
        return extract_number(text)
    except Exception as e:
        return None


def evaluate_config(records, config_name, preprocess_fn, ocr_config):
    """Evaluate a configuration"""
    correct = 0
    total_valid = 0
    invalid_detected = 0
    total_invalid = 0
    errors = []

    for record in records:
        img = cv2.imread(str(record['path']))
        if img is None:
            continue

        processed = preprocess_fn(img)

        if record['is_invalid']:
            total_invalid += 1
            result = run_ocr(processed, ocr_config)
            if result is None:
                invalid_detected += 1
        else:
            total_valid += 1
            result = run_ocr(processed, ocr_config)

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
        'name': config_name,
        'accuracy': accuracy,
        'correct': correct,
        'total_valid': total_valid,
        'invalid_detection_rate': invalid_rate,
        'errors': errors
    }


def main():
    print("=" * 70)
    print("OCR Tuning: Tesseract Configuration Optimization")
    print("=" * 70)

    records = load_verified_data()
    valid_count = sum(1 for r in records if not r['is_invalid'])
    print(f"Loaded {len(records)} samples ({valid_count} valid)")

    # Base config with digit whitelist
    whitelist = '-c tessedit_char_whitelist=0123456789'

    # PSM modes to test
    # 6 = Assume uniform block of text
    # 7 = Treat as single text line (default we used)
    # 8 = Treat as single word
    # 10 = Treat as single character
    # 11 = Sparse text
    # 13 = Raw line
    psm_modes = [6, 7, 8, 10, 11, 13]

    # Padding values to test
    padding_values = [0, 5, 10, 15, 20]

    # Scale factors
    scale_factors = [1, 2, 3]

    results = []

    # Test 1: PSM modes with base preprocessing (no padding)
    print("\n" + "-" * 70)
    print("Test 1: PSM Modes (no padding)")
    print("-" * 70)

    for psm in psm_modes:
        config = f'--psm {psm} {whitelist}'
        name = f'psm_{psm}'
        print(f"Testing {name}...", end=" ", flush=True)

        result = evaluate_config(records, name, preprocess_base, config)
        results.append(result)
        print(f"Accuracy: {result['accuracy']*100:.1f}%")

    # Test 2: Padding with best PSM from Test 1
    best_psm_result = max(results, key=lambda x: x['accuracy'])
    best_psm = int(best_psm_result['name'].split('_')[1])
    print(f"\nBest PSM so far: {best_psm} ({best_psm_result['accuracy']*100:.1f}%)")

    print("\n" + "-" * 70)
    print(f"Test 2: Padding (with PSM {best_psm})")
    print("-" * 70)

    for padding in padding_values:
        if padding == 0:
            continue  # Already tested
        config = f'--psm {best_psm} {whitelist}'
        name = f'pad_{padding}'

        def preprocess_with_padding(img, p=padding):
            processed = preprocess_base(img)
            return add_padding(processed, p, color=255)

        print(f"Testing {name}...", end=" ", flush=True)
        result = evaluate_config(records, name, preprocess_with_padding, config)
        results.append(result)
        print(f"Accuracy: {result['accuracy']*100:.1f}%")

    # Test 3: Scale with best padding
    best_pad_results = [r for r in results if r['name'].startswith('pad_')]
    if best_pad_results:
        best_pad_result = max(best_pad_results, key=lambda x: x['accuracy'])
        best_pad = int(best_pad_result['name'].split('_')[1])
    else:
        best_pad = 0
        best_pad_result = best_psm_result

    print(f"\nBest padding so far: {best_pad} ({best_pad_result['accuracy']*100:.1f}%)")

    print("\n" + "-" * 70)
    print(f"Test 3: Scale factors (with PSM {best_psm}, padding {best_pad})")
    print("-" * 70)

    for scale in scale_factors:
        if scale == 1:
            continue  # Already tested
        config = f'--psm {best_psm} {whitelist}'
        name = f'scale_{scale}x'

        def preprocess_with_scale(img, s=scale, p=best_pad):
            processed = preprocess_base(img)
            if p > 0:
                processed = add_padding(processed, p, color=255)
            return scale_image(processed, s)

        print(f"Testing {name}...", end=" ", flush=True)
        result = evaluate_config(records, name, preprocess_with_scale, config)
        results.append(result)
        print(f"Accuracy: {result['accuracy']*100:.1f}%")

    # Test 4: OEM modes
    print("\n" + "-" * 70)
    print("Test 4: OEM Modes (OCR Engine)")
    print("-" * 70)

    # Find best scale
    scale_results = [r for r in results if r['name'].startswith('scale_')]
    if scale_results:
        best_scale_result = max(scale_results, key=lambda x: x['accuracy'])
        best_scale = int(best_scale_result['name'].split('_')[1].replace('x', ''))
    else:
        best_scale = 1

    # OEM modes:
    # 0 = Legacy engine only
    # 1 = Neural nets LSTM engine only
    # 2 = Legacy + LSTM
    # 3 = Default (based on what's available)
    oem_modes = [0, 1, 2, 3]

    for oem in oem_modes:
        config = f'--oem {oem} --psm {best_psm} {whitelist}'
        name = f'oem_{oem}'

        def preprocess_best(img, s=best_scale, p=best_pad):
            processed = preprocess_base(img)
            if p > 0:
                processed = add_padding(processed, p, color=255)
            if s > 1:
                processed = scale_image(processed, s)
            return processed

        print(f"Testing {name}...", end=" ", flush=True)
        try:
            result = evaluate_config(records, name, preprocess_best, config)
            results.append(result)
            print(f"Accuracy: {result['accuracy']*100:.1f}%")
        except Exception as e:
            print(f"Failed: {e}")

    # Final results
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    print("\n" + "=" * 70)
    print("ALL RESULTS (sorted by accuracy)")
    print("=" * 70)

    print(f"\n{'Config':<20} {'Accuracy':>10} {'Correct':>10} {'Invalid Det':>12}")
    print("-" * 55)

    for r in results:
        print(f"{r['name']:<20} {r['accuracy']*100:>9.1f}% {r['correct']:>7}/{r['total_valid']:<3} {r['invalid_detection_rate']*100:>10.1f}%")

    # Best result
    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST CONFIGURATION: {best['name']}")
    print(f"{'='*70}")
    print(f"Accuracy: {best['accuracy']*100:.1f}% ({best['correct']}/{best['total_valid']})")

    print(f"\n{'-'*40}")
    print("COMPARISON")
    print(f"{'-'*40}")
    print(f"Best tuned:    {best['accuracy']*100:.1f}%")
    print(f"Previous best: {BEST_SO_FAR*100:.1f}%")
    print(f"LLM:           {LLM_ACCURACY*100:.1f}%")
    print(f"Template:      {TEMPLATE_ACCURACY*100:.1f}%")

    improvement = best['accuracy'] - BEST_SO_FAR
    if improvement > 0:
        print(f"\n✓ Improved by {improvement*100:.1f}pp")
    elif improvement < 0:
        print(f"\n✗ Worse by {abs(improvement)*100:.1f}pp")
    else:
        print(f"\n= No change")

    # Error analysis for best
    if best['errors']:
        print(f"\nRemaining errors ({len(best['errors'])}):")
        error_types = defaultdict(list)
        for e in best['errors']:
            if e['got'] is None:
                error_types['failed_to_read'].append(e['expected'])
            elif e['got'] < 10 and e['expected'] >= 10:
                error_types['truncated'].append(f"{e['expected']}->{e['got']}")
            else:
                error_types['wrong_digit'].append(f"{e['expected']}->{e['got']}")

        for error_type, examples in error_types.items():
            print(f"  {error_type}: {len(examples)} - {examples[:5]}")


if __name__ == "__main__":
    main()
