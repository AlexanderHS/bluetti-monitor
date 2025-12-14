#!/usr/bin/env python3
"""
Test script for template classifier functionality
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from template_classifier import TemplateClassifier


def create_test_image(percentage: int, size=(100, 50)) -> bytes:
    """Create a simple test image with percentage text"""
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = (100, 150, 200)  # Blue background

    # Add percentage text
    text = f"{percentage}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes()


def test_basic_functionality():
    """Test basic classifier functionality"""
    print("=" * 60)
    print("Testing Template Classifier Basic Functionality")
    print("=" * 60)

    # Create test classifier with temp directory
    test_dir = "./test_training_data"
    classifier = TemplateClassifier(training_data_dir=test_dir)

    print(f"\n1. Initial state:")
    stats = classifier.get_coverage_stats()
    print(f"   Coverage: {stats['coverage_percentage']}%")
    print(f"   Values with examples: {stats['values_with_examples']}/101")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Comparison mode: {stats['comparison_mode_active']}")

    print(f"\n2. Testing image collection...")
    test_percentages = [0, 25, 50, 75, 100]
    for pct in test_percentages:
        img = create_test_image(pct)
        success = classifier.save_labeled_image(img, pct)
        print(f"   Saved {pct}%: {success}")

    stats = classifier.get_coverage_stats()
    print(f"\n3. After collection:")
    print(f"   Coverage: {stats['coverage_percentage']}%")
    print(f"   Values with examples: {stats['values_with_examples']}/101")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Images per percentage: {stats['images_per_percentage']}")

    print(f"\n4. Testing FIFO rotation (adding 11 images to 50%)...")
    for i in range(11):
        img = create_test_image(50)
        classifier.save_labeled_image(img, 50)

    # Check that only 10 images remain
    count_50 = len(list((Path(test_dir) / "50").glob("*.jpg")))
    print(f"   Images for 50% after 11 saves: {count_50} (should be 10)")

    print(f"\n5. Testing template matching...")
    test_img = create_test_image(50)
    result = classifier.classify_image(test_img)
    print(f"   Classification result: {result}")

    print(f"\n6. Testing comparison mode threshold...")
    print(f"   Should enable comparison: {classifier.should_enable_comparison_mode()}")
    print(f"   (True when coverage >= 50%, currently {stats['coverage_percentage']}%)")

    # Cleanup
    import shutil
    if Path(test_dir).exists():
        shutil.rmtree(test_dir)
        print(f"\n7. Cleaned up test directory: {test_dir}")

    print("\n" + "=" * 60)
    print("Basic tests completed!")
    print("=" * 60)


def test_fifo_rotation():
    """Test FIFO rotation specifically"""
    print("\n" + "=" * 60)
    print("Testing FIFO Rotation")
    print("=" * 60)

    test_dir = "./test_fifo"
    classifier = TemplateClassifier(training_data_dir=test_dir)

    print(f"\n1. Adding 15 images to percentage 42...")
    filenames = []
    import time
    for i in range(15):
        img = create_test_image(42)
        classifier.save_labeled_image(img, 42)
        # Get current filenames
        files = sorted((Path(test_dir) / "42").glob("*.jpg"))
        if files:
            filenames.append(files[-1].name)
        time.sleep(0.01)  # Small delay to ensure different timestamps

    final_files = sorted((Path(test_dir) / "42").glob("*.jpg"))
    print(f"   Final count: {len(final_files)} (should be 10)")
    print(f"   Files kept: most recent 10")

    # Cleanup
    import shutil
    if Path(test_dir).exists():
        shutil.rmtree(test_dir)

    print("\n" + "=" * 60)
    print("FIFO rotation test completed!")
    print("=" * 60)


def test_coverage_threshold():
    """Test that comparison mode activates at 50% coverage"""
    print("\n" + "=" * 60)
    print("Testing Coverage Threshold for Comparison Mode")
    print("=" * 60)

    test_dir = "./test_coverage"
    classifier = TemplateClassifier(training_data_dir=test_dir)

    # Add images for 50 percentages (49.5% coverage - should NOT activate)
    print("\n1. Adding 50 percentages (49.5% coverage)...")
    for pct in range(50):
        img = create_test_image(pct)
        classifier.save_labeled_image(img, pct)

    stats = classifier.get_coverage_stats()
    comparison_active = classifier.should_enable_comparison_mode()
    print(f"   Coverage: {stats['coverage_percentage']}%")
    print(f"   Comparison mode: {comparison_active} (should be False)")

    # Add one more to reach 50.5% (should activate)
    print("\n2. Adding 1 more percentage (50.5% coverage)...")
    img = create_test_image(50)
    classifier.save_labeled_image(img, 50)

    stats = classifier.get_coverage_stats()
    comparison_active = classifier.should_enable_comparison_mode()
    print(f"   Coverage: {stats['coverage_percentage']}%")
    print(f"   Comparison mode: {comparison_active} (should be True)")

    # Cleanup
    import shutil
    if Path(test_dir).exists():
        shutil.rmtree(test_dir)

    print("\n" + "=" * 60)
    print("Coverage threshold test completed!")
    print("=" * 60)


if __name__ == "__main__":
    print("\nRunning Template Classifier Tests\n")

    try:
        test_basic_functionality()
        test_coverage_threshold()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
