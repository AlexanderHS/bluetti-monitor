#!/usr/bin/env python3
"""
Demonstration of template matching classifier system

This script demonstrates the full workflow:
1. Collect labeled images (simulating Gemini classifications)
2. Show coverage progress
3. Demonstrate template matching classification
4. Show comparison mode activation at 50% coverage
"""

import sys
import cv2
import numpy as np
from template_classifier import template_classifier, log_comparison

def create_demo_image(percentage: int, variation: int = 0) -> bytes:
    """Create a demo image simulating battery LCD screen"""
    # Create blue background (similar to real LCD)
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    img[:] = (180 + variation, 150, 100)  # Blue LCD-like background

    # Add percentage text
    text = f"{percentage}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2

    # Center text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()


def demo_collection_phase():
    """Demonstrate image collection from Gemini classifications"""
    print("\n" + "=" * 70)
    print("PHASE 1: IMAGE COLLECTION (Simulating Gemini Classifications)")
    print("=" * 70)

    print("\n📥 Simulating Gemini classifying battery percentages...")
    print("   (In production, worker captures these automatically)")

    # Simulate collecting images for various percentages
    test_percentages = [
        20, 25, 30, 35, 40, 45, 50, 55, 60, 65,  # 10 values
        20, 25, 30, 35, 40, 45, 50, 55, 60, 65,  # 2nd images
        20, 25, 30, 35, 40, 45, 50, 55, 60, 65,  # 3rd images
    ]

    collected = 0
    for i, pct in enumerate(test_percentages):
        # Create image with slight variation
        img = create_demo_image(pct, variation=i % 5)
        success = template_classifier.save_labeled_image(img, pct)

        if success:
            collected += 1
            if (i + 1) % 10 == 0:
                stats = template_classifier.get_coverage_stats()
                print(f"   Progress: {collected} images → {stats['coverage_percentage']:.1f}% coverage")

    print(f"\n✅ Collection complete: {collected} images saved")

    stats = template_classifier.get_coverage_stats()
    print(f"   Coverage: {stats['coverage_percentage']:.1f}%")
    print(f"   Percentages trained: {stats['values_with_examples']}/101")


def demo_classification():
    """Demonstrate template matching classification"""
    print("\n" + "=" * 70)
    print("PHASE 2: TEMPLATE MATCHING CLASSIFICATION")
    print("=" * 70)

    print("\n🎯 Testing template matching on known values...")

    test_cases = [20, 35, 50, 65]  # Values we trained on
    for pct in test_cases:
        test_img = create_demo_image(pct)
        result = template_classifier.classify_image(test_img)

        if result['success']:
            match = "✅" if result['percentage'] == pct else "❌"
            print(f"   {match} Expected: {pct}% | Predicted: {result['percentage']}% | Confidence: {result['confidence']:.3f}")
        else:
            print(f"   ❌ Classification failed: {result.get('error')}")


def demo_comparison_mode():
    """Demonstrate comparison mode activation"""
    print("\n" + "=" * 70)
    print("PHASE 3: COMPARISON MODE (Gemini vs Template)")
    print("=" * 70)

    stats = template_classifier.get_coverage_stats()

    if stats['comparison_mode_active']:
        print(f"\n✅ Comparison mode is ACTIVE (coverage: {stats['coverage_percentage']:.1f}%)")
        print("   Template predictions will be logged alongside Gemini results\n")

        # Simulate comparison logging
        print("   Example comparison logs:")
        test_cases = [(50, 50), (35, 35), (65, 65), (45, 43)]  # (gemini, actual_shown)

        for gemini_pct, actual_pct in test_cases:
            img = create_demo_image(actual_pct)
            template_result = template_classifier.classify_image(img)

            # Simulate log output
            if template_result['success']:
                match = "YES" if gemini_pct == template_result['percentage'] else "NO"
                print(f"   [COMPARE] Gemini: {gemini_pct}% | Template: {template_result['percentage']}% "
                      f"(confidence: {template_result['confidence']:.3f}) | Match: {match}")
    else:
        print(f"\n⏸️  Comparison mode INACTIVE (coverage: {stats['coverage_percentage']:.1f}%)")
        threshold = template_classifier.comparison_threshold * 100
        needed = threshold - stats['coverage_percentage']
        print(f"   Need {needed:.1f}% more coverage to activate")
        print(f"   (Threshold: {threshold}%)")


def demo_fifo_rotation():
    """Demonstrate FIFO rotation"""
    print("\n" + "=" * 70)
    print("PHASE 4: FIFO ROTATION (Max 10 images per percentage)")
    print("=" * 70)

    print("\n🔄 Adding 15 images to percentage 42 to test rotation...")

    initial_stats = template_classifier.get_coverage_stats()
    initial_count = initial_stats.get('images_per_percentage', {}).get(42, 0)
    print(f"   Initial: {initial_count} images for 42%")

    # Add 15 images
    for i in range(15):
        img = create_demo_image(42, variation=i)
        template_classifier.save_labeled_image(img, 42)

    final_stats = template_classifier.get_coverage_stats()
    final_count = final_stats.get('images_per_percentage', {}).get(42, 0)

    print(f"   After adding 15: {final_count} images for 42%")

    if final_count == template_classifier.max_images_per_percentage:
        print(f"   ✅ FIFO rotation working - kept exactly {final_count} images")
    else:
        print(f"   ❌ FIFO rotation issue - expected {template_classifier.max_images_per_percentage}, got {final_count}")


def demo_api_endpoints():
    """Show available API endpoints"""
    print("\n" + "=" * 70)
    print("PHASE 5: API ENDPOINTS")
    print("=" * 70)

    print("\n📡 Available endpoints for monitoring and control:\n")

    endpoints = [
        ("GET", "/training/status", "Get coverage stats and training progress"),
        ("POST", "/training/enable?enabled=true", "Enable/disable image collection"),
        ("POST", "/training/label/50", "Manually label current image as 50%")
    ]

    for method, endpoint, description in endpoints:
        print(f"   {method:6} {endpoint:35} - {description}")

    print("\n💡 Example usage:")
    print("   curl http://localhost:8000/training/status")


def main():
    print("\n" + "=" * 70)
    print("TEMPLATE MATCHING CLASSIFIER - FULL SYSTEM DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how the system works in production:\n")
    print("1. Worker captures images after successful Gemini classifications")
    print("2. Images are automatically saved with labels (collection mode)")
    print("3. At 50% coverage, comparison logging activates automatically")
    print("4. Template matching runs alongside Gemini for accuracy observation")

    # Run demo phases
    demo_collection_phase()
    demo_classification()
    demo_comparison_mode()
    demo_fifo_rotation()
    demo_api_endpoints()

    # Final summary
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

    final_stats = template_classifier.get_coverage_stats()
    print(f"\nFinal state:")
    print(f"  - Coverage: {final_stats['coverage_percentage']:.1f}%")
    print(f"  - Values with data: {final_stats['values_with_examples']}/101")
    print(f"  - Total images: {final_stats['total_images']}")
    print(f"  - Comparison mode: {'ACTIVE ✅' if final_stats['comparison_mode_active'] else 'INACTIVE ⏸️'}")

    print("\n🚀 System is ready for production use!")
    print("   Start the worker to begin automatic training data collection.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
