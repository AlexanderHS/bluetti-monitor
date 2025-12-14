"""
Template Matching Classifier for Battery Percentage Recognition

This module provides a local image classification system that learns from
Gemini API classifications and builds a template-based classifier for
battery percentage recognition (0-100%).

Two modes:
1. Collection mode: Always active - saves labeled images from Gemini
2. Comparison mode: Activates at 50% coverage - logs predictions alongside Gemini
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class TemplateClassifier:
    """Template-based battery percentage classifier with automatic training"""

    def __init__(self, training_data_dir: str = "./training_data"):
        """
        Initialize template classifier

        Args:
            training_data_dir: Directory to store training images (default: ./training_data)
        """
        self.training_data_dir = Path(training_data_dir)
        self.max_images_per_percentage = 10  # FIFO rotation limit
        self.comparison_threshold = 0.50  # 50% coverage to enable comparison mode
        self.collection_enabled = True  # Collection always active by default

        # Template cache: {percentage: [template_images]}
        self.templates = {}

        # Initialize directory structure
        self._init_directories()

        # Load existing templates into memory
        self._load_templates()

        logger.info(f"TemplateClassifier initialized: {self.get_coverage_stats()}")

    def _init_directories(self):
        """Create training_data directory structure (0-100 subdirectories + invalid)"""
        self.training_data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each percentage value (0-100)
        for percentage in range(101):
            percentage_dir = self.training_data_dir / str(percentage)
            percentage_dir.mkdir(exist_ok=True)

        # Create invalid directory for garbled/unreadable images
        invalid_dir = self.training_data_dir / "invalid"
        invalid_dir.mkdir(exist_ok=True)

        logger.debug(f"Initialized {self.training_data_dir} with 0-100 subdirectories + invalid")

    def _load_templates(self):
        """Load all existing training images into memory as templates"""
        self.templates.clear()

        # Load percentage templates (0-100)
        for percentage in range(101):
            percentage_dir = self.training_data_dir / str(percentage)
            if not percentage_dir.exists():
                continue

            # Load all images for this percentage
            image_files = sorted(percentage_dir.glob("*.jpg"))
            if image_files:
                templates = []
                for img_file in image_files:
                    try:
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            templates.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load template {img_file}: {e}")

                if templates:
                    self.templates[percentage] = templates

        # Load invalid templates (special category)
        invalid_dir = self.training_data_dir / "invalid"
        if invalid_dir.exists():
            image_files = sorted(invalid_dir.glob("*.jpg"))
            if image_files:
                templates = []
                for img_file in image_files:
                    try:
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            templates.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load invalid template {img_file}: {e}")

                if templates:
                    self.templates["invalid"] = templates

        logger.debug(f"Loaded templates for {len(self.templates)} categories (including invalid if present)")

    def get_coverage_stats(self) -> Dict:
        """
        Calculate coverage statistics

        Returns:
            Dictionary with coverage metrics
        """
        total_values = 101  # 0-100 inclusive
        values_with_examples = 0
        total_images = 0
        images_per_percentage = {}

        for percentage in range(101):
            percentage_dir = self.training_data_dir / str(percentage)
            if percentage_dir.exists():
                image_count = len(list(percentage_dir.glob("*.jpg")))
                if image_count > 0:
                    values_with_examples += 1
                    total_images += image_count
                    images_per_percentage[percentage] = image_count

        coverage_percentage = (values_with_examples / total_values) * 100

        return {
            "total_values": total_values,
            "values_with_examples": values_with_examples,
            "coverage_percentage": round(coverage_percentage, 2),
            "total_images": total_images,
            "images_per_percentage": images_per_percentage,
            "comparison_mode_active": coverage_percentage >= (self.comparison_threshold * 100),
            "collection_enabled": self.collection_enabled
        }

    def save_labeled_image(self, image_data: bytes, category) -> bool:
        """
        Save a labeled image to training data with FIFO rotation

        Args:
            image_data: Raw image bytes (JPEG)
            category: Labeled category (0-100 or "invalid")

        Returns:
            True if saved successfully, False otherwise
        """
        if not self.collection_enabled:
            logger.debug("Collection disabled, skipping image save")
            return False

        # Validate category
        if category != "invalid" and not (isinstance(category, int) and 0 <= category <= 100):
            logger.warning(f"Invalid category {category}, must be 0-100 or 'invalid'")
            return False

        try:
            category_dir = self.training_data_dir / str(category)
            category_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path = category_dir / f"{timestamp}.jpg"

            # Check if we need to delete oldest image (FIFO rotation)
            existing_images = sorted(category_dir.glob("*.jpg"))
            if len(existing_images) >= self.max_images_per_percentage:
                # Delete oldest image(s) to make room
                num_to_delete = len(existing_images) - self.max_images_per_percentage + 1
                for old_image in existing_images[:num_to_delete]:
                    old_image.unlink()
                    logger.debug(f"FIFO rotation: deleted {old_image.name}")

            # Save new image
            with open(image_path, 'wb') as f:
                f.write(image_data)

            # Reload templates to include new image
            self._load_templates()

            # Log progress periodically (every 10 images)
            stats = self.get_coverage_stats()
            if stats["total_images"] % 10 == 0:
                logger.info(f"Collection progress: {stats['values_with_examples']}/101 values, {stats['total_images']} total images ({stats['coverage_percentage']:.1f}% coverage)")

            return True

        except Exception as e:
            logger.error(f"Failed to save labeled image for {category}: {e}")
            return False

    def _compute_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute similarity between two images using normalized cross-correlation

        Args:
            img1: First image (grayscale)
            img2: Second image (grayscale)

        Returns:
            Similarity score (0.0 to 1.0, higher is more similar)
        """
        try:
            # Resize images to same size if different
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # Normalize images to 0-1 range
            img1_norm = img1.astype(np.float32) / 255.0
            img2_norm = img2.astype(np.float32) / 255.0

            # Compute normalized cross-correlation
            correlation = cv2.matchTemplate(img1_norm, img2_norm, cv2.TM_CCORR_NORMED)
            similarity = correlation[0, 0]

            # Clamp to 0-1 range
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            logger.debug(f"Similarity computation failed: {e}")
            return 0.0

    def classify_image(self, image_data: bytes) -> Optional[Dict]:
        """
        Classify an image using template matching

        Args:
            image_data: Raw image bytes (JPEG)

        Returns:
            Dictionary with classification results, or None if image matches "invalid" template
            None indicates the calling code should skip this cycle silently
        """
        if not self.templates:
            return {
                "success": False,
                "error": "No templates available for classification",
                "percentage": None,
                "confidence": 0.0
            }

        try:
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            test_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if test_image is None:
                return {
                    "success": False,
                    "error": "Failed to decode image",
                    "percentage": None,
                    "confidence": 0.0
                }

            # Compare against all templates
            best_category = None
            best_confidence = 0.0
            all_scores = {}

            for category, templates in self.templates.items():
                # Compute average similarity across all templates for this category
                similarities = []
                for template in templates:
                    similarity = self._compute_similarity(test_image, template)
                    similarities.append(similarity)

                avg_similarity = np.mean(similarities) if similarities else 0.0
                all_scores[category] = round(avg_similarity, 4)

                if avg_similarity > best_confidence:
                    best_confidence = avg_similarity
                    best_category = category

            # Check if best match is "invalid" category
            if best_category == "invalid":
                logger.debug("Image matched invalid template, skipping cycle")
                return None  # Signal to skip this cycle

            # Get top 3 matches for debugging
            top_3 = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:3]

            return {
                "success": True,
                "percentage": best_category,
                "confidence": round(best_confidence, 4),
                "top_3_matches": dict(top_3),
                "total_templates_checked": len(self.templates)
            }

        except Exception as e:
            logger.error(f"Template classification failed: {e}")
            return {
                "success": False,
                "error": f"Classification exception: {str(e)}",
                "percentage": None,
                "confidence": 0.0
            }

    def should_enable_comparison_mode(self) -> bool:
        """Check if comparison mode should be enabled (>= 50% coverage)"""
        stats = self.get_coverage_stats()
        return stats["coverage_percentage"] >= (self.comparison_threshold * 100)

    def enable_collection(self, enabled: bool = True):
        """Enable or disable image collection"""
        self.collection_enabled = enabled
        logger.info(f"Collection {'enabled' if enabled else 'disabled'}")

    def manually_label_image(self, image_data: bytes, category) -> bool:
        """
        Manually label/relabel an image

        Args:
            image_data: Raw image bytes
            category: Correct category label (0-100 or "invalid")

        Returns:
            True if saved successfully
        """
        return self.save_labeled_image(image_data, category)


# Global instance
template_classifier = TemplateClassifier()


def get_training_status() -> Dict:
    """
    Get current training/collection status

    Returns:
        Dictionary with coverage stats and classifier info
    """
    stats = template_classifier.get_coverage_stats()
    stats["training_data_dir"] = str(template_classifier.training_data_dir)
    stats["max_images_per_percentage"] = template_classifier.max_images_per_percentage
    return stats


def log_comparison(gemini_percentage: int, template_result: Optional[Dict]):
    """
    Log comparison between Gemini and template matching results

    Args:
        gemini_percentage: Percentage from Gemini API
        template_result: Result dictionary from template matching, or None if invalid match
    """
    if template_result is None:
        logger.info(f"[COMPARE] Gemini: {gemini_percentage}% | Template: INVALID (image matched invalid template)")
        return

    if not template_result["success"]:
        logger.info(f"[COMPARE] Gemini: {gemini_percentage}% | Template: FAILED ({template_result.get('error', 'unknown error')})")
        return

    template_percentage = template_result["percentage"]
    template_confidence = template_result["confidence"]
    match = "YES" if gemini_percentage == template_percentage else "NO"

    logger.info(
        f"[COMPARE] Gemini: {gemini_percentage}% | Template: {template_percentage}% "
        f"(confidence: {template_confidence:.3f}) | Match: {match}"
    )
