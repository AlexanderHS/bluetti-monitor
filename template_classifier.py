"""
Tesseract OCR Classifier for Battery Percentage Recognition

This module provides a local OCR-based classification system for battery
percentage recognition (0-100%) using Tesseract with optimized preprocessing.

Achieves ~96.6% accuracy on validated test data (vs 53.8% for template matching).

Two modes:
1. Collection mode: Always active - saves labeled images from Gemini for accuracy monitoring
2. Comparison mode: Activates at 25% coverage - logs OCR predictions alongside Gemini
"""

import os
import re
import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import json
import pytesseract

logger = logging.getLogger(__name__)


class TemplateClassifier:
    """Tesseract OCR-based battery percentage classifier

    Uses Tesseract OCR with optimized preprocessing (Otsu threshold + invert + padding)
    to extract battery percentages from LCD screen images. Achieves ~96.6% accuracy.
    """

    # Black screen detection thresholds
    BLACK_SCREEN_BRIGHTNESS_THRESHOLD = 50  # Mean brightness below this = black
    BLACK_SCREEN_VARIANCE_THRESHOLD = 20    # Std dev below this = uniform (no content)

    # Tesseract configuration (PSM 6 = uniform text block, digits only)
    TESSERACT_CONFIG = '--psm 6 -c tessedit_char_whitelist=0123456789'

    def __init__(self, training_data_dir: str = "./training_data"):
        """
        Initialize OCR classifier

        Args:
            training_data_dir: Directory to store training images (default: ./training_data)
        """
        self.training_data_dir = Path(training_data_dir)

        # Read configuration from environment
        self.max_images_per_percentage = int(os.getenv("MAX_IMAGES_PER_CATEGORY", "10"))
        self.comparison_threshold = float(os.getenv("COMPARISON_MODE_THRESHOLD", "0.25"))
        self.collection_enabled = True  # Collection always active by default

        # Initialize directory structure
        self._init_directories()

        logger.info(f"TemplateClassifier (OCR mode) initialized: {self.get_coverage_stats()}")

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

    def _is_black_screen(self, img: np.ndarray) -> bool:
        """
        Detect if an image is a black/dark screen (display off or failed capture)

        Args:
            img: Grayscale image as numpy array

        Returns:
            True if image appears to be a black screen
        """
        mean_brightness = np.mean(img)
        std_dev = np.std(img)

        is_black = (mean_brightness < self.BLACK_SCREEN_BRIGHTNESS_THRESHOLD and
                    std_dev < self.BLACK_SCREEN_VARIANCE_THRESHOLD)

        if is_black:
            logger.debug(f"Black screen detected: brightness={mean_brightness:.1f}, variance={std_dev:.1f}")

        return is_black

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for Tesseract OCR

        Applies optimized preprocessing pipeline validated on 199 samples:
        1. Otsu threshold - automatic binarization
        2. Invert - black digits on white background (Tesseract preference)
        3. 5px white padding - improves edge digit recognition

        Args:
            img: Grayscale image as numpy array

        Returns:
            Preprocessed image ready for OCR
        """
        # Otsu's thresholding (automatic binarization)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert: white background, black digits (Tesseract preference)
        inverted = cv2.bitwise_not(binary)

        # Add 5px white padding (improves edge digit recognition)
        padded = cv2.copyMakeBorder(
            inverted, 5, 5, 5, 5,
            cv2.BORDER_CONSTANT,
            value=255
        )

        return padded

    def is_verified(self, filename: str) -> bool:
        """
        Check if a filename indicates a verified image.

        Args:
            filename: Filename to check (e.g., "image.jpg" or "image.verified.jpg")

        Returns:
            True if filename contains ".verified." suffix
        """
        return ".verified." in filename

    def mark_verified(self, filepath: Path) -> bool:
        """
        Mark an image file as verified by adding .verified. suffix.

        Args:
            filepath: Path to the image file

        Returns:
            True if successfully renamed, False otherwise
        """
        if not filepath.exists():
            logger.warning(f"Cannot mark as verified - file not found: {filepath}")
            return False

        if self.is_verified(filepath.name):
            logger.debug(f"File already verified: {filepath.name}")
            return True

        # Insert ".verified" before the extension
        # e.g., "20231201_123456_789012.jpg" -> "20231201_123456_789012.verified.jpg"
        stem = filepath.stem
        suffix = filepath.suffix
        new_name = f"{stem}.verified{suffix}"
        new_path = filepath.parent / new_name

        try:
            filepath.rename(new_path)
            logger.info(f"Marked as verified: {filepath.name} -> {new_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark as verified: {e}")
            return False

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

            # Generate timestamp-based filename (new images are always unverified)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path = category_dir / f"{timestamp}.jpg"

            # Check if we need to delete oldest image (Smart FIFO rotation)
            existing_images = sorted(category_dir.glob("*.jpg"))
            if len(existing_images) >= self.max_images_per_percentage:
                # Smart FIFO: prefer deleting unverified images
                unverified_images = [img for img in existing_images if not self.is_verified(img.name)]
                verified_images = [img for img in existing_images if self.is_verified(img.name)]

                num_to_delete = len(existing_images) - self.max_images_per_percentage + 1

                if unverified_images:
                    # Delete oldest unverified image(s) first
                    for old_image in unverified_images[:num_to_delete]:
                        old_image.unlink()
                        logger.debug(f"FIFO rotation: deleted unverified {old_image.name}")
                elif verified_images:
                    # Only delete verified images if ALL images are verified
                    logger.info(f"All images in {category} are verified, deleting oldest verified image")
                    for old_image in verified_images[:num_to_delete]:
                        old_image.unlink()
                        logger.debug(f"FIFO rotation: deleted verified {old_image.name}")

            # Save new image
            with open(image_path, 'wb') as f:
                f.write(image_data)

            # Log progress periodically (every 10 images)
            stats = self.get_coverage_stats()
            if stats["total_images"] % 10 == 0:
                logger.info(f"Collection progress: {stats['values_with_examples']}/101 values, {stats['total_images']} total images ({stats['coverage_percentage']:.1f}% coverage)")

            return True

        except Exception as e:
            logger.error(f"Failed to save labeled image for {category}: {e}")
            return False

    def _extract_percentage(self, preprocessed_img: np.ndarray) -> Optional[int]:
        """
        Extract battery percentage from preprocessed image using Tesseract OCR

        Args:
            preprocessed_img: Preprocessed grayscale image (binary, inverted, padded)

        Returns:
            Integer 0-100 if successful, None if OCR fails or returns invalid value
        """
        try:
            text = pytesseract.image_to_string(preprocessed_img, config=self.TESSERACT_CONFIG)
            text = text.strip()

            # Extract digits with regex
            match = re.search(r'\b(\d{1,3})\b', text)
            if match:
                num = int(match.group(1))
                if 0 <= num <= 100:
                    return num

            # Fallback: concatenate all digits found
            digits = re.findall(r'\d', text)
            if digits:
                num = int(''.join(digits[:3]))
                if 0 <= num <= 100:
                    return num

            return None

        except Exception as e:
            logger.debug(f"OCR extraction failed: {e}")
            return None

    def classify_image(self, image_data: bytes) -> Optional[Dict]:
        """
        Classify an image using Tesseract OCR

        Args:
            image_data: Raw image bytes (JPEG)

        Returns:
            Dictionary with classification results, or None if image is invalid
            (black screen or OCR fails to extract valid percentage)
            None indicates the calling code should skip this cycle silently
        """
        try:
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            raw_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if raw_image is None:
                return {
                    "success": False,
                    "error": "Failed to decode image",
                    "percentage": None,
                    "confidence": 0.0
                }

            # Check for black screen BEFORE preprocessing
            # (preprocessing would distort the brightness/variance check)
            if self._is_black_screen(raw_image):
                logger.info("Black screen detected, skipping cycle")
                return None  # Signal to skip this cycle

            # Apply OCR preprocessing
            preprocessed = self._preprocess_image(raw_image)

            # Run OCR to extract percentage
            percentage = self._extract_percentage(preprocessed)

            if percentage is None:
                # OCR failed to extract valid number - treat as invalid
                logger.debug("OCR failed to extract valid percentage")
                return None

            return {
                "success": True,
                "percentage": percentage,
                "confidence": 1.0,  # OCR doesn't provide confidence; use 1.0 for success
                "method": "tesseract_ocr"
            }

        except Exception as e:
            logger.error(f"OCR classification failed: {e}")
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

    def get_preprocessed_image(self, image_data: bytes) -> Optional[bytes]:
        """
        Apply preprocessing to an image and return as JPEG bytes.
        Used by UI to show "classifier view" of images.

        Args:
            image_data: Raw image bytes (JPEG)

        Returns:
            Preprocessed image as JPEG bytes, or None on error
        """
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            preprocessed = self._preprocess_image(img)
            _, encoded = cv2.imencode('.jpg', preprocessed, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return encoded.tobytes()
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            return None


def extract_timestamp_from_filename(filename: str) -> Optional[float]:
    """
    Extract Unix timestamp from training image filename

    Args:
        filename: Training image filename (e.g., "20231201_123456_789012.jpg" or "20231201_123456_789012.verified.jpg")

    Returns:
        Unix timestamp as float, or None if unable to parse
    """
    try:
        # Remove .verified. suffix if present
        name = filename.replace(".verified.", ".")

        # Remove extension
        stem = name.rsplit(".", 1)[0]

        # Parse timestamp: YYYYMMDD_HHMMSS_ffffff
        dt = datetime.strptime(stem, "%Y%m%d_%H%M%S_%f")
        return dt.timestamp()

    except Exception as e:
        logger.warning(f"Failed to extract timestamp from filename {filename}: {e}")
        return None


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


def log_comparison(gemini_percentage: int, ocr_result: Optional[Dict]):
    """
    Log comparison between Gemini and OCR results

    Args:
        gemini_percentage: Percentage from Gemini API
        ocr_result: Result dictionary from OCR, or None if invalid/failed
    """
    if ocr_result is None:
        logger.info(f"[COMPARE] Gemini: {gemini_percentage}% | OCR: INVALID (failed to extract percentage)")
        return

    if not ocr_result["success"]:
        logger.info(f"[COMPARE] Gemini: {gemini_percentage}% | OCR: FAILED ({ocr_result.get('error', 'unknown error')})")
        return

    ocr_percentage = ocr_result["percentage"]
    ocr_confidence = ocr_result["confidence"]
    match = "YES" if gemini_percentage == ocr_percentage else "NO"

    logger.info(
        f"[COMPARE] Gemini: {gemini_percentage}% | OCR: {ocr_percentage}% "
        f"(confidence: {ocr_confidence:.3f}) | Match: {match}"
    )
