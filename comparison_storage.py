"""
Comparison Statistics Storage Module

Manages SQLite storage for LLM vs Template strategy comparison data with FIFO rotation.
Tracks agreement/disagreement patterns and stores images for review.
"""
import os
import logging
import aiosqlite
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class ComparisonStorage:
    """Manage comparison records with FIFO rotation and image storage"""

    def __init__(self, db_path: str = "./data/battery_readings.db"):
        """
        Initialize comparison storage

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.comparison_images_dir = Path("./data/comparison_images")
        self.max_records = int(os.getenv("MAX_COMPARISON_RECORDS", "1000"))

        # Ensure comparison images directory exists
        self.comparison_images_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ComparisonStorage initialized: max_records={self.max_records}")

    async def init_db(self):
        """Initialize database table for comparison records"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS comparison_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    gemini_percentage INTEGER,
                    groq_percentage INTEGER,
                    template_percentage INTEGER,
                    template_confidence REAL,
                    human_verified_percentage INTEGER,
                    human_verified_invalid BOOLEAN DEFAULT 0,
                    image_filename TEXT NOT NULL,
                    agreement BOOLEAN NOT NULL,
                    llm_source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_comparison_timestamp
                ON comparison_records(timestamp)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_comparison_agreement
                ON comparison_records(agreement)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_comparison_human_verified
                ON comparison_records(human_verified_percentage)
            """)

            # Migration: Add human_verified_invalid column if it doesn't exist
            try:
                await db.execute("SELECT human_verified_invalid FROM comparison_records LIMIT 1")
            except:
                logger.info("Migrating: adding human_verified_invalid column")
                await db.execute("""
                    ALTER TABLE comparison_records
                    ADD COLUMN human_verified_invalid BOOLEAN DEFAULT 0
                """)

            await db.commit()
            logger.info("Comparison records table initialized")

    async def save_comparison(
        self,
        image_data: bytes,
        gemini_percentage: Optional[int] = None,
        groq_percentage: Optional[int] = None,
        template_percentage: Optional[int] = None,
        template_confidence: Optional[float] = None,
        llm_source: str = "none"
    ) -> bool:
        """
        Save a comparison record with image and apply FIFO rotation

        Args:
            image_data: Raw image bytes (JPEG)
            gemini_percentage: Gemini result (None if failed)
            groq_percentage: Groq result (None if not used or failed)
            template_percentage: Template matching result (None if failed)
            template_confidence: Template confidence score
            llm_source: Which LLM was used: "gemini", "groq", or "none"

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            timestamp = datetime.now().timestamp()

            # Determine primary LLM result for agreement check
            llm_percentage = gemini_percentage if gemini_percentage is not None else groq_percentage

            # Check agreement (both strategies must have valid results)
            agreement = False
            if llm_percentage is not None and template_percentage is not None:
                agreement = (llm_percentage == template_percentage)

            # Generate image filename
            gemini_str = str(gemini_percentage) if gemini_percentage is not None else "none"
            template_str = str(template_percentage) if template_percentage is not None else "none"
            image_filename = f"{int(timestamp)}_{gemini_str}_{template_str}.jpg"
            image_path = self.comparison_images_dir / image_filename

            # Save image
            with open(image_path, 'wb') as f:
                f.write(image_data)

            # Insert record
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO comparison_records
                    (timestamp, gemini_percentage, groq_percentage, template_percentage,
                     template_confidence, image_filename, agreement, llm_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    gemini_percentage,
                    groq_percentage,
                    template_percentage,
                    template_confidence,
                    image_filename,
                    agreement,
                    llm_source
                ))
                await db.commit()

            # Apply FIFO rotation
            await self._apply_fifo_rotation()

            logger.debug(f"Saved comparison: LLM={llm_percentage}% ({llm_source}), Template={template_percentage}%, Agreement={agreement}")
            return True

        except Exception as e:
            logger.error(f"Failed to save comparison record: {e}")
            return False

    async def _apply_fifo_rotation(self):
        """Delete oldest records and images when exceeding max_records"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Count total records
                async with db.execute("SELECT COUNT(*) FROM comparison_records") as cursor:
                    row = await cursor.fetchone()
                    total_records = row[0] if row else 0

                if total_records > self.max_records:
                    # Calculate how many to delete
                    num_to_delete = total_records - self.max_records

                    # Get oldest records to delete
                    async with db.execute("""
                        SELECT id, image_filename
                        FROM comparison_records
                        ORDER BY timestamp ASC
                        LIMIT ?
                    """, (num_to_delete,)) as cursor:
                        records_to_delete = await cursor.fetchall()

                    # Delete images and records
                    for record_id, image_filename in records_to_delete:
                        # Delete image file
                        image_path = self.comparison_images_dir / image_filename
                        if image_path.exists():
                            image_path.unlink()
                            logger.debug(f"FIFO: deleted image {image_filename}")

                        # Delete database record
                        await db.execute("DELETE FROM comparison_records WHERE id = ?", (record_id,))

                    await db.commit()
                    logger.info(f"FIFO rotation: deleted {num_to_delete} oldest comparison records")

        except Exception as e:
            logger.error(f"FIFO rotation failed: {e}")

    async def get_statistics(self) -> Dict:
        """
        Get aggregated comparison statistics

        Returns:
            Dictionary with statistics including by_value and by_llm_source breakdowns
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Total comparisons and agreement rate
                async with db.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN agreement = 1 THEN 1 ELSE 0 END) as agreements
                    FROM comparison_records
                """) as cursor:
                    row = await cursor.fetchone()
                    total_comparisons = row[0] if row else 0
                    agreements = row[1] if row else 0
                    agreement_rate = agreements / total_comparisons if total_comparisons > 0 else 0.0

                # Human verified count
                async with db.execute("""
                    SELECT COUNT(*)
                    FROM comparison_records
                    WHERE human_verified_percentage IS NOT NULL OR human_verified_invalid = 1
                """) as cursor:
                    row = await cursor.fetchone()
                    human_verified_count = row[0] if row else 0

                # By value statistics (using LLM result as reference)
                by_value = {}
                async with db.execute("""
                    SELECT
                        COALESCE(gemini_percentage, groq_percentage) as value,
                        COUNT(*) as count,
                        SUM(CASE WHEN agreement = 1 THEN 1 ELSE 0 END) as agreements
                    FROM comparison_records
                    WHERE COALESCE(gemini_percentage, groq_percentage) IS NOT NULL
                    GROUP BY value
                    ORDER BY value
                """) as cursor:
                    async for row in cursor:
                        value, count, agreements = row
                        by_value[str(value)] = {
                            "count": count,
                            "agreements": agreements,
                            "agreement_rate": round(agreements / count, 3) if count > 0 else 0.0
                        }

                # By LLM source statistics
                by_llm_source = {}
                async with db.execute("""
                    SELECT
                        llm_source,
                        COUNT(*) as count,
                        SUM(CASE WHEN agreement = 1 THEN 1 ELSE 0 END) as agreements
                    FROM comparison_records
                    WHERE llm_source IS NOT NULL
                    GROUP BY llm_source
                """) as cursor:
                    async for row in cursor:
                        source, count, agreements = row
                        # Ensure source is a string (SQLite might return bytes)
                        if isinstance(source, bytes):
                            try:
                                source_str = source.decode('utf-8')
                            except UnicodeDecodeError:
                                source_str = source.decode('latin-1')
                        else:
                            source_str = str(source) if source else "unknown"
                        by_llm_source[source_str] = {
                            "count": count,
                            "agreement_rate": round(agreements / count, 3) if count > 0 else 0.0
                        }

                # Recent disagreements (last 24 hours)
                twenty_four_hours_ago = datetime.now().timestamp() - (24 * 60 * 60)
                async with db.execute("""
                    SELECT COUNT(*)
                    FROM comparison_records
                    WHERE agreement = 0 AND timestamp >= ?
                """, (twenty_four_hours_ago,)) as cursor:
                    row = await cursor.fetchone()
                    recent_disagreements = row[0] if row else 0

                # Calculate accuracy metrics
                accuracy_metrics = await self.calculate_accuracy_metrics()

                # Get error patterns
                error_patterns = await self.get_error_patterns(limit=10)

                return {
                    "total_comparisons": total_comparisons,
                    "agreement_rate": round(agreement_rate, 3),
                    "human_verified_count": human_verified_count,
                    "accuracy": accuracy_metrics,
                    "error_patterns": error_patterns,
                    "by_value": by_value,
                    "by_llm_source": by_llm_source,
                    "recent_disagreements": recent_disagreements
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "total_comparisons": 0,
                "agreement_rate": 0.0,
                "human_verified_count": 0,
                "accuracy": {
                    "llm": {"total_verified": 0, "correct": 0, "accuracy_rate": 0.0, "errors_by_type": {}},
                    "template": {"total_verified": 0, "correct": 0, "accuracy_rate": 0.0, "errors_by_type": {}}
                },
                "error_patterns": [],
                "by_value": {},
                "by_llm_source": {},
                "recent_disagreements": 0,
                "error": str(e)
            }

    async def get_records(
        self,
        limit: int = 50,
        offset: int = 0,
        filter_type: str = "all",
        value: Optional[int] = None
    ) -> List[Dict]:
        """
        Get comparison records with pagination and filtering

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            filter_type: "all" or "disagreements"
            value: Optional filter for specific battery percentage value

        Returns:
            List of comparison record dictionaries
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Build query with filters
                query = """
                    SELECT
                        id, timestamp, gemini_percentage, groq_percentage,
                        template_percentage, template_confidence,
                        human_verified_percentage, human_verified_invalid,
                        image_filename, agreement, llm_source
                    FROM comparison_records
                    WHERE 1=1
                """
                params = []

                if filter_type == "disagreements":
                    query += " AND agreement = 0"

                if value is not None:
                    query += " AND (gemini_percentage = ? OR groq_percentage = ?)"
                    params.extend([value, value])

                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                    records = []
                    for row in rows:
                        # Helper to safely decode bytes from SQLite
                        def safe_decode(val):
                            if isinstance(val, bytes):
                                try:
                                    return val.decode('utf-8')
                                except UnicodeDecodeError:
                                    return val.decode('latin-1')
                            return val

                        # Handle potential bytes from SQLite for all text fields
                        image_filename = safe_decode(row[8])
                        llm_source = safe_decode(row[10])

                        # Handle boolean fields that might come back as bytes
                        human_verified_invalid = row[7]
                        if isinstance(human_verified_invalid, bytes):
                            human_verified_invalid = int.from_bytes(human_verified_invalid, 'little')

                        agreement = row[9]
                        if isinstance(agreement, bytes):
                            agreement = int.from_bytes(agreement, 'little')

                        records.append({
                            "id": row[0],
                            "timestamp": row[1],
                            "gemini_percentage": row[2],
                            "groq_percentage": row[3],
                            "template_percentage": row[4],
                            "template_confidence": row[5],
                            "human_verified_percentage": row[6],
                            "human_verified_invalid": bool(human_verified_invalid) if human_verified_invalid is not None else False,
                            "image_filename": image_filename,
                            "agreement": bool(agreement) if agreement is not None else False,
                            "llm_source": llm_source
                        })

                    return records

        except Exception as e:
            logger.error(f"Failed to get records: {e}")
            return []

    async def update_human_verification(self, record_id: int, human_percentage: int = None, is_invalid: bool = False) -> bool:
        """
        Update a comparison record with human-verified ground truth

        Args:
            record_id: Record ID to update
            human_percentage: Human-verified battery percentage (0-100), or None if invalid
            is_invalid: True if the image is invalid/unreadable

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Validate inputs
            if is_invalid:
                if human_percentage is not None:
                    logger.warning("human_percentage should be None when is_invalid is True")
                    return False
            else:
                if human_percentage is None or not (0 <= human_percentage <= 100):
                    logger.warning(f"Invalid human percentage: {human_percentage}")
                    return False

            async with aiosqlite.connect(self.db_path) as db:
                if is_invalid:
                    await db.execute("""
                        UPDATE comparison_records
                        SET human_verified_percentage = NULL,
                            human_verified_invalid = 1
                        WHERE id = ?
                    """, (record_id,))
                    logger.info(f"Updated record {record_id} with human verification: INVALID")
                else:
                    await db.execute("""
                        UPDATE comparison_records
                        SET human_verified_percentage = ?,
                            human_verified_invalid = 0
                        WHERE id = ?
                    """, (human_percentage, record_id))
                    logger.info(f"Updated record {record_id} with human verification: {human_percentage}%")

                await db.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to update human verification: {e}")
            return False

    async def find_record_by_timestamp(self, timestamp: float, tolerance: float = 2.0) -> Optional[int]:
        """
        Find a comparison record by timestamp with tolerance

        Args:
            timestamp: Target timestamp to match
            tolerance: Time tolerance in seconds (default: ±2 seconds)

        Returns:
            Record ID if found, None otherwise
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT id, timestamp
                    FROM comparison_records
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY ABS(timestamp - ?) ASC
                    LIMIT 1
                """, (timestamp - tolerance, timestamp + tolerance, timestamp)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        record_id, found_timestamp = row
                        logger.debug(f"Found comparison record {record_id} at timestamp {found_timestamp} (target: {timestamp})")
                        return record_id
                    return None

        except Exception as e:
            logger.error(f"Failed to find record by timestamp: {e}")
            return None

    async def update_verification_by_timestamp(self, timestamp: float, human_percentage: int = None, is_invalid: bool = False) -> bool:
        """
        Update a comparison record with human verification by timestamp

        Args:
            timestamp: Timestamp from training image filename
            human_percentage: Human-verified battery percentage (0-100), or None if invalid
            is_invalid: True if the image is invalid/unreadable

        Returns:
            True if updated successfully, False otherwise
        """
        record_id = await self.find_record_by_timestamp(timestamp)
        if record_id is None:
            logger.debug(f"No comparison record found for timestamp {timestamp}")
            return False

        return await self.update_human_verification(record_id, human_percentage, is_invalid)

    async def calculate_accuracy_metrics(self) -> Dict:
        """
        Calculate accuracy metrics for LLM and Template methods based on human-verified records

        Returns:
            Dictionary with accuracy metrics for both methods
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Count total human-verified records
                async with db.execute("""
                    SELECT COUNT(*)
                    FROM comparison_records
                    WHERE human_verified_percentage IS NOT NULL OR human_verified_invalid = 1
                """) as cursor:
                    row = await cursor.fetchone()
                    total_verified = row[0] if row else 0

                if total_verified == 0:
                    return {
                        "llm": {"total_verified": 0, "correct": 0, "accuracy_rate": 0.0, "errors_by_type": {}},
                        "template": {"total_verified": 0, "correct": 0, "accuracy_rate": 0.0, "errors_by_type": {}}
                    }

                # LLM accuracy
                async with db.execute("""
                    SELECT
                        COALESCE(gemini_percentage, groq_percentage) as llm_pct,
                        human_verified_percentage,
                        human_verified_invalid,
                        template_percentage
                    FROM comparison_records
                    WHERE human_verified_percentage IS NOT NULL OR human_verified_invalid = 1
                """) as cursor:
                    rows = await cursor.fetchall()

                    llm_correct = 0
                    llm_errors = defaultdict(int)
                    template_correct = 0
                    template_errors = defaultdict(int)

                    for row in rows:
                        llm_pct, human_pct, is_invalid, template_pct = row

                        # Determine ground truth
                        if is_invalid:
                            ground_truth = "invalid"
                        else:
                            ground_truth = human_pct

                        # Check LLM accuracy
                        if llm_pct is not None:
                            if ground_truth == "invalid":
                                if llm_pct is None:  # LLM said invalid (represented as None)
                                    llm_correct += 1
                                else:
                                    llm_errors["invalid_as_value"] += 1
                            else:
                                if llm_pct == ground_truth:
                                    llm_correct += 1
                                elif llm_pct is None:
                                    llm_errors["value_as_invalid"] += 1
                                elif abs(llm_pct - ground_truth) == 1:
                                    llm_errors["off_by_one"] += 1
                                else:
                                    llm_errors["off_by_more"] += 1

                        # Check Template accuracy
                        if template_pct is not None or ground_truth == "invalid":
                            if ground_truth == "invalid":
                                if template_pct is None:  # Template said invalid (represented as None)
                                    template_correct += 1
                                else:
                                    template_errors["invalid_as_value"] += 1
                            else:
                                if template_pct == ground_truth:
                                    template_correct += 1
                                elif template_pct is None:
                                    template_errors["value_as_invalid"] += 1
                                elif abs(template_pct - ground_truth) == 1:
                                    template_errors["off_by_one"] += 1
                                else:
                                    template_errors["off_by_more"] += 1

                    llm_accuracy = llm_correct / total_verified if total_verified > 0 else 0.0
                    template_accuracy = template_correct / total_verified if total_verified > 0 else 0.0

                    return {
                        "llm": {
                            "total_verified": total_verified,
                            "correct": llm_correct,
                            "accuracy_rate": round(llm_accuracy, 3),
                            "errors_by_type": dict(llm_errors)
                        },
                        "template": {
                            "total_verified": total_verified,
                            "correct": template_correct,
                            "accuracy_rate": round(template_accuracy, 3),
                            "errors_by_type": dict(template_errors)
                        }
                    }

        except Exception as e:
            logger.error(f"Failed to calculate accuracy metrics: {e}")
            return {
                "llm": {"total_verified": 0, "correct": 0, "accuracy_rate": 0.0, "errors_by_type": {}},
                "template": {"total_verified": 0, "correct": 0, "accuracy_rate": 0.0, "errors_by_type": {}},
                "error": str(e)
            }

    async def get_error_patterns(self, limit: int = 10) -> List[Dict]:
        """
        Get most common error patterns for each method

        Args:
            limit: Maximum number of patterns to return per method

        Returns:
            List of error pattern dictionaries
        """
        try:
            patterns = []

            async with aiosqlite.connect(self.db_path) as db:
                # LLM error patterns
                async with db.execute("""
                    SELECT
                        'llm' as method,
                        COALESCE(gemini_percentage, groq_percentage) as predicted,
                        CASE
                            WHEN human_verified_invalid = 1 THEN 'invalid'
                            ELSE CAST(human_verified_percentage AS TEXT)
                        END as actual,
                        COUNT(*) as count
                    FROM comparison_records
                    WHERE (human_verified_percentage IS NOT NULL OR human_verified_invalid = 1)
                        AND (
                            (human_verified_invalid = 1 AND COALESCE(gemini_percentage, groq_percentage) IS NOT NULL)
                            OR (human_verified_invalid = 0 AND COALESCE(gemini_percentage, groq_percentage) != human_verified_percentage)
                        )
                    GROUP BY predicted, actual
                    ORDER BY count DESC
                    LIMIT ?
                """, (limit,)) as cursor:
                    async for row in cursor:
                        method, predicted, actual, count = row
                        # Handle potential bytes from SQLite with fallback encoding
                        if isinstance(method, bytes):
                            try:
                                method = method.decode('utf-8')
                            except UnicodeDecodeError:
                                method = method.decode('latin-1')
                        if isinstance(actual, bytes):
                            try:
                                actual = actual.decode('utf-8')
                            except UnicodeDecodeError:
                                actual = actual.decode('latin-1')
                        patterns.append({
                            "method": method,
                            "predicted": predicted if predicted is not None else "invalid",
                            "actual": actual,
                            "count": count
                        })

                # Template error patterns
                async with db.execute("""
                    SELECT
                        'template' as method,
                        template_percentage as predicted,
                        CASE
                            WHEN human_verified_invalid = 1 THEN 'invalid'
                            ELSE CAST(human_verified_percentage AS TEXT)
                        END as actual,
                        COUNT(*) as count
                    FROM comparison_records
                    WHERE (human_verified_percentage IS NOT NULL OR human_verified_invalid = 1)
                        AND (
                            (human_verified_invalid = 1 AND template_percentage IS NOT NULL)
                            OR (human_verified_invalid = 0 AND template_percentage != human_verified_percentage)
                        )
                    GROUP BY predicted, actual
                    ORDER BY count DESC
                    LIMIT ?
                """, (limit,)) as cursor:
                    async for row in cursor:
                        method, predicted, actual, count = row
                        # Handle potential bytes from SQLite with fallback encoding
                        if isinstance(method, bytes):
                            try:
                                method = method.decode('utf-8')
                            except UnicodeDecodeError:
                                method = method.decode('latin-1')
                        if isinstance(actual, bytes):
                            try:
                                actual = actual.decode('utf-8')
                            except UnicodeDecodeError:
                                actual = actual.decode('latin-1')
                        patterns.append({
                            "method": method,
                            "predicted": predicted if predicted is not None else "invalid",
                            "actual": actual,
                            "count": count
                        })

            # Sort all patterns by count
            patterns.sort(key=lambda x: x["count"], reverse=True)
            return patterns

        except Exception as e:
            logger.error(f"Failed to get error patterns: {e}")
            return []


# Global instance
comparison_storage = ComparisonStorage()
