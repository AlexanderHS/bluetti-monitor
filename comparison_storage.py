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
                        by_llm_source[source] = {
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

                return {
                    "total_comparisons": total_comparisons,
                    "agreement_rate": round(agreement_rate, 3),
                    "by_value": by_value,
                    "by_llm_source": by_llm_source,
                    "recent_disagreements": recent_disagreements
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "total_comparisons": 0,
                "agreement_rate": 0.0,
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
                        human_verified_percentage, image_filename,
                        agreement, llm_source
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
                        records.append({
                            "id": row[0],
                            "timestamp": row[1],
                            "gemini_percentage": row[2],
                            "groq_percentage": row[3],
                            "template_percentage": row[4],
                            "template_confidence": row[5],
                            "human_verified_percentage": row[6],
                            "image_filename": row[7],
                            "agreement": bool(row[8]),
                            "llm_source": row[9]
                        })

                    return records

        except Exception as e:
            logger.error(f"Failed to get records: {e}")
            return []

    async def update_human_verification(self, record_id: int, human_percentage: int) -> bool:
        """
        Update a comparison record with human-verified ground truth

        Args:
            record_id: Record ID to update
            human_percentage: Human-verified battery percentage (0-100)

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Validate percentage
            if not (0 <= human_percentage <= 100):
                logger.warning(f"Invalid human percentage: {human_percentage}")
                return False

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE comparison_records
                    SET human_verified_percentage = ?
                    WHERE id = ?
                """, (human_percentage, record_id))
                await db.commit()

                logger.info(f"Updated record {record_id} with human verification: {human_percentage}%")
                return True

        except Exception as e:
            logger.error(f"Failed to update human verification: {e}")
            return False


# Global instance
comparison_storage = ComparisonStorage()
