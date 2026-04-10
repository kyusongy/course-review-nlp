#!/usr/bin/env python
"""Scrape all UNC reviews and build the reviews_all.parquet dataset.

Usage:
    uv run python run_pipeline.py

After this, run train.py to train the model and score all reviews.
"""

import json
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def parse_all_reviews() -> pd.DataFrame:
    """Parse all raw JSON files into a single DataFrame."""
    # Load teacher metadata for legacy_id -> name/dept mapping
    teachers = pd.read_parquet(PROCESSED_DIR / "teachers_all.parquet")
    teacher_map = {
        row["legacy_id"]: {
            "professor_name": row["professor_name"],
            "department": row["department"],
        }
        for _, row in teachers.iterrows()
    }

    all_reviews = []
    for f in sorted(RAW_DIR.glob("*.json")):
        legacy_id = int(f.stem)
        info = teacher_map.get(legacy_id)
        if not info:
            continue

        data = json.loads(f.read_text())
        if not isinstance(data, list):
            continue

        for r in data:
            comment = r.get("comment", "").strip()
            if not comment:
                continue
            comment = re.sub(r"\s+", " ", comment).strip()

            all_reviews.append(
                {
                    "review_text": comment,
                    "star_rating": int(r.get("qualityRating", 0)),
                    "difficulty_rating": int(r.get("difficultyRating", 0)),
                    "would_take_again": (
                        None
                        if r.get("wouldTakeAgain") is None
                        else bool(r["wouldTakeAgain"])
                    ),
                    "course_name": r.get("class", ""),
                    "professor_name": info["professor_name"],
                    "department": info["department"],
                    "date": r.get("date", ""),
                    "thumbs_up": r.get("thumbsUpTotal", 0),
                    "thumbs_down": r.get("thumbsDownTotal", 0),
                }
            )

    df = pd.DataFrame(all_reviews)
    df = df.drop_duplicates(
        subset=["review_text", "professor_name", "course_name"]
    ).reset_index(drop=True)
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output = PROCESSED_DIR / "reviews_all.parquet"
    if output.exists():
        print(
            f"reviews_all.parquet already exists ({len(pd.read_parquet(output))} reviews). Skipping."
        )
    else:
        if not any(RAW_DIR.glob("*.json")):
            print(
                "No raw data found. Scrape first with: uv run python -m src.scraper.run"
            )
            return

        print("Parsing all raw reviews...")
        df = parse_all_reviews()
        df.to_parquet(output, index=False)
        print(f"Saved {len(df)} reviews to {output}")

    print("\nNext: uv run python train.py")


if __name__ == "__main__":
    main()
