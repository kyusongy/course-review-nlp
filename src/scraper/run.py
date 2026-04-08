"""Scrape RMP for UNC Stats/Biostat professors and save ratings."""

import asyncio
import json
from pathlib import Path

import httpx
import pandas as pd

from src.scraper.client import (
    TARGET_DEPARTMENTS,
    fetch_all_ratings,
    fetch_teachers,
)
from src.scraper.parse import parse_rating, parse_teacher, ratings_to_dataframe

OUT_DIR = Path("data/processed")


async def scrape() -> pd.DataFrame:
    async with httpx.AsyncClient() as client:
        print("Fetching teacher list...")
        all_teachers = await fetch_teachers(client)
        teachers = [t for t in all_teachers if t["department"] in TARGET_DEPARTMENTS]
        print(f"Found {len(teachers)} professors in target departments")

        all_ratings: list[dict] = []
        teacher_records: list[dict] = []

        for teacher in teachers:
            parsed = parse_teacher(teacher)
            teacher_records.append(parsed)
            name = parsed["professor_name"]
            print(f"  Fetching ratings for {name} ({teacher['department']})...")

            raw_ratings = await fetch_all_ratings(
                client, teacher["id"], teacher["legacyId"]
            )
            for r in raw_ratings:
                all_ratings.append(parse_rating(r, name))

            await asyncio.sleep(0.5)

    df_ratings = ratings_to_dataframe(all_ratings)
    df_teachers = pd.DataFrame(teacher_records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_ratings.to_parquet(OUT_DIR / "reviews.parquet", index=False)
    df_teachers.to_parquet(OUT_DIR / "teachers.parquet", index=False)

    print(f"\nDone. {len(df_ratings)} ratings, {len(df_teachers)} professors")
    print(f"Saved to {OUT_DIR}/")
    return df_ratings


def main():
    asyncio.run(scrape())


if __name__ == "__main__":
    main()
