#!/usr/bin/env python
"""Run the full Course Compass pipeline end-to-end."""

import asyncio
from pathlib import Path

import pandas as pd

from src.scraper.run import scrape
from src.scraper.preprocess import preprocess_reviews
from src.models.process import process_all


def main():
    raw_path = Path("data/processed/reviews.parquet")
    processed_dir = Path("data/processed")

    if not raw_path.exists():
        print("Step 1: Scraping RateMyProfessor...")
        asyncio.run(scrape())

    print("Step 2: Preprocessing...")
    df = pd.read_parquet(raw_path)
    df = preprocess_reviews(df)
    df.to_parquet(raw_path, index=False)
    print(f"  {len(df)} cleaned reviews")

    print("Step 3: Running zero-shot pipeline...")
    process_all(raw_path, processed_dir)

    print("\nPipeline complete!")
    print("Next steps:")
    print("  1. Label data:  uv run python -m src.models.labeling")
    print("  2. Fine-tune:   uv run python -c 'from src.models.fine_tune import ...'")
    print("  3. Evaluate:    uv run python -c 'from src.models.evaluate import ...'")
    print("  4. Launch app:  uv run streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()
