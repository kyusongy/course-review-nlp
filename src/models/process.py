"""Batch-process all reviews through the model pipelines."""

from pathlib import Path
import pandas as pd
from src.models.baseline import compute_baseline_scores
from src.models.zero_shot import TopicClassifier, SentimentAnalyzer, TOPICS


def process_all(reviews_path: Path, output_dir: Path):
    """Run all approaches on the review dataset."""
    df = pd.read_parquet(reviews_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layer 1: Star rating baseline
    baseline_df = compute_baseline_scores(df)
    baseline_df.to_parquet(output_dir / "baseline_scores.parquet", index=False)

    # Layer 2: Zero-shot
    tc = TopicClassifier()
    sa = SentimentAnalyzer()

    records = []
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  Processing review {i}/{len(df)}...")
        text = row["review_text"]
        topic_scores = tc.classify_with_scores(text)
        topic_sentiments = sa.analyze_by_topic(text, tc)
        overall = sa.analyze(text)

        record = {
            "idx": i,
            "professor_name": row["professor_name"],
            "course_name": row["course_name"],
            "review_text": text,
            "star_rating": row["star_rating"],
            "overall_sentiment": overall["label"],
            "overall_score": overall["score"],
        }
        for topic in TOPICS:
            key = topic.lower().replace(" ", "_")
            record[f"topic_{key}_conf"] = topic_scores.get(topic, 0.0)
            if topic in topic_sentiments:
                record[f"topic_{key}_sentiment"] = topic_sentiments[topic]["label"]
                record[f"topic_{key}_score"] = topic_sentiments[topic]["score"]
            else:
                record[f"topic_{key}_sentiment"] = None
                record[f"topic_{key}_score"] = None
        records.append(record)

    results_df = pd.DataFrame(records)
    results_df.to_parquet(output_dir / "zero_shot_scores.parquet", index=False)
    print(f"Saved zero-shot scores for {len(results_df)} reviews.")
    return results_df


if __name__ == "__main__":
    process_all(
        reviews_path=Path("data/processed/reviews.parquet"),
        output_dir=Path("data/processed"),
    )
