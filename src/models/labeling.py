"""Labeling workflow: interactive annotation + train/test split."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.evaluate import _majority_sentiment
from src.models.zero_shot import TOPICS

ANNOTATIONS_PATH = Path("data/labels/annotations.json")


def load_annotations() -> dict:
    if ANNOTATIONS_PATH.exists():
        return json.loads(ANNOTATIONS_PATH.read_text())
    return {}


def save_annotations(annotations: dict) -> None:
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_PATH.write_text(json.dumps(annotations, indent=2))


def _conf_col(topic: str) -> str:
    return f"topic_{topic.lower().replace(' ', '_')}_conf"


def label_reviews(scores_path: Path, n: int = 200) -> None:
    """Interactive CLI labeling. Shows zero-shot predictions; accept or correct."""
    df = pd.read_parquet(scores_path)
    annotations = load_annotations()

    labeled = 0
    for _, row in df.iterrows():
        if labeled >= n:
            break

        idx = str(int(row["idx"]))
        if idx in annotations:
            continue

        # Predicted topics above threshold
        predicted_topics = [t for t in TOPICS if (row.get(_conf_col(t), 0) or 0) > 0.3]
        predicted_sentiment = row.get("overall_sentiment", "neutral")

        print("\n" + "=" * 60)
        print(f"[{idx}] {row.get('review_text', row.get('text', ''))}\n")
        print(f"  Topics    : {predicted_topics or '(none)'}")
        print(f"  Sentiment : {predicted_sentiment}")
        print()
        print("  'a' = accept  |  'q' = quit  |  or type correction")
        print("  Correction format:  topics=Workload,Grading  sentiment=positive")

        choice = input("  > ").strip()

        if choice == "q":
            save_annotations(annotations)
            print(f"Saved {len(annotations)} annotations.")
            return
        elif choice == "a" or choice == "":
            annotations[idx] = {
                "topics": predicted_topics,
                "sentiment": predicted_sentiment,
            }
        else:
            # Parse manual override
            topics = predicted_topics
            sentiment = predicted_sentiment
            for part in choice.split():
                if part.startswith("topics="):
                    raw = part[len("topics=") :]
                    topics = [t.strip() for t in raw.split(",") if t.strip()]
                elif part.startswith("sentiment="):
                    sentiment = part[len("sentiment=") :]
            annotations[idx] = {"topics": topics, "sentiment": sentiment}

        save_annotations(annotations)
        labeled += 1

    print(f"\nDone. {len(annotations)} total annotations saved.")


def split_labeled_data(
    reviews_path: Path, train_ratio: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load annotations, merge onto reviews, return (train_df, test_df).

    Handles both label formats:
    - Old: {"topics": ["A", "B"], "sentiment": "positive"}
    - New: {"topics": {"A": "positive", "B": "negative"}}
    """
    df = pd.read_parquet(reviews_path)
    annotations = load_annotations()

    if not annotations:
        raise ValueError("No annotations found. Run label_reviews() first.")

    rows = []
    for idx_str, ann in annotations.items():
        topics = ann["topics"]
        if isinstance(topics, dict):
            # New format: per-topic sentiment
            row = {
                "idx": int(idx_str),
                "label_topics": topics,  # {topic: sentiment}
                "label_sentiment": _majority_sentiment(topics),
            }
        else:
            # Old format: topic list + overall sentiment
            row = {
                "idx": int(idx_str),
                "label_topics": {t: ann.get("sentiment", "neutral") for t in topics},
                "label_sentiment": ann.get("sentiment", "neutral"),
            }
        rows.append(row)
    ann_df = pd.DataFrame(rows)

    # Merge on index position (idx col if present, otherwise use df index)
    if "idx" in df.columns:
        df = df.merge(ann_df, on="idx", how="inner")
    else:
        df = df.reset_index().rename(columns={"index": "idx"})
        df = df.merge(ann_df, on="idx", how="inner")

    train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def auto_label_from_zero_shot(scores_path: Path, conf_threshold: float = 0.3) -> dict:
    """Generate annotations from zero-shot predictions. Use as bootstrapping for manual review."""
    df = pd.read_parquet(scores_path)
    annotations = {}
    for _, row in df.iterrows():
        idx = str(int(row["idx"]))
        topics = []
        for topic in TOPICS:
            key = topic.lower().replace(" ", "_")
            conf = row.get(f"topic_{key}_conf", 0)
            if conf and conf > conf_threshold:
                topics.append(topic)
        annotations[idx] = {
            "topics": topics,
            "sentiment": row.get("overall_sentiment", "neutral"),
        }
    return annotations
