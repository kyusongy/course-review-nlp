#!/usr/bin/env python
"""Train the joint model, evaluate on test set, and score all reviews.

Usage:
    uv run python train.py              # train + evaluate + score
    uv run python train.py --epochs 10  # custom epochs
    uv run python train.py --eval-only  # skip training, just evaluate + score
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.models.evaluate import compare_approaches
from src.models.fine_tune import (
    MODELS_DIR,
    TOPICS,
    TopicSentimentDataset,
    TopicSentimentModel,
    _DEVICE,
    decode_topic_sentiments,
    predict_joint,
)
from src.models.zero_shot import SentimentAnalyzer, TopicClassifier

DATA_DIR = Path("data/processed")
SENT_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
TOPIC_KEYS = [t.lower().replace(" ", "_") for t in TOPICS]


def train(epochs: int = 10, batch_size: int = 16, lr: float = 2e-5) -> None:
    """Train joint model with validation-based early stopping."""
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")

    train_ds = TopicSentimentDataset(
        train_df["review_text"].tolist(), train_df["label_topics"].tolist()
    )
    val_ds = TopicSentimentDataset(
        val_df["review_text"].tolist(), val_df["label_topics"].tolist()
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TopicSentimentModel().to(_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs
    )

    save_dir = MODELS_DIR / "joint_classifier"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Device: {_DEVICE}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'':>5}")
    print("-" * 38)

    best_val = float("inf")
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(_DEVICE) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            out["loss"].backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += out["loss"].item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(_DEVICE) for k, v in batch.items()}
                out = model(
                    batch["input_ids"], batch["attention_mask"], batch["labels"]
                )
                val_loss += out["loss"].item()
        val_loss /= len(val_loader)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_dir / "model.pt")
            train_ds._tokenizer.save_pretrained(save_dir)
            marker = " *"

        print(f"{epoch + 1:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} |{marker}")

    print(f"\nBest epoch: {best_epoch} (val loss: {best_val:.4f})")


def evaluate() -> dict:
    """Evaluate fine-tuned vs zero-shot vs baseline on test set."""
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    texts = test_df["review_text"].tolist()
    print(f"\nEvaluating on {len(test_df)} test reviews...")

    print("  Running fine-tuned model...")
    ft = predict_joint(texts)

    print("  Running zero-shot pipeline...")
    tc = TopicClassifier()
    sa = SentimentAnalyzer()
    zs = [sa.analyze_by_topic_flat(t, tc) for t in texts]

    print("  Comparing...")
    results = compare_approaches(test_df, zs, ft)

    # Print summary
    print("\n  Per-Topic Sentiment (Fine-Tuned):")
    for topic in TOPICS:
        info = results["per_topic_sentiment"]["finetuned"].get(topic, {})
        acc = info.get("accuracy", 0) or 0
        print(f"    {topic}: {acc:.1%}")

    det = results["per_topic_sentiment"]["finetuned"]["_topic_detection"]
    print(
        f"\n  Topic Detection F1: {det['f1_macro']:.3f} macro, {det['f1_micro']:.3f} micro"
    )

    for approach in ["baseline", "zero_shot", "finetuned"]:
        d = results["overall_sentiment"][approach]
        print(f"  Overall Sentiment ({approach}): {d['accuracy']:.1%}")

    Path(DATA_DIR / "evaluation_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print("\n  Saved evaluation_results.json")
    return results


def score_all() -> None:
    """Score all 65K reviews with the fine-tuned model."""
    reviews = pd.read_parquet(DATA_DIR / "reviews_all.parquet")
    texts = reviews["review_text"].tolist()
    print(f"\nScoring {len(texts)} reviews...")

    # Load model once for batched inference
    model_dir = MODELS_DIR / "joint_classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TopicSentimentModel()
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location=_DEVICE, weights_only=True)
    )
    model.to(_DEVICE)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            ).to(_DEVICE)
            logits = model(enc["input_ids"], enc["attention_mask"])["logits"]
            for s in logits.argmax(dim=-1).cpu().tolist():
                all_preds.append(decode_topic_sentiments(s))
            if i % 10000 == 0:
                print(f"  {i}/{len(texts)}...")

    # Build scored dataframe
    records = []
    for i, (pred, (_, row)) in enumerate(zip(all_preds, reviews.iterrows())):
        record = {
            "idx": i,
            "professor_name": row["professor_name"],
            "department": row["department"],
            "course_name": row["course_name"],
            "review_text": row["review_text"],
            "star_rating": row["star_rating"],
        }
        for topic, key in zip(TOPICS, TOPIC_KEYS):
            if topic in pred:
                record[f"topic_{key}_sentiment"] = pred[topic]
                record[f"topic_{key}_score"] = SENT_SCORE[pred[topic]]
            else:
                record[f"topic_{key}_sentiment"] = None
                record[f"topic_{key}_score"] = None
        records.append(record)

    scored = pd.DataFrame(records)
    scored.to_parquet(DATA_DIR / "scored_all.parquet", index=False)
    print(f"  Saved {len(scored)} scored reviews to scored_all.parquet")


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate, and score reviews")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if not args.eval_only:
        train(epochs=args.epochs)

    evaluate()
    score_all()
    print("\nDone. Launch app: uv run streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()
