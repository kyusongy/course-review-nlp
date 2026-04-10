"""Evaluation framework: compare zero-shot vs fine-tuned approaches.

Supports both:
- Topic detection (multi-label: is topic present?)
- Per-topic sentiment (4-class per topic: not_discussed/positive/neutral/negative)
"""

from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from src.models.baseline import star_to_sentiment
from src.models.fine_tune import create_topic_labels, encode_topic_sentiments
from src.models.zero_shot import TOPICS


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Single-label classification metrics (e.g. sentiment)."""
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_per_class": {
            label: score
            for label, score in zip(
                labels,
                f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def compute_multilabel_metrics(y_true, y_pred) -> dict:
    """Multi-label classification metrics (topic detection only)."""
    per_topic = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_per_topic": {
            topic: float(score) for topic, score in zip(TOPICS, per_topic)
        },
    }


def compute_per_topic_sentiment_metrics(
    y_true: list[dict[str, str]],
    y_pred: list[dict[str, str]],
) -> dict:
    """Evaluate per-topic sentiment predictions.

    y_true and y_pred are lists of {topic: sentiment} dicts.
    Returns per-topic accuracy and F1 for sentiment classification,
    evaluated only on reviews where the topic is present in ground truth.
    """
    results = {}
    for topic in TOPICS:
        true_labels = []
        pred_labels = []
        for gt, pr in zip(y_true, y_pred):
            gt_val = gt.get(topic)
            if gt_val is not None:  # topic was labeled (not None/missing)
                true_labels.append(gt_val)
                pr_val = pr.get(topic)
                pred_labels.append(pr_val if pr_val is not None else "not_discussed")
        if not true_labels:
            results[topic] = {"n": 0, "accuracy": None, "f1_macro": None}
            continue
        try:
            acc = accuracy_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        except Exception:
            acc = 0.0
            f1 = 0.0
        results[topic] = {"n": len(true_labels), "accuracy": acc, "f1_macro": f1}

    # Also compute topic detection accuracy (did we find the right topics?)
    topic_detection_true = []
    topic_detection_pred = []
    for gt, pr in zip(y_true, y_pred):
        topic_detection_true.append(create_topic_labels(gt))
        topic_detection_pred.append(create_topic_labels(pr))

    results["_topic_detection"] = compute_multilabel_metrics(
        topic_detection_true, topic_detection_pred
    )

    return results


def compare_approaches(
    test_df,
    zero_shot_topic_sentiments: list[dict[str, str]],
    finetuned_topic_sentiments: list[dict[str, str]],
) -> dict:
    """Compare zero-shot vs fine-tuned vs baseline on per-topic sentiment.

    test_df must have:
    - 'label_topics': dict[str, str] (per-topic sentiment ground truth)
    - 'label_sentiment': str (overall sentiment)
    - 'star_rating': int
    """
    ground_truth = test_df["label_topics"].tolist()
    true_sentiments = test_df["label_sentiment"].tolist()
    star_ratings = test_df["star_rating"].tolist()

    # Overall sentiment from each approach
    baseline_sentiments = [star_to_sentiment(r) for r in star_ratings]
    zs_overall = [_majority_sentiment(ts) for ts in zero_shot_topic_sentiments]
    ft_overall = [_majority_sentiment(ts) for ts in finetuned_topic_sentiments]

    return {
        "per_topic_sentiment": {
            "zero_shot": compute_per_topic_sentiment_metrics(
                ground_truth, zero_shot_topic_sentiments
            ),
            "finetuned": compute_per_topic_sentiment_metrics(
                ground_truth, finetuned_topic_sentiments
            ),
        },
        "overall_sentiment": {
            "baseline": compute_classification_metrics(
                true_sentiments, baseline_sentiments
            ),
            "zero_shot": compute_classification_metrics(true_sentiments, zs_overall),
            "finetuned": compute_classification_metrics(true_sentiments, ft_overall),
        },
        "agreement": {
            "zero_shot_vs_finetuned": cohen_kappa_score(zs_overall, ft_overall),
            "zero_shot_vs_baseline": cohen_kappa_score(zs_overall, baseline_sentiments),
            "finetuned_vs_baseline": cohen_kappa_score(ft_overall, baseline_sentiments),
        },
    }


def _majority_sentiment(topic_sentiments: dict[str, str]) -> str:
    """Derive overall sentiment from per-topic sentiments."""
    if not topic_sentiments:
        return "neutral"
    # Filter out None values (parquet expands sparse dicts)
    values = [v for v in topic_sentiments.values() if v is not None]
    if not values:
        return "neutral"
    counts = Counter(values)
    return counts.most_common(1)[0][0]
