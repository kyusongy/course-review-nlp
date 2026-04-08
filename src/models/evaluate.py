"""Evaluation framework: compare zero-shot vs fine-tuned approaches."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from src.models.baseline import star_to_sentiment
from src.models.fine_tune import create_topic_labels
from src.models.zero_shot import TOPICS


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Single-label classification metrics (e.g. sentiment)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_per_class": {
            label: score
            for label, score in zip(
                sorted(set(y_true) | set(y_pred)),
                f1_score(
                    y_true,
                    y_pred,
                    labels=sorted(set(y_true) | set(y_pred)),
                    average=None,
                    zero_division=0,
                ),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def compute_multilabel_metrics(y_true, y_pred) -> dict:
    """Multi-label classification metrics (e.g. topics)."""
    per_topic = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_per_topic": {
            topic: float(score) for topic, score in zip(TOPICS, per_topic)
        },
    }


def compare_approaches(
    test_df,
    zero_shot_topics: list[list[str]],
    zero_shot_sentiments: list[str],
    finetuned_topics: list[list[str]],
    finetuned_sentiments: list[str],
) -> dict:
    """Compare zero-shot vs fine-tuned vs baseline on the test set."""
    true_sentiments = test_df["label_sentiment"].tolist()
    true_topics_raw = test_df["label_topics"].tolist()
    star_ratings = test_df["star_rating"].tolist()

    # Multi-hot vectors
    true_multihot = [create_topic_labels(t) for t in true_topics_raw]
    zs_multihot = [create_topic_labels(t) for t in zero_shot_topics]
    ft_multihot = [create_topic_labels(t) for t in finetuned_topics]

    # Baseline sentiment from star ratings
    baseline_sentiments = [star_to_sentiment(r) for r in star_ratings]

    return {
        "topic_classification": {
            "zero_shot": compute_multilabel_metrics(true_multihot, zs_multihot),
            "finetuned": compute_multilabel_metrics(true_multihot, ft_multihot),
        },
        "sentiment": {
            "baseline": compute_classification_metrics(
                true_sentiments, baseline_sentiments
            ),
            "zero_shot": compute_classification_metrics(
                true_sentiments, zero_shot_sentiments
            ),
            "finetuned": compute_classification_metrics(
                true_sentiments, finetuned_sentiments
            ),
        },
        "agreement": {
            "zero_shot_vs_finetuned_sentiment": cohen_kappa_score(
                zero_shot_sentiments, finetuned_sentiments
            ),
            "zero_shot_vs_baseline_sentiment": cohen_kappa_score(
                zero_shot_sentiments, baseline_sentiments
            ),
            "finetuned_vs_baseline_sentiment": cohen_kappa_score(
                finetuned_sentiments, baseline_sentiments
            ),
        },
    }
