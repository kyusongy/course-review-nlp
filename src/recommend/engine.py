import pandas as pd
from src.models.zero_shot import TOPICS

TOPIC_KEYS = [t.lower().replace(" ", "_") for t in TOPICS]
# ["workload", "grading", "teaching_quality", "accessibility", "exam_difficulty"]


def score_professors(
    prof_scores: pd.DataFrame, weights: dict[str, float]
) -> pd.DataFrame:
    df = prof_scores.copy()
    total_weight = sum(weights.values())
    if total_weight == 0:
        df["score"] = 0.0
    else:
        df["score"] = sum(weights[k] * df[k] for k in weights) / total_weight
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def filter_results(
    df: pd.DataFrame, min_reviews: int = 3, course_prefix: str | None = None
) -> pd.DataFrame:
    mask = df["num_reviews"] >= min_reviews
    if course_prefix is not None:
        mask &= df["course_name"].str.startswith(course_prefix)
    return df[mask].reset_index(drop=True)


def aggregate_professor_scores(reviews_df: pd.DataFrame) -> pd.DataFrame:
    topic_score_cols = [f"topic_{k}_score" for k in TOPIC_KEYS]
    agg = (
        reviews_df.groupby("professor_name")
        .agg(
            num_reviews=("professor_name", "count"),
            **{k: (f"topic_{k}_score", "mean") for k in TOPIC_KEYS},
        )
        .fillna(0.0)
        .reset_index()
    )
    return agg
