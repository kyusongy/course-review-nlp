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


def aggregate_professor_scores(
    reviews_df: pd.DataFrame, prior_strength: float = 5.0
) -> pd.DataFrame:
    """Per-topic mean per professor, shrunk toward the department prior.

    For each (prof, topic):
        shrunk = (n * prof_mean + m * dept_mean) / (n + m)

    n is the count of reviews *mentioning that topic* (non-null score), not
    the prof's total review count — a prof with 20 reviews but only 2
    mentioning Exam Difficulty shrinks based on n=2 for that topic.

    m (prior_strength) controls shrinkage. Low-N profs land near their
    department's mean; high-N profs keep their own mean. Filtering on
    num_reviews is a separate lever (visibility, not score).
    """
    topic_cols = [f"topic_{k}_score" for k in TOPIC_KEYS]

    dept_mean = reviews_df.groupby("department")[topic_cols].mean().fillna(0.0)

    grouped = reviews_df.groupby("professor_name")
    n = grouped[topic_cols].count()
    raw_mean = grouped[topic_cols].mean().fillna(0.0)
    num_reviews = grouped.size().rename("num_reviews")
    prof_dept = grouped["department"].first()

    prior = dept_mean.loc[prof_dept.values]
    prior.index = prof_dept.index

    m = prior_strength
    blended = (n * raw_mean + m * prior) / (n + m)
    blended.columns = TOPIC_KEYS

    out = blended.reset_index()
    out.insert(1, "department", prof_dept.values)
    out.insert(2, "num_reviews", num_reviews.values)
    return out
