import pandas as pd
from src.recommend.engine import (
    aggregate_professor_scores,
    filter_results,
    score_professors,
)


def make_scores_df():
    return pd.DataFrame(
        [
            {
                "professor_name": "Alice",
                "num_reviews": 10,
                "workload": 0.5,
                "grading": 0.8,
                "teaching_quality": 0.9,
                "accessibility": 0.6,
                "exam_difficulty": -0.2,
            },
            {
                "professor_name": "Bob",
                "num_reviews": 5,
                "workload": -0.3,
                "grading": -0.5,
                "teaching_quality": 0.2,
                "accessibility": 0.1,
                "exam_difficulty": -0.8,
            },
            {
                "professor_name": "Carol",
                "num_reviews": 2,
                "workload": 0.1,
                "grading": 0.1,
                "teaching_quality": 0.1,
                "accessibility": 0.1,
                "exam_difficulty": 0.1,
            },
        ]
    )


def test_score_equal_weights():
    df = make_scores_df()
    weights = {
        "workload": 5,
        "grading": 5,
        "teaching_quality": 5,
        "accessibility": 5,
        "exam_difficulty": 5,
    }
    result = score_professors(df, weights)
    assert result.iloc[0]["professor_name"] == "Alice"
    assert "score" in result.columns


def test_score_grading_only():
    df = make_scores_df()
    weights = {
        "workload": 0,
        "grading": 10,
        "teaching_quality": 0,
        "accessibility": 0,
        "exam_difficulty": 0,
    }
    result = score_professors(df, weights)
    assert result.iloc[0]["professor_name"] == "Alice"


def test_filter_min_reviews():
    df = make_scores_df()
    df["score"] = [0.5, 0.3, 0.1]
    result = filter_results(df, min_reviews=3)
    assert len(result) == 2
    assert "Carol" not in result["professor_name"].values


def _reviews_with(rows):
    """Build a reviews_df with all topic_score columns defaulting to NaN."""
    cols = [
        "topic_workload_score",
        "topic_grading_score",
        "topic_teaching_quality_score",
        "topic_accessibility_score",
        "topic_exam_difficulty_score",
    ]
    return pd.DataFrame(
        [
            {c: r.get(c, float("nan")) for c in cols}
            | {"professor_name": r["p"], "department": r["d"]}
            for r in rows
        ]
    )


def test_shrinkage_low_n_pulled_to_dept_mean():
    # Many zero anchors dilute Lucky's contribution to the dept mean.
    rows = [
        {"p": f"Anchor{i}", "d": "A", "topic_workload_score": 0.0} for i in range(99)
    ]
    rows.append({"p": "Lucky", "d": "A", "topic_workload_score": 1.0})
    agg = aggregate_professor_scores(_reviews_with(rows), prior_strength=5.0)
    lucky = agg[agg["professor_name"] == "Lucky"].iloc[0]
    # dept_mean = 1/100 = 0.01, n=1, m=5 → (1*1 + 5*0.01)/6 ≈ 0.175
    assert abs(lucky["workload"] - 0.175) < 0.01
    # Strong pull toward the dept's 0 prior despite Lucky's perfect score.
    assert lucky["workload"] < 0.2


def test_shrinkage_high_n_keeps_own_mean():
    rows = [{"p": "Veteran", "d": "A", "topic_workload_score": 1.0} for _ in range(50)]
    rows += [
        {"p": f"Other{i}", "d": "A", "topic_workload_score": 0.0} for i in range(5)
    ]
    agg = aggregate_professor_scores(_reviews_with(rows), prior_strength=5.0)
    vet = agg[agg["professor_name"] == "Veteran"].iloc[0]
    # n=50, m=5 → 50/55 weight on own mean, still close to 1.0
    assert vet["workload"] > 0.9


def test_shrinkage_zero_prior_is_raw_mean():
    rows = [{"p": "Solo", "d": "A", "topic_workload_score": 1.0}]
    agg = aggregate_professor_scores(_reviews_with(rows), prior_strength=0.0)
    assert agg.iloc[0]["workload"] == 1.0


def test_aggregate_includes_department_and_num_reviews():
    rows = [
        {"p": "P1", "d": "STAT", "topic_workload_score": 0.5},
        {"p": "P1", "d": "STAT", "topic_workload_score": -0.5},
        {"p": "P2", "d": "CS", "topic_workload_score": 1.0},
    ]
    agg = aggregate_professor_scores(_reviews_with(rows), prior_strength=0.0)
    p1 = agg[agg["professor_name"] == "P1"].iloc[0]
    assert p1["department"] == "STAT"
    assert p1["num_reviews"] == 2
