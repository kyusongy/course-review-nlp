import pandas as pd
from src.recommend.engine import score_professors, filter_results


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
