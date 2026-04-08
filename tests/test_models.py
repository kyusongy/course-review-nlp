import pandas as pd
from src.models.baseline import star_to_sentiment, compute_baseline_scores


def test_star_to_sentiment():
    assert star_to_sentiment(1) == "negative"
    assert star_to_sentiment(2) == "negative"
    assert star_to_sentiment(3) == "neutral"
    assert star_to_sentiment(4) == "positive"
    assert star_to_sentiment(5) == "positive"


def test_compute_baseline_scores():
    df = pd.DataFrame(
        {
            "review_text": ["Great!", "Bad!", "Ok."],
            "star_rating": [5, 1, 3],
            "difficulty_rating": [2, 5, 3],
            "professor_name": ["A", "A", "A"],
        }
    )
    result = compute_baseline_scores(df)
    assert "sentiment" in result.columns
    assert "sentiment_score" in result.columns
    assert result.loc[0, "sentiment"] == "positive"
    assert result.loc[1, "sentiment"] == "negative"
    assert result.loc[2, "sentiment"] == "neutral"
    assert result.loc[0, "sentiment_score"] == 1.0
    assert result.loc[1, "sentiment_score"] == -1.0
    assert result.loc[2, "sentiment_score"] == 0.0
