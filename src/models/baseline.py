import pandas as pd

SENTIMENT_MAP = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}
SCORE_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def star_to_sentiment(rating: int) -> str:
    """Map 1-5 star rating to sentiment label."""
    return SENTIMENT_MAP[rating]


def compute_baseline_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'sentiment' and 'sentiment_score' columns based on star_rating. Returns copy."""
    result = df.copy()
    result["sentiment"] = result["star_rating"].map(SENTIMENT_MAP)
    result["sentiment_score"] = result["sentiment"].map(SCORE_MAP)
    return result
