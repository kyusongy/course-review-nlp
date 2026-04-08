import re
import pandas as pd


def clean_text(text: str) -> str:
    if not text:
        return ""
    # collapse whitespace (spaces, newlines, tabs) to single space
    return re.sub(r"\s+", " ", text).strip()


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_text"] = df["review_text"].apply(
        lambda x: clean_text(x) if isinstance(x, str) else ""
    )
    df = df[df["review_text"] != ""]
    df = df.drop_duplicates(subset=["review_text", "professor_name", "course_name"])
    return df.reset_index(drop=True)
