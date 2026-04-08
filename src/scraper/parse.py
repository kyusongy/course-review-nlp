"""Parse RMP GraphQL responses into standardized dicts."""

import pandas as pd


SCHOOL_NAME = "UNC Chapel Hill"

RATING_COLUMNS = [
    "review_text",
    "star_rating",
    "difficulty_rating",
    "would_take_again",
    "course_name",
    "professor_name",
    "date",
    "thumbs_up",
    "thumbs_down",
]


def parse_teacher(node: dict) -> dict:
    return {
        "rmp_id": node["id"],
        "legacy_id": node["legacyId"],
        "professor_name": f"{node['firstName']} {node['lastName']}",
        "department": node["department"],
        "avg_rating": node["avgRating"],
        "avg_difficulty": node["avgDifficulty"],
        "num_ratings": node["numRatings"],
        "would_take_again_pct": node["wouldTakeAgainPercent"],
        "school": SCHOOL_NAME,
    }


def parse_rating(node: dict, professor_name: str) -> dict:
    wta_raw = node["wouldTakeAgain"]
    if wta_raw is None:
        wta = None
    else:
        wta = bool(wta_raw)

    return {
        "review_text": node["comment"],
        "star_rating": node["qualityRating"],
        "difficulty_rating": node["difficultyRating"],
        "would_take_again": wta,
        "course_name": node["class"],
        "professor_name": professor_name,
        "date": node["date"],
        "thumbs_up": node["thumbsUpTotal"],
        "thumbs_down": node["thumbsDownTotal"],
    }


def ratings_to_dataframe(ratings: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(ratings, columns=RATING_COLUMNS)
    for col in ("star_rating", "difficulty_rating", "thumbs_up", "thumbs_down"):
        if col in df.columns and len(df) > 0:
            df[col] = df[col].astype(int)
    return df
