"""Tests for src/scraper/parse.py"""

import pytest
import pandas as pd
from src.scraper.parse import parse_teacher, parse_rating, ratings_to_dataframe


# --- Sample data matching confirmed RMP GraphQL schema ---

TEACHER_NODE = {
    "id": "VGVhY2hlci0yNDEyMTM3",
    "legacyId": 2412137,
    "firstName": "Jane",
    "lastName": "Smith",
    "department": "Statistics",
    "avgRating": 4.2,
    "numRatings": 85,
    "avgDifficulty": 3.1,
    "wouldTakeAgainPercent": 88.5,
}

RATING_NODE_WTA_ONE = {
    "id": "UmF0aW5nLTEyMzQ1",
    "comment": "Great professor, very clear explanations.",
    "qualityRating": 5,
    "difficultyRating": 3,
    "class": "STOR 155",
    "date": "2024-03-15 00:00:00 +0000 UTC",
    "thumbsUpTotal": 7,
    "thumbsDownTotal": 1,
    "wouldTakeAgain": 1,
    "isForOnlineClass": False,
}

RATING_NODE_WTA_ZERO = {
    "id": "UmF0aW5nLTU2Nzg5",
    "comment": "Hard grader but fair.",
    "qualityRating": 3,
    "difficultyRating": 4,
    "class": "STOR 435",
    "date": "2024-01-20 00:00:00 +0000 UTC",
    "thumbsUpTotal": 2,
    "thumbsDownTotal": 3,
    "wouldTakeAgain": 0,
    "isForOnlineClass": False,
}

RATING_NODE_WTA_NULL = {
    "id": "UmF0aW5nLTk5OTk5",
    "comment": "Decent course.",
    "qualityRating": 4,
    "difficultyRating": 2,
    "class": "STOR 215",
    "date": "2023-11-05 00:00:00 +0000 UTC",
    "thumbsUpTotal": 0,
    "thumbsDownTotal": 0,
    "wouldTakeAgain": None,
    "isForOnlineClass": True,
}


# --- parse_teacher ---


def test_parse_teacher_extracts_all_fields():
    result = parse_teacher(TEACHER_NODE)
    assert result["rmp_id"] == "VGVhY2hlci0yNDEyMTM3"
    assert result["legacy_id"] == 2412137
    assert result["professor_name"] == "Jane Smith"
    assert result["department"] == "Statistics"
    assert result["avg_rating"] == 4.2
    assert result["avg_difficulty"] == 3.1
    assert result["num_ratings"] == 85
    assert result["would_take_again_pct"] == 88.5
    assert result["school"] == "UNC Chapel Hill"


def test_parse_teacher_name_concat():
    node = {**TEACHER_NODE, "firstName": "Bob", "lastName": "Jones"}
    result = parse_teacher(node)
    assert result["professor_name"] == "Bob Jones"


def test_parse_teacher_school_always_unc():
    result = parse_teacher(TEACHER_NODE)
    assert result["school"] == "UNC Chapel Hill"


# --- parse_rating ---


def test_parse_rating_wta_one():
    result = parse_rating(RATING_NODE_WTA_ONE, "Jane Smith")
    assert result["review_text"] == "Great professor, very clear explanations."
    assert result["star_rating"] == 5
    assert result["difficulty_rating"] == 3
    assert result["would_take_again"] is True
    assert result["course_name"] == "STOR 155"
    assert result["professor_name"] == "Jane Smith"
    assert result["date"] == "2024-03-15 00:00:00 +0000 UTC"
    assert result["thumbs_up"] == 7
    assert result["thumbs_down"] == 1


def test_parse_rating_wta_zero():
    result = parse_rating(RATING_NODE_WTA_ZERO, "Jane Smith")
    assert result["would_take_again"] is False


def test_parse_rating_wta_null():
    result = parse_rating(RATING_NODE_WTA_NULL, "Jane Smith")
    assert result["would_take_again"] is None


def test_parse_rating_all_fields_present():
    result = parse_rating(RATING_NODE_WTA_ONE, "Jane Smith")
    expected_keys = {
        "review_text",
        "star_rating",
        "difficulty_rating",
        "would_take_again",
        "course_name",
        "professor_name",
        "date",
        "thumbs_up",
        "thumbs_down",
    }
    assert set(result.keys()) == expected_keys


# --- ratings_to_dataframe ---

RATINGS = [
    parse_rating(RATING_NODE_WTA_ONE, "Jane Smith"),
    parse_rating(RATING_NODE_WTA_ZERO, "Jane Smith"),
    parse_rating(RATING_NODE_WTA_NULL, "Jane Smith"),
]

EXPECTED_COLUMNS = [
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


def test_ratings_to_dataframe_columns():
    df = ratings_to_dataframe(RATINGS)
    assert list(df.columns) == EXPECTED_COLUMNS


def test_ratings_to_dataframe_row_count():
    df = ratings_to_dataframe(RATINGS)
    assert len(df) == 3


def test_ratings_to_dataframe_dtypes():
    df = ratings_to_dataframe(RATINGS)
    assert df["star_rating"].dtype == int
    assert df["difficulty_rating"].dtype == int
    assert df["thumbs_up"].dtype == int
    assert df["thumbs_down"].dtype == int
    # pandas 2.x may use StringDtype instead of object for string columns
    assert df["review_text"].dtype.kind in ("O", "U") or hasattr(
        df["review_text"].dtype, "na_value"
    )
    assert df["professor_name"].dtype.kind in ("O", "U") or hasattr(
        df["professor_name"].dtype, "na_value"
    )


def test_ratings_to_dataframe_wta_preserves_none():
    df = ratings_to_dataframe(RATINGS)
    # Third row has wouldTakeAgain=None — should be NaN/None in df, not False
    assert pd.isna(df.iloc[2]["would_take_again"])


def test_ratings_to_dataframe_wta_bool_values():
    df = ratings_to_dataframe(RATINGS)
    assert df.iloc[0]["would_take_again"] is True
    assert df.iloc[1]["would_take_again"] is False


def test_ratings_to_dataframe_empty():
    df = ratings_to_dataframe([])
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 0
