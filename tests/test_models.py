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


import pytest
from src.models.zero_shot import TopicClassifier, SentimentAnalyzer, TOPICS


@pytest.fixture(scope="module")
def topic_clf():
    return TopicClassifier()


@pytest.fixture(scope="module")
def sentiment_ana():
    return SentimentAnalyzer()


def test_topic_classifier_returns_valid_topics(topic_clf):
    results = topic_clf.classify(
        "The homework was way too much but lectures were great."
    )
    assert isinstance(results, list)
    assert all(t in TOPICS for t in results)
    assert len(results) >= 1


def test_topic_classifier_threshold(topic_clf):
    results = topic_clf.classify("It was fine.", threshold=0.5)
    assert isinstance(results, list)


def test_sentiment_analyzer_returns_valid(sentiment_ana):
    result = sentiment_ana.analyze("This class was absolutely amazing!")
    assert result["label"] in ("positive", "neutral", "negative")
    assert -1.0 <= result["score"] <= 1.0


def test_sentiment_analyzer_negative(sentiment_ana):
    result = sentiment_ana.analyze("Terrible class, worst professor ever.")
    assert result["label"] == "negative"
    assert result["score"] < 0


import torch
from src.models.fine_tune import TopicDataset, SentimentDataset, create_topic_labels


def test_create_topic_labels():
    labels = create_topic_labels(["Workload", "Grading"])
    assert len(labels) == len(TOPICS)
    assert labels[TOPICS.index("Workload")] == 1.0
    assert labels[TOPICS.index("Grading")] == 1.0
    assert labels[TOPICS.index("Teaching Quality")] == 0.0


def test_topic_dataset():
    texts = ["Great class", "Hard exams"]
    topic_lists = [["Teaching Quality"], ["Exam Difficulty", "Grading"]]
    ds = TopicDataset(texts, topic_lists)
    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["labels"].shape == (len(TOPICS),)
