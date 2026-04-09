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


from src.models.fine_tune import (
    TopicSentimentDataset,
    create_topic_labels,
    encode_topic_sentiments,
    decode_topic_sentiments,
)


def test_create_topic_labels_from_dict():
    labels = create_topic_labels({"Workload": "negative", "Grading": "positive"})
    assert len(labels) == len(TOPICS)
    assert labels[TOPICS.index("Workload")] == 1.0
    assert labels[TOPICS.index("Grading")] == 1.0
    assert labels[TOPICS.index("Teaching Quality")] == 0.0


def test_encode_topic_sentiments():
    states = encode_topic_sentiments(
        {"Teaching Quality": "positive", "Exam Difficulty": "negative"}
    )
    assert len(states) == 5
    assert states[TOPICS.index("Teaching Quality")] == 1  # positive
    assert states[TOPICS.index("Exam Difficulty")] == 3  # negative
    assert states[TOPICS.index("Workload")] == 0  # not discussed


def test_decode_topic_sentiments():
    # 5 topics: Workload, Grading, Teaching Quality, Accessibility, Exam Difficulty
    states = [0, 1, 3, 0, 2]
    result = decode_topic_sentiments(states)
    assert result == {
        "Grading": "positive",
        "Teaching Quality": "negative",
        "Exam Difficulty": "neutral",
    }


def test_topic_sentiment_dataset():
    texts = ["Great lectures but hard exams", "Easy class"]
    topic_sents = [
        {"Teaching Quality": "positive", "Exam Difficulty": "negative"},
        {"Workload": "positive"},
    ]
    ds = TopicSentimentDataset(texts, topic_sents)
    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["labels"].shape == (len(TOPICS),)  # 6 ints, one per topic


from src.models.evaluate import (
    compute_classification_metrics,
    compute_multilabel_metrics,
)


def test_compute_classification_metrics():
    y_true = ["positive", "negative", "neutral", "positive"]
    y_pred = ["positive", "negative", "positive", "positive"]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert metrics["accuracy"] == 0.75


def test_compute_multilabel_metrics():
    y_true = [[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]]
    y_pred = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 1]]
    metrics = compute_multilabel_metrics(y_true, y_pred)
    assert "f1_macro" in metrics
    assert "f1_per_topic" in metrics
    assert len(metrics["f1_per_topic"]) == 5
