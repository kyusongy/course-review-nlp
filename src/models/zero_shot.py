import re

from transformers import pipeline

TOPICS = [
    "Workload",
    "Grading",
    "Teaching Quality",
    "Accessibility",
    "Exam Difficulty",
]

# Descriptive labels improve zero-shot accuracy
TOPIC_DESCRIPTIONS = [
    "homework volume and time commitment",
    "grading fairness and grade distribution",
    "lecture quality and teaching effectiveness",
    "professor availability and approachability",
    "exam difficulty and test fairness",
]


class TopicClassifier:
    def __init__(self):
        self._pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )

    def classify(self, text: str, threshold: float = 0.3) -> list[str]:
        result = self._pipe(text, candidate_labels=TOPIC_DESCRIPTIONS, multi_label=True)
        topics = []
        for label, score in zip(result["labels"], result["scores"]):
            if score >= threshold:
                idx = TOPIC_DESCRIPTIONS.index(label)
                topics.append(TOPICS[idx])
        return topics

    def classify_with_scores(self, text: str) -> dict[str, float]:
        result = self._pipe(text, candidate_labels=TOPIC_DESCRIPTIONS, multi_label=True)
        scores = {}
        for label, score in zip(result["labels"], result["scores"]):
            idx = TOPIC_DESCRIPTIONS.index(label)
            scores[TOPICS[idx]] = score
        return scores


class SentimentAnalyzer:
    def __init__(self):
        self._pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
        )

    def analyze(self, text: str) -> dict:
        text = text[:512]
        results = self._pipe(text)[0]  # list of {label, score} dicts

        scores = {}
        for item in results:
            label = item["label"].lower()
            scores[label] = item["score"]

        pos = scores.get("positive", 0.0)
        neg = scores.get("negative", 0.0)
        composite = pos - neg

        if composite > 0.25:
            label = "positive"
        elif composite < -0.25:
            label = "negative"
        else:
            label = "neutral"

        return {"label": label, "score": composite}

    def analyze_by_topic(
        self, text: str, topic_clf: TopicClassifier
    ) -> dict[str, dict]:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        topic_data: dict[str, list[float]] = {t: [] for t in TOPICS}
        topic_sentences: dict[str, list[str]] = {t: [] for t in TOPICS}

        for sentence in sentences:
            matched = topic_clf.classify(sentence, threshold=0.25)
            if not matched:
                continue
            sentiment = self.analyze(sentence)
            for topic in matched:
                topic_data[topic].append(sentiment["score"])
                topic_sentences[topic].append(sentence)

        output = {}
        for topic in TOPICS:
            scores_list = topic_data[topic]
            if not scores_list:
                continue
            avg = sum(scores_list) / len(scores_list)
            if avg > 0.25:
                label = "positive"
            elif avg < -0.25:
                label = "negative"
            else:
                label = "neutral"
            output[topic] = {
                "label": label,
                "score": avg,
                "sentences": topic_sentences[topic],
            }

        return output

    def analyze_by_topic_flat(
        self, text: str, topic_clf: TopicClassifier
    ) -> dict[str, str]:
        """Like analyze_by_topic but returns {topic: sentiment_label} only."""
        full = self.analyze_by_topic(text, topic_clf)
        return {topic: data["label"] for topic, data in full.items()}
