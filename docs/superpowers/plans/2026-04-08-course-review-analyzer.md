# UNC Course Compass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an ML pipeline that scrapes UNC Stats & DS professor reviews from RateMyProfessor, classifies topics and sentiment using three approaches (star proxy, zero-shot, fine-tuned), recommends courses based on preferences, and presents results in a Streamlit app.

**Architecture:** Zero-shot baseline → fine-tune comparison (Approach C). Three-layer sentiment analysis: star ratings as proxy baseline, zero-shot transformer models (BART-mnli + RoBERTa), and fine-tuned DistilBERT. Weighted scoring recommendation engine. Streamlit frontend with explore, recommend, and model comparison views.

**Tech Stack:** Python 3.11+, uv (package management), PyTorch (MPS), HuggingFace Transformers, httpx, pandas, scikit-learn, Streamlit, plotly

---

## File Structure

```
course_review/
├── pyproject.toml              # project metadata + dependencies
├── src/
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── client.py           # RMP GraphQL API client (search school, teachers, ratings)
│   │   └── parse.py            # parse GraphQL responses → dicts/DataFrames
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py         # star_to_sentiment(), compute_baseline_scores()
│   │   ├── zero_shot.py        # TopicClassifier, SentimentAnalyzer using pretrained models
│   │   ├── fine_tune.py        # TopicDataset, SentimentDataset, train_*, predict_*
│   │   └── evaluate.py         # compute_metrics(), compare_approaches()
│   ├── recommend/
│   │   ├── __init__.py
│   │   └── engine.py           # score_professors(), filter_results()
│   └── app/
│       ├── __init__.py
│       └── streamlit_app.py    # 3-tab Streamlit frontend
├── tests/
│   ├── test_scraper.py
│   ├── test_models.py
│   └── test_recommend.py
├── notebooks/
│   ├── eda.ipynb
│   ├── labeling.ipynb
│   └── experiments.ipynb
├── data/
│   ├── raw/                    # JSON per professor
│   ├── processed/              # reviews.parquet
│   └── labels/                 # hand-corrected annotations
├── models/                     # saved fine-tuned model weights
└── docs/
    └── superpowers/
        ├── specs/
        └── plans/
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: all `__init__.py` files
- Create: `.gitignore`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/kyle/projects/course_review
git init
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
data/raw/
data/processed/
data/labels/
models/
*.egg-info/
dist/
.ipynb_checkpoints/
.DS_Store
```

Note: `data/` subdirs are ignored (large/generated). Only `data/.gitkeep` files are tracked.

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "course-review"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27",
    "pandas>=2.2",
    "pyarrow>=15.0",
    "torch>=2.2",
    "transformers>=4.40",
    "scikit-learn>=1.4",
    "streamlit>=1.33",
    "plotly>=5.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ipykernel>=6.29",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 4: Create directory structure and `__init__.py` files**

Create these empty files:
- `src/__init__.py`
- `src/scraper/__init__.py`
- `src/models/__init__.py`
- `src/recommend/__init__.py`
- `src/app/__init__.py`

Create `.gitkeep` in:
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `data/labels/.gitkeep`
- `models/.gitkeep`
- `notebooks/.gitkeep`

- [ ] **Step 5: Initialize uv project and install**

```bash
uv sync --extra dev
```

This creates `.venv` automatically and installs all dependencies including dev extras.

- [ ] **Step 6: Verify setup**

```bash
uv run python -c "import torch; print(torch.backends.mps.is_available())"
uv run python -c "import transformers; print(transformers.__version__)"
uv run pytest --co  # collect tests (none yet, should exit clean)
```

Expected: MPS available = True, transformers version printed, pytest exits 0.

- [ ] **Step 7: Commit**

```bash
git add .
git commit -m "Scaffold project structure and dependencies"
```

---

### Task 2: RMP Scraper (Hybrid Approach)

**Strategy:** Use Claude Code's WebFetch to explore RMP's current page structure (Next.js `__NEXT_DATA__` or GraphQL API), then write a script that automates bulk collection based on what we find.

**Files:**
- Create: `src/scraper/client.py`
- Create: `src/scraper/parse.py`
- Create: `tests/test_scraper.py`

- [ ] **Step 1: Explore RMP structure with Claude Code**

Use WebFetch to fetch a known UNC professor's RMP page (e.g. search for "University of North Carolina at Chapel Hill" on RMP). Inspect the response for:
- `__NEXT_DATA__` script tag (Next.js SSR data — contains teacher + ratings JSON)
- GraphQL API endpoint and auth token (may be embedded in page source)
- Data schema: what fields are available for teachers and ratings

Document what you find — the actual field names, data structure, and access method determine the parser code.

- [ ] **Step 2: Write parser tests based on discovered schema**

After exploring, write tests using the **actual data shape** found in Step 1. The test fixtures below are templates — update field names and structure to match reality:

```python
# tests/test_scraper.py
import pytest
from src.scraper.parse import parse_teacher, parse_rating, ratings_to_dataframe

# UPDATE THESE FIXTURES with actual field names from Step 1
SAMPLE_TEACHER_NODE = {
    # Fields discovered from RMP page exploration
    # e.g. "id", "firstName", "lastName", "department", "avgRating", etc.
}

SAMPLE_RATING_NODE = {
    # Fields discovered from RMP page exploration
    # e.g. "comment", "qualityRating", "difficultyRating", "class", "date", etc.
}


def test_parse_teacher():
    result = parse_teacher(SAMPLE_TEACHER_NODE)
    assert "professor_name" in result
    assert "department" in result
    assert "avg_rating" in result
    assert "num_ratings" in result


def test_parse_rating():
    result = parse_rating(SAMPLE_RATING_NODE, professor_name="Jane Doe")
    assert "review_text" in result
    assert "star_rating" in result
    assert "difficulty_rating" in result
    assert "course_name" in result
    assert "professor_name" in result


def test_ratings_to_dataframe():
    ratings = [
        parse_rating(SAMPLE_RATING_NODE, professor_name="Jane Doe"),
        parse_rating(SAMPLE_RATING_NODE, professor_name="Jane Doe"),
    ]
    df = ratings_to_dataframe(ratings)
    assert len(df) == 2
    expected_cols = [
        "review_text", "star_rating", "difficulty_rating",
        "would_take_again", "course_name", "professor_name",
        "date", "thumbs_up", "thumbs_down",
    ]
    assert list(df.columns) == expected_cols
```

- [ ] **Step 3: Run tests — verify they fail**

```bash
uv run pytest tests/test_scraper.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement `src/scraper/parse.py`**

Adapt the parser to match the actual data structure found in Step 1. Target output schema is fixed (our downstream code depends on it):

```python
# src/scraper/parse.py
"""Parse RMP data into standardized format.

Output schema is fixed — all parsers must produce these fields:
  Teacher: rmp_id, legacy_id, professor_name, department, avg_rating,
           avg_difficulty, num_ratings, would_take_again_pct, school
  Rating:  review_text, star_rating, difficulty_rating, would_take_again,
           course_name, professor_name, date, thumbs_up, thumbs_down
"""
import pandas as pd


def parse_teacher(node: dict) -> dict:
    """Parse a teacher node into standardized format.
    Adapt field access based on actual RMP data structure.
    """
    return {
        "rmp_id": node.get("id", ""),
        "legacy_id": node.get("legacyId", ""),
        "professor_name": f"{node['firstName']} {node['lastName']}",
        "department": node.get("department", ""),
        "avg_rating": node.get("avgRating", 0),
        "avg_difficulty": node.get("avgDifficulty", 0),
        "num_ratings": node.get("numRatings", 0),
        "would_take_again_pct": node.get("wouldTakeAgainPercent", -1),
        "school": node.get("school", {}).get("name", ""),
    }


def parse_rating(node: dict, professor_name: str) -> dict:
    """Parse a rating node into standardized format.
    Adapt field access based on actual RMP data structure.
    """
    return {
        "review_text": node.get("comment", ""),
        "star_rating": int(node.get("qualityRating", 0)),
        "difficulty_rating": int(node.get("difficultyRating", 0)),
        "would_take_again": node.get("wouldTakeAgain", 0) == 1,
        "course_name": node.get("class", ""),
        "professor_name": professor_name,
        "date": node.get("date", ""),
        "thumbs_up": node.get("thumbsUpTotal", 0),
        "thumbs_down": node.get("thumbsDownTotal", 0),
    }


REVIEW_COLUMNS = [
    "review_text", "star_rating", "difficulty_rating",
    "would_take_again", "course_name", "professor_name",
    "date", "thumbs_up", "thumbs_down",
]


def ratings_to_dataframe(ratings: list[dict]) -> pd.DataFrame:
    """Convert list of parsed ratings to a DataFrame."""
    df = pd.DataFrame(ratings, columns=REVIEW_COLUMNS)
    df["star_rating"] = df["star_rating"].astype(int)
    df["difficulty_rating"] = df["difficulty_rating"].astype(int)
    return df
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/test_scraper.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Implement `src/scraper/client.py` based on discovered access method**

Write the client based on what Step 1 revealed. Two likely paths:

**Path A — GraphQL API works:** Write async httpx client hitting the GraphQL endpoint with discovered auth token and query structure.

**Path B — `__NEXT_DATA__` parsing:** Write httpx client that fetches professor HTML pages and extracts the `__NEXT_DATA__` JSON from `<script id="__NEXT_DATA__">` tags.

Either way, the client must:
- Accept a school name and department filter
- Return list of teacher dicts and their rating dicts
- Cache raw JSON to `data/raw/{legacy_id}.json`
- Handle pagination (GraphQL cursors or page numbers)
- Include retry logic and 0.5s delay between requests

```python
# src/scraper/client.py
"""RateMyProfessor scraper client.

Access method (GraphQL vs __NEXT_DATA__) determined during
Claude Code exploration in Task 2 Step 1.
"""
import json
from pathlib import Path

import httpx

# TODO: Fill in after Step 1 exploration reveals the actual access method.
# The implementation will follow one of two paths:
#   Path A: GraphQL client with discovered auth token + queries
#   Path B: HTML fetcher with __NEXT_DATA__ JSON extraction
#
# This file is intentionally left as a skeleton to be completed
# during implementation after the exploration step.
```

Note: the actual `client.py` implementation will be written during execution after the WebFetch exploration reveals the current RMP structure. The plan cannot hardcode API details that may have changed.

- [ ] **Step 7: Commit**

```bash
git add src/scraper/ tests/test_scraper.py
git commit -m "Add RMP scraper parser and client skeleton"
```

---

### Task 3: Data Collection (Hybrid)

**Files:**
- Create: `src/scraper/run.py`
- Uses: `src/scraper/client.py`, `src/scraper/parse.py`
- Creates: `data/raw/*.json`, `data/processed/reviews.parquet`

- [ ] **Step 1: Claude Code explores RMP professor pages**

Use WebFetch to fetch a few UNC Stats professor pages. Identify the list of professors in the department (search RMP for UNC Chapel Hill → Statistics). Record professor URLs/IDs found.

- [ ] **Step 2: Claude Code extracts sample data**

For 2-3 professors, use WebFetch to fetch their full pages. Extract and save the raw JSON data (from `__NEXT_DATA__` or GraphQL responses) to `data/raw/`. Verify the data contains: reviews, ratings, course names, dates.

- [ ] **Step 3: Write `src/scraper/run.py` based on discovered structure**

Write a script that automates what Claude Code did manually — iterate through all professors, fetch pages, extract and parse data, save to parquet. Use the actual URL patterns and data paths discovered in Steps 1-2.

```python
# src/scraper/run.py
"""Run the RMP scraper for UNC Stats & DS department.

Uses the access method discovered during Claude Code exploration.
"""
from pathlib import Path

import pandas as pd

from src.scraper.client import RMPClient
from src.scraper.parse import parse_teacher, parse_rating, ratings_to_dataframe

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def main():
    client = RMPClient(cache_dir=RAW_DIR)
    teachers, ratings_by_teacher = client.scrape_department(
        school_name="University of North Carolina at Chapel Hill",
        dept_query="Statistics",
    )

    all_ratings = []
    teacher_records = []
    for teacher in teachers:
        tid = teacher.get("id") or teacher.get("legacyId")
        prof_name = teacher.get("professor_name") or f"{teacher['firstName']} {teacher['lastName']}"
        teacher_records.append(parse_teacher(teacher))
        for rating in ratings_by_teacher.get(str(tid), []):
            text = rating.get("comment", "")
            if text:
                all_ratings.append(parse_rating(rating, professor_name=prof_name))

    reviews_df = ratings_to_dataframe(all_ratings)
    teachers_df = pd.DataFrame(teacher_records)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reviews_df.to_parquet(PROCESSED_DIR / "reviews.parquet", index=False)
    teachers_df.to_parquet(PROCESSED_DIR / "teachers.parquet", index=False)

    print(f"\nScraped {len(teachers)} professors, {len(reviews_df)} reviews")
    print(f"Saved to {PROCESSED_DIR}")
    print(f"\nTop professors by review count:")
    print(reviews_df.groupby("professor_name").size().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the scraper**

```bash
uv run python -m src.scraper.run
```

Expected: prints professor names with rating counts, saves JSON to `data/raw/` and parquet to `data/processed/`.

If the script fails, fall back to Claude Code doing the fetching directly for remaining professors, or use a Kaggle RMP dataset.

- [ ] **Step 5: Verify data**

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/processed/reviews.parquet')
print(f'Reviews: {len(df)}')
print(f'Professors: {df.professor_name.nunique()}')
print(f'Columns: {list(df.columns)}')
print(f'\nSample review:\n{df.review_text.iloc[0][:200]}')
print(f'\nStar rating distribution:\n{df.star_rating.value_counts().sort_index()}')
"
```

Expected: 500+ reviews, 20+ professors, all columns present.

- [ ] **Step 6: Commit**

```bash
git add src/scraper/
git commit -m "Add RMP scraper and collect UNC Stats data"
```

---

### Task 4: Data Preprocessing

**Files:**
- Create: `src/scraper/preprocess.py`
- Modify: `tests/test_scraper.py` (add preprocessing tests)

- [ ] **Step 1: Write preprocessing tests**

Append to `tests/test_scraper.py`:

```python
from src.scraper.preprocess import clean_text, preprocess_reviews
import pandas as pd


def test_clean_text():
    assert clean_text("  Hello\n\nWorld  ") == "Hello World"
    assert clean_text("") == ""
    assert clean_text("This is GREAT!!!") == "This is GREAT!!!"


def test_preprocess_reviews():
    df = pd.DataFrame({
        "review_text": ["Great class!", "Great class!", "", "  Loved it  "],
        "star_rating": [5, 5, 3, 4],
        "difficulty_rating": [2, 2, 3, 1],
        "would_take_again": [True, True, False, True],
        "course_name": ["STOR 435", "STOR 435", "STOR 155", "STOR 435"],
        "professor_name": ["Jane Doe", "Jane Doe", "Jane Doe", "Jane Doe"],
        "date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-03-01"],
        "thumbs_up": [5, 5, 0, 3],
        "thumbs_down": [0, 0, 1, 0],
    })
    result = preprocess_reviews(df)
    # Should drop empty reviews and duplicates
    assert len(result) == 2
    assert result["review_text"].iloc[0] == "Great class!"
    assert result["review_text"].iloc[1] == "Loved it"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_scraper.py::test_clean_text tests/test_scraper.py::test_preprocess_reviews -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/scraper/preprocess.py`**

```python
# src/scraper/preprocess.py
"""Clean and preprocess scraped review data."""
import re

import pandas as pd


def clean_text(text: str) -> str:
    """Normalize whitespace, strip leading/trailing space."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text, drop empties and duplicates."""
    df = df.copy()
    df["review_text"] = df["review_text"].apply(clean_text)
    df = df[df["review_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["review_text", "professor_name", "course_name"])
    df = df.reset_index(drop=True)
    return df
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_scraper.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Run preprocessing on scraped data**

```bash
uv run python -c "
import pandas as pd
from src.scraper.preprocess import preprocess_reviews

df = pd.read_parquet('data/processed/reviews.parquet')
print(f'Before: {len(df)} reviews')
df = preprocess_reviews(df)
print(f'After: {len(df)} reviews')
df.to_parquet('data/processed/reviews.parquet', index=False)
print('Saved cleaned data.')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/scraper/preprocess.py tests/test_scraper.py
git commit -m "Add data preprocessing with cleaning and dedup"
```

---

### Task 5: Star Rating Baseline

**Files:**
- Create: `src/models/baseline.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write baseline tests**

```python
# tests/test_models.py
import pandas as pd
from src.models.baseline import star_to_sentiment, compute_baseline_scores


def test_star_to_sentiment():
    assert star_to_sentiment(1) == "negative"
    assert star_to_sentiment(2) == "negative"
    assert star_to_sentiment(3) == "neutral"
    assert star_to_sentiment(4) == "positive"
    assert star_to_sentiment(5) == "positive"


def test_compute_baseline_scores():
    df = pd.DataFrame({
        "review_text": ["Great!", "Bad!", "Ok."],
        "star_rating": [5, 1, 3],
        "difficulty_rating": [2, 5, 3],
        "professor_name": ["A", "A", "A"],
    })
    result = compute_baseline_scores(df)
    assert "sentiment" in result.columns
    assert "sentiment_score" in result.columns
    assert result.loc[0, "sentiment"] == "positive"
    assert result.loc[1, "sentiment"] == "negative"
    assert result.loc[2, "sentiment"] == "neutral"
    # sentiment_score maps to [-1, 0, 1]
    assert result.loc[0, "sentiment_score"] == 1.0
    assert result.loc[1, "sentiment_score"] == -1.0
    assert result.loc[2, "sentiment_score"] == 0.0
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/models/baseline.py`**

```python
# src/models/baseline.py
"""Star rating proxy baseline for sentiment."""
import pandas as pd

SENTIMENT_MAP = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
SCORE_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


def star_to_sentiment(rating: int) -> str:
    """Map 1-5 star rating to sentiment label."""
    return SENTIMENT_MAP[rating]


def compute_baseline_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment and sentiment_score columns based on star_rating."""
    df = df.copy()
    df["sentiment"] = df["star_rating"].map(SENTIMENT_MAP)
    df["sentiment_score"] = df["sentiment"].map(SCORE_MAP)
    return df
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_models.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/models/baseline.py tests/test_models.py
git commit -m "Add star rating baseline sentiment model"
```

---

### Task 6: Zero-Shot Pipeline

**Files:**
- Create: `src/models/zero_shot.py`
- Modify: `tests/test_models.py` (add zero-shot tests)

- [ ] **Step 1: Write zero-shot tests**

Append to `tests/test_models.py`:

```python
import pytest
from src.models.zero_shot import TopicClassifier, SentimentAnalyzer, TOPICS


@pytest.fixture(scope="module")
def topic_clf():
    return TopicClassifier()


@pytest.fixture(scope="module")
def sentiment_ana():
    return SentimentAnalyzer()


def test_topic_classifier_returns_valid_topics(topic_clf):
    results = topic_clf.classify("The homework was way too much but lectures were great.")
    assert isinstance(results, list)
    assert all(t in TOPICS for t in results)
    assert len(results) >= 1


def test_topic_classifier_threshold(topic_clf):
    # Very generic text should return fewer topics
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_models.py::test_topic_classifier_returns_valid_topics -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `src/models/zero_shot.py`**

```python
# src/models/zero_shot.py
"""Zero-shot topic classification and sentiment analysis."""
import re

from transformers import pipeline

TOPICS = [
    "Workload",
    "Grading",
    "Teaching Quality",
    "Course Content",
    "Accessibility",
    "Exam Difficulty",
]

# Descriptive labels improve zero-shot accuracy
TOPIC_DESCRIPTIONS = [
    "homework volume and time commitment",
    "grading fairness and grade distribution",
    "lecture quality and teaching effectiveness",
    "course material relevance and organization",
    "professor availability and approachability",
    "exam difficulty and test fairness",
]


class TopicClassifier:
    """Zero-shot topic classification using BART-large-mnli."""

    def __init__(self, model: str = "facebook/bart-large-mnli"):
        self.pipe = pipeline("zero-shot-classification", model=model)

    def classify(self, text: str, threshold: float = 0.3) -> list[str]:
        """Classify text into topics above confidence threshold."""
        result = self.pipe(text, TOPIC_DESCRIPTIONS, multi_label=True)
        topics = []
        for label, score in zip(result["labels"], result["scores"]):
            if score >= threshold:
                idx = TOPIC_DESCRIPTIONS.index(label)
                topics.append(TOPICS[idx])
        return topics

    def classify_with_scores(self, text: str) -> dict[str, float]:
        """Return all topics with their confidence scores."""
        result = self.pipe(text, TOPIC_DESCRIPTIONS, multi_label=True)
        return {
            TOPICS[TOPIC_DESCRIPTIONS.index(label)]: score
            for label, score in zip(result["labels"], result["scores"])
        }


class SentimentAnalyzer:
    """Sentiment analysis using pretrained RoBERTa."""

    LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    SCORE_MAP = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

    def __init__(self, model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.pipe = pipeline("sentiment-analysis", model=model, top_k=None)

    def analyze(self, text: str) -> dict:
        """Analyze sentiment of text. Returns {label, score}."""
        results = self.pipe(text[:512])[0]  # truncate to model max
        # Get the top label
        top = max(results, key=lambda x: x["score"])
        label = self.LABEL_MAP.get(top["label"], top["label"])
        # Compute weighted score in [-1, 1]
        score_map = {r["label"]: r["score"] for r in results}
        neg = score_map.get("LABEL_0", score_map.get("negative", 0))
        pos = score_map.get("LABEL_2", score_map.get("positive", 0))
        score = pos - neg  # range [-1, 1]
        return {"label": label, "score": round(score, 4)}

    def analyze_by_topic(self, text: str, topic_clf: TopicClassifier) -> dict[str, dict]:
        """Aspect-based sentiment: split into sentences, classify topic, score sentiment.

        Returns {topic: {label, score, sentences}} for each detected topic.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        topic_sentiments = {}

        for sent in sentences:
            if len(sent.strip()) < 5:
                continue
            topics = topic_clf.classify(sent, threshold=0.3)
            if not topics:
                continue
            sentiment = self.analyze(sent)
            for topic in topics:
                if topic not in topic_sentiments:
                    topic_sentiments[topic] = {"scores": [], "sentences": []}
                topic_sentiments[topic]["scores"].append(sentiment["score"])
                topic_sentiments[topic]["sentences"].append(sent)

        # Aggregate per topic
        result = {}
        for topic, data in topic_sentiments.items():
            avg_score = sum(data["scores"]) / len(data["scores"])
            label = "positive" if avg_score > 0.25 else "negative" if avg_score < -0.25 else "neutral"
            result[topic] = {
                "label": label,
                "score": round(avg_score, 4),
                "sentences": data["sentences"],
            }
        return result
```

- [ ] **Step 4: Run all zero-shot tests**

```bash
uv run pytest tests/test_models.py -v -k "zero_shot or topic or sentiment_analyzer"
```

Expected: 4 passed (first run will download models — ~1.5GB total, takes a few minutes).

- [ ] **Step 5: Smoke test on real data**

```bash
uv run python -c "
import pandas as pd
from src.models.zero_shot import TopicClassifier, SentimentAnalyzer

df = pd.read_parquet('data/processed/reviews.parquet')
tc = TopicClassifier()
sa = SentimentAnalyzer()

# Test on 3 sample reviews
for _, row in df.head(3).iterrows():
    print(f'Review: {row.review_text[:100]}...')
    topics = tc.classify(row.review_text)
    sent = sa.analyze(row.review_text)
    print(f'  Topics: {topics}')
    print(f'  Sentiment: {sent}')
    print()
"
```

Expected: each review gets 1+ topics and a sentiment label/score. Verify they look reasonable.

- [ ] **Step 6: Commit**

```bash
git add src/models/zero_shot.py tests/test_models.py
git commit -m "Add zero-shot topic and sentiment pipeline"
```

---

### Task 7: Process All Reviews with Zero-Shot

**Files:**
- Create: `src/models/process.py`

This produces the scored dataset used by the recommendation engine and as noisy labels for fine-tuning.

- [ ] **Step 1: Create batch processing script**

```python
# src/models/process.py
"""Batch-process all reviews through the model pipelines."""
import json
from pathlib import Path

import pandas as pd

from src.models.baseline import compute_baseline_scores
from src.models.zero_shot import TopicClassifier, SentimentAnalyzer, TOPICS


def process_all(reviews_path: Path, output_dir: Path):
    """Run all three approaches on the review dataset."""
    df = pd.read_parquet(reviews_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Layer 1: Star rating baseline
    print("Computing star rating baseline...")
    baseline_df = compute_baseline_scores(df)
    baseline_df.to_parquet(output_dir / "baseline_scores.parquet", index=False)

    # Layer 2: Zero-shot
    print("Running zero-shot pipeline (this takes a while)...")
    tc = TopicClassifier()
    sa = SentimentAnalyzer()

    records = []
    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"  Processing review {i}/{len(df)}...")
        text = row["review_text"]

        # Topic classification
        topic_scores = tc.classify_with_scores(text)

        # Aspect-based sentiment
        topic_sentiments = sa.analyze_by_topic(text, tc)

        # Overall sentiment
        overall = sa.analyze(text)

        record = {
            "idx": i,
            "professor_name": row["professor_name"],
            "course_name": row["course_name"],
            "review_text": text,
            "star_rating": row["star_rating"],
            "overall_sentiment": overall["label"],
            "overall_score": overall["score"],
        }
        # Add per-topic fields
        for topic in TOPICS:
            key = topic.lower().replace(" ", "_")
            record[f"topic_{key}_conf"] = topic_scores.get(topic, 0.0)
            if topic in topic_sentiments:
                record[f"topic_{key}_sentiment"] = topic_sentiments[topic]["label"]
                record[f"topic_{key}_score"] = topic_sentiments[topic]["score"]
            else:
                record[f"topic_{key}_sentiment"] = None
                record[f"topic_{key}_score"] = None

        records.append(record)

    results_df = pd.DataFrame(records)
    results_df.to_parquet(output_dir / "zero_shot_scores.parquet", index=False)
    print(f"Saved zero-shot scores for {len(results_df)} reviews.")
    return results_df


if __name__ == "__main__":
    process_all(
        reviews_path=Path("data/processed/reviews.parquet"),
        output_dir=Path("data/processed"),
    )
```

- [ ] **Step 2: Run batch processing**

```bash
uv run python -m src.models.process
```

Expected: processes all reviews, prints progress, saves `baseline_scores.parquet` and `zero_shot_scores.parquet` to `data/processed/`. This will take 10-30 min depending on dataset size.

- [ ] **Step 3: Verify outputs**

```bash
uv run python -c "
import pandas as pd
zs = pd.read_parquet('data/processed/zero_shot_scores.parquet')
print(f'Zero-shot scores: {len(zs)} reviews')
print(f'Columns: {[c for c in zs.columns if \"topic\" in c][:6]}')
print(f'\nOverall sentiment distribution:')
print(zs['overall_sentiment'].value_counts())
print(f'\nTeaching quality topic detected in {zs.topic_teaching_quality_conf.gt(0.3).sum()} reviews')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/models/process.py
git commit -m "Add batch processing for zero-shot pipeline"
```

---

### Task 8: Labeling Workflow

**Files:**
- Create: `notebooks/labeling.ipynb` (or `src/models/labeling.py` for script-based approach)

- [ ] **Step 1: Create labeling script**

Create `src/models/labeling.py` — a simple CLI labeling tool that shows zero-shot predictions and lets you correct them:

```python
# src/models/labeling.py
"""Simple CLI labeling tool for topic + sentiment annotations."""
import json
from pathlib import Path

import pandas as pd

from src.models.zero_shot import TOPICS

LABELS_DIR = Path("data/labels")
LABELS_FILE = LABELS_DIR / "annotations.json"

SENTIMENT_OPTIONS = ["positive", "neutral", "negative"]


def load_annotations() -> dict:
    """Load existing annotations."""
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text())
    return {}


def save_annotations(annotations: dict):
    """Save annotations to disk."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_FILE.write_text(json.dumps(annotations, indent=2))


def label_reviews(scores_path: Path, n: int = 200):
    """Interactive labeling loop.

    Shows zero-shot predictions, lets you accept or correct.
    Type 'q' to quit and save progress.
    """
    df = pd.read_parquet(scores_path)
    annotations = load_annotations()
    labeled_ids = set(annotations.keys())

    # Prioritize reviews not yet labeled
    unlabeled = df[~df["idx"].astype(str).isin(labeled_ids)]
    if len(unlabeled) == 0:
        print("All reviews labeled!")
        return

    print(f"\n{len(labeled_ids)} already labeled, {len(unlabeled)} remaining.")
    print(f"Labeling up to {n} reviews. Type 'q' to quit.\n")

    count = 0
    for _, row in unlabeled.iterrows():
        if count >= n:
            break

        idx = str(int(row["idx"]))
        text = row["review_text"]

        # Show zero-shot predictions
        print(f"\n{'='*60}")
        print(f"[{idx}] {text[:300]}")
        print(f"{'='*60}")

        # Topics
        predicted_topics = []
        for topic in TOPICS:
            key = topic.lower().replace(" ", "_")
            conf = row.get(f"topic_{key}_conf", 0)
            if conf > 0.3:
                predicted_topics.append(topic)

        print(f"\nPredicted topics: {predicted_topics}")
        print(f"All topics: {[f'{i}={t}' for i, t in enumerate(TOPICS)]}")

        resp = input("Topics (enter numbers comma-sep, or 'a' to accept, 'q' to quit): ").strip()
        if resp == "q":
            break
        elif resp == "a":
            topics = predicted_topics
        else:
            topics = [TOPICS[int(i)] for i in resp.split(",") if i.strip().isdigit()]

        # Sentiment
        predicted_sent = row.get("overall_sentiment", "neutral")
        print(f"\nPredicted sentiment: {predicted_sent}")
        print("0=positive, 1=neutral, 2=negative")
        resp = input("Sentiment (0/1/2, or 'a' to accept, 'q' to quit): ").strip()
        if resp == "q":
            break
        elif resp == "a":
            sentiment = predicted_sent
        else:
            sentiment = SENTIMENT_OPTIONS[int(resp)]

        annotations[idx] = {"topics": topics, "sentiment": sentiment}
        save_annotations(annotations)
        count += 1
        print(f"  ✓ Labeled ({len(annotations)} total)")

    print(f"\nDone. {len(annotations)} total annotations saved to {LABELS_FILE}")


if __name__ == "__main__":
    label_reviews(Path("data/processed/zero_shot_scores.parquet"))
```

- [ ] **Step 2: Run labeling session**

```bash
uv run python -m src.models.labeling
```

Label at least 200 reviews. You can quit and resume — progress is saved to `data/labels/annotations.json`.

Target split:
- ~150 for fine-tuning training
- ~50-75 for evaluation test set

- [ ] **Step 3: Create train/test split helper**

Append to `src/models/labeling.py`:

```python
def split_labeled_data(
    scores_path: Path, train_ratio: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labeled reviews into train and test sets."""
    annotations = load_annotations()
    df = pd.read_parquet(scores_path)
    df["idx"] = df["idx"].astype(str)

    labeled = df[df["idx"].isin(annotations.keys())].copy()
    labeled["label_topics"] = labeled["idx"].map(lambda x: annotations[x]["topics"])
    labeled["label_sentiment"] = labeled["idx"].map(lambda x: annotations[x]["sentiment"])

    # Stratified-ish split by sentiment
    train = labeled.sample(frac=train_ratio, random_state=42)
    test = labeled.drop(train.index)
    return train, test
```

- [ ] **Step 4: Commit**

```bash
git add src/models/labeling.py
git commit -m "Add interactive labeling tool with train/test split"
```

---

### Task 9: Fine-Tune Topic Classifier

**Files:**
- Create: `src/models/fine_tune.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write fine-tuning tests**

Append to `tests/test_models.py`:

```python
import torch
from src.models.fine_tune import TopicDataset, create_topic_labels
from src.models.zero_shot import TOPICS


def test_create_topic_labels():
    topics = ["Workload", "Grading"]
    labels = create_topic_labels(topics)
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_models.py::test_create_topic_labels tests/test_models.py::test_topic_dataset -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `src/models/fine_tune.py`**

```python
# src/models/fine_tune.py
"""Fine-tuning pipelines for topic classification and sentiment analysis."""
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.models.zero_shot import TOPICS

MODEL_NAME = "distilbert-base-uncased"
MODELS_DIR = Path("models")

# --- Topic Classification ---


def create_topic_labels(topics: list[str]) -> list[float]:
    """Convert topic list to multi-hot vector."""
    return [1.0 if t in topics else 0.0 for t in TOPICS]


class TopicDataset(Dataset):
    def __init__(self, texts: list[str], topic_lists: list[list[str]], max_len: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.encodings = self.tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor([create_topic_labels(t) for t in topic_lists], dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def train_topic_classifier(
    train_texts: list[str],
    train_topics: list[list[str]],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> AutoModelForSequenceClassification:
    """Fine-tune DistilBERT for multi-label topic classification."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = TopicDataset(train_texts, train_topics)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(TOPICS), problem_type="multi_label_classification"
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(loader) * epochs
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} — loss: {total_loss / len(loader):.4f}")

    # Save model
    save_dir = MODELS_DIR / "topic_classifier"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    dataset.tokenizer.save_pretrained(save_dir)
    print(f"Saved topic classifier to {save_dir}")
    return model


def predict_topics(
    texts: list[str],
    model_dir: Path = MODELS_DIR / "topic_classifier",
    threshold: float = 0.5,
) -> list[list[str]]:
    """Predict topics using fine-tuned model."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.sigmoid(outputs.logits)

    results = []
    for row in probs.cpu():
        topics = [TOPICS[i] for i, p in enumerate(row) if p > threshold]
        results.append(topics)
    return results


# --- Sentiment Classification ---


SENTIMENT_LABELS = ["positive", "neutral", "negative"]


class SentimentDataset(Dataset):
    def __init__(self, texts: list[str], sentiments: list[str], max_len: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.encodings = self.tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(
            [SENTIMENT_LABELS.index(s) for s in sentiments], dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def train_sentiment_classifier(
    train_texts: list[str],
    train_sentiments: list[str],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> AutoModelForSequenceClassification:
    """Fine-tune DistilBERT for 3-class sentiment classification."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = SentimentDataset(train_texts, train_sentiments)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(loader) * epochs
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} — loss: {total_loss / len(loader):.4f}")

    save_dir = MODELS_DIR / "sentiment_classifier"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    dataset.tokenizer.save_pretrained(save_dir)
    print(f"Saved sentiment classifier to {save_dir}")
    return model


def predict_sentiment(
    texts: list[str],
    model_dir: Path = MODELS_DIR / "sentiment_classifier",
) -> list[dict]:
    """Predict sentiment using fine-tuned model."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)

    results = []
    for row in probs.cpu():
        idx = row.argmax().item()
        label = SENTIMENT_LABELS[idx]
        # Score in [-1, 1]: positive_prob - negative_prob
        score = row[0].item() - row[2].item()
        results.append({"label": label, "score": round(score, 4)})
    return results
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_models.py -v
```

Expected: all tests pass (including dataset/label utility tests).

- [ ] **Step 5: Run fine-tuning on labeled data**

```bash
uv run python -c "
from pathlib import Path
from src.models.labeling import split_labeled_data
from src.models.fine_tune import train_topic_classifier, train_sentiment_classifier

train, test = split_labeled_data(Path('data/processed/zero_shot_scores.parquet'))
print(f'Train: {len(train)}, Test: {len(test)}')

# Train topic classifier
print('\n--- Training Topic Classifier ---')
train_topic_classifier(
    train_texts=train['review_text'].tolist(),
    train_topics=train['label_topics'].tolist(),
)

# Train sentiment classifier
print('\n--- Training Sentiment Classifier ---')
train_sentiment_classifier(
    train_texts=train['review_text'].tolist(),
    train_sentiments=train['label_sentiment'].tolist(),
)
"
```

Expected: both models train in ~5 minutes on MPS, saved to `models/topic_classifier/` and `models/sentiment_classifier/`.

- [ ] **Step 6: Commit**

```bash
git add src/models/fine_tune.py tests/test_models.py
git commit -m "Add fine-tuning pipeline for topic and sentiment classifiers"
```

---

### Task 10: Evaluation Framework

**Files:**
- Create: `src/models/evaluate.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write evaluation tests**

Append to `tests/test_models.py`:

```python
from src.models.evaluate import compute_classification_metrics, compute_multilabel_metrics


def test_compute_classification_metrics():
    y_true = ["positive", "negative", "neutral", "positive"]
    y_pred = ["positive", "negative", "positive", "positive"]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert metrics["accuracy"] == 0.75


def test_compute_multilabel_metrics():
    y_true = [[1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1]]
    y_pred = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1]]
    metrics = compute_multilabel_metrics(y_true, y_pred)
    assert "f1_macro" in metrics
    assert "f1_per_topic" in metrics
    assert len(metrics["f1_per_topic"]) == 6
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_models.py::test_compute_classification_metrics tests/test_models.py::test_compute_multilabel_metrics -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `src/models/evaluate.py`**

```python
# src/models/evaluate.py
"""Evaluation metrics and model comparison."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)

from src.models.zero_shot import TOPICS
from src.models.fine_tune import create_topic_labels


def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute accuracy, F1 macro, and per-class F1 for single-label classification."""
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_per_class": dict(zip(labels, f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


def compute_multilabel_metrics(y_true: list[list[int]], y_pred: list[list[int]]) -> dict:
    """Compute F1 macro and per-topic F1 for multi-label classification."""
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_per_topic": dict(zip(
            TOPICS,
            f1_score(y_true, y_pred, average=None, zero_division=0),
        )),
    }


def compare_approaches(
    test_df: pd.DataFrame,
    zero_shot_topics: list[list[str]],
    zero_shot_sentiments: list[str],
    finetuned_topics: list[list[str]],
    finetuned_sentiments: list[str],
) -> dict:
    """Compare all three approaches against hand-labeled ground truth.

    test_df must have 'label_topics' (list[str]) and 'label_sentiment' (str) columns.
    """
    # Ground truth
    gt_topics = [create_topic_labels(t) for t in test_df["label_topics"]]
    gt_sentiments = test_df["label_sentiment"].tolist()

    # Baseline
    from src.models.baseline import star_to_sentiment
    baseline_sentiments = [star_to_sentiment(r) for r in test_df["star_rating"]]

    # Convert topic lists to multi-hot
    zs_topics = [create_topic_labels(t) for t in zero_shot_topics]
    ft_topics = [create_topic_labels(t) for t in finetuned_topics]

    results = {
        "topic_classification": {
            "zero_shot": compute_multilabel_metrics(gt_topics, zs_topics),
            "fine_tuned": compute_multilabel_metrics(gt_topics, ft_topics),
        },
        "sentiment": {
            "baseline_stars": compute_classification_metrics(gt_sentiments, baseline_sentiments),
            "zero_shot": compute_classification_metrics(gt_sentiments, zero_shot_sentiments),
            "fine_tuned": compute_classification_metrics(gt_sentiments, finetuned_sentiments),
        },
        "agreement": {
            "topic_zs_ft_kappa": cohen_kappa_score(
                [str(t) for t in zs_topics], [str(t) for t in ft_topics]
            ),
            "sent_zs_ft_kappa": cohen_kappa_score(zero_shot_sentiments, finetuned_sentiments),
        },
    }
    return results
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_models.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run full evaluation**

```bash
uv run python -c "
import json
from pathlib import Path
from src.models.labeling import split_labeled_data
from src.models.zero_shot import TopicClassifier, SentimentAnalyzer
from src.models.fine_tune import predict_topics, predict_sentiment
from src.models.evaluate import compare_approaches

_, test = split_labeled_data(Path('data/processed/zero_shot_scores.parquet'))
texts = test['review_text'].tolist()

print('Running zero-shot on test set...')
tc = TopicClassifier()
sa = SentimentAnalyzer()
zs_topics = [tc.classify(t) for t in texts]
zs_sents = [sa.analyze(t)['label'] for t in texts]

print('Running fine-tuned on test set...')
ft_topics = predict_topics(texts)
ft_sents = [r['label'] for r in predict_sentiment(texts)]

print('Comparing...')
results = compare_approaches(test, zs_topics, zs_sents, ft_topics, ft_sents)
print(json.dumps(results, indent=2, default=str))

# Save results
Path('data/processed/evaluation_results.json').write_text(
    json.dumps(results, indent=2, default=str)
)
print('Saved to data/processed/evaluation_results.json')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/models/evaluate.py tests/test_models.py
git commit -m "Add evaluation framework with three-way comparison"
```

---

### Task 11: Recommendation Engine

**Files:**
- Create: `src/recommend/engine.py`
- Create: `tests/test_recommend.py`

- [ ] **Step 1: Write recommendation tests**

```python
# tests/test_recommend.py
import pandas as pd
from src.recommend.engine import score_professors, filter_results


def make_scores_df():
    """Helper: fake professor sentiment scores."""
    return pd.DataFrame([
        {
            "professor_name": "Alice",
            "num_reviews": 10,
            "workload": 0.5,
            "grading": 0.8,
            "teaching_quality": 0.9,
            "course_content": 0.3,
            "accessibility": 0.6,
            "exam_difficulty": -0.2,
        },
        {
            "professor_name": "Bob",
            "num_reviews": 5,
            "workload": -0.3,
            "grading": -0.5,
            "teaching_quality": 0.2,
            "course_content": 0.7,
            "accessibility": 0.1,
            "exam_difficulty": -0.8,
        },
        {
            "professor_name": "Carol",
            "num_reviews": 2,
            "workload": 0.1,
            "grading": 0.1,
            "teaching_quality": 0.1,
            "course_content": 0.1,
            "accessibility": 0.1,
            "exam_difficulty": 0.1,
        },
    ])


def test_score_equal_weights():
    df = make_scores_df()
    weights = {
        "workload": 5, "grading": 5, "teaching_quality": 5,
        "course_content": 5, "accessibility": 5, "exam_difficulty": 5,
    }
    result = score_professors(df, weights)
    assert result.iloc[0]["professor_name"] == "Alice"  # highest average
    assert "score" in result.columns


def test_score_grading_only():
    df = make_scores_df()
    weights = {
        "workload": 0, "grading": 10, "teaching_quality": 0,
        "course_content": 0, "accessibility": 0, "exam_difficulty": 0,
    }
    result = score_professors(df, weights)
    assert result.iloc[0]["professor_name"] == "Alice"  # grading=0.8


def test_filter_min_reviews():
    df = make_scores_df()
    df["score"] = [0.5, 0.3, 0.1]
    result = filter_results(df, min_reviews=3)
    assert len(result) == 2
    assert "Carol" not in result["professor_name"].values
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_recommend.py -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `src/recommend/engine.py`**

```python
# src/recommend/engine.py
"""Weighted scoring recommendation engine."""
import pandas as pd

from src.models.zero_shot import TOPICS

TOPIC_KEYS = [t.lower().replace(" ", "_") for t in TOPICS]


def score_professors(prof_scores: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Score and rank professors by weighted topic sentiment.

    prof_scores: DataFrame with columns for each topic key + professor_name.
    weights: {topic_key: 0-10 weight}.
    Returns DataFrame sorted by score descending.
    """
    df = prof_scores.copy()
    total_weight = sum(weights.values())
    if total_weight == 0:
        df["score"] = 0.0
        return df.sort_values("score", ascending=False).reset_index(drop=True)

    df["score"] = sum(
        weights.get(key, 0) * df[key] for key in TOPIC_KEYS if key in df.columns
    ) / total_weight

    return df.sort_values("score", ascending=False).reset_index(drop=True)


def filter_results(
    df: pd.DataFrame,
    min_reviews: int = 3,
    course_prefix: str | None = None,
) -> pd.DataFrame:
    """Filter ranked results by minimum reviews and optional course prefix."""
    result = df.copy()
    if "num_reviews" in result.columns:
        result = result[result["num_reviews"] >= min_reviews]
    if course_prefix and "course_name" in result.columns:
        result = result[result["course_name"].str.startswith(course_prefix)]
    return result.reset_index(drop=True)


def aggregate_professor_scores(
    reviews_df: pd.DataFrame, approach: str = "zero_shot"
) -> pd.DataFrame:
    """Aggregate per-review scores into per-professor topic sentiment scores.

    approach: 'baseline', 'zero_shot', or 'fine_tuned'
    """
    topic_cols = [f"topic_{k}_score" for k in TOPIC_KEYS]
    available = [c for c in topic_cols if c in reviews_df.columns]

    agg = reviews_df.groupby("professor_name").agg(
        num_reviews=("review_text", "count"),
        **{
            col.replace("topic_", "").replace("_score", ""): (col, "mean")
            for col in available
        },
    ).reset_index()

    # Fill NaN with 0 (no data for that topic)
    for key in TOPIC_KEYS:
        if key not in agg.columns:
            agg[key] = 0.0
        agg[key] = agg[key].fillna(0.0)

    return agg
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_recommend.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/recommend/engine.py tests/test_recommend.py
git commit -m "Add weighted scoring recommendation engine"
```

---

### Task 12: Streamlit App

**Files:**
- Create: `src/app/streamlit_app.py`

- [ ] **Step 1: Implement the Streamlit app**

```python
# src/app/streamlit_app.py
"""UNC Course Compass — Streamlit frontend."""
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.models.zero_shot import TOPICS, TopicClassifier, SentimentAnalyzer
from src.models.baseline import star_to_sentiment
from src.models.fine_tune import predict_topics, predict_sentiment
from src.recommend.engine import score_professors, filter_results, TOPIC_KEYS

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

st.set_page_config(page_title="UNC Course Compass", layout="wide")
st.title("UNC Course Compass")
st.caption("ML-powered course recommendations for UNC Stats & Data Science")


@st.cache_data
def load_data():
    reviews = pd.read_parquet(DATA_DIR / "reviews.parquet")
    scores = pd.read_parquet(DATA_DIR / "zero_shot_scores.parquet")
    eval_path = DATA_DIR / "evaluation_results.json"
    eval_results = json.loads(eval_path.read_text()) if eval_path.exists() else None
    return reviews, scores, eval_results


@st.cache_resource
def load_models():
    tc = TopicClassifier()
    sa = SentimentAnalyzer()
    return tc, sa


def radar_chart(scores: dict[str, float], title: str) -> go.Figure:
    """Create a radar chart for topic sentiment scores."""
    topics = list(scores.keys())
    values = list(scores.values())
    # Close the polygon
    topics += [topics[0]]
    values += [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=topics, fill="toself", name=title
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[-1, 1])),
        showlegend=False,
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


reviews, scores, eval_results = load_data()

tab1, tab2, tab3 = st.tabs(["Explore", "Recommend", "Model Comparison"])

# --- Tab 1: Explore ---
with tab1:
    professors = sorted(reviews["professor_name"].unique())
    selected = st.selectbox("Select Professor", professors)

    prof_reviews = reviews[reviews["professor_name"] == selected]
    prof_scores = scores[scores["professor_name"] == selected]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Reviews", len(prof_reviews))
        avg_star = prof_reviews["star_rating"].mean()
        avg_diff = prof_reviews["difficulty_rating"].mean()
        st.metric("Avg Rating", f"{avg_star:.1f}/5")
        st.metric("Avg Difficulty", f"{avg_diff:.1f}/5")

        # Radar chart from zero-shot scores
        if len(prof_scores) > 0:
            topic_avgs = {}
            for key in TOPIC_KEYS:
                col_name = f"topic_{key}_score"
                if col_name in prof_scores.columns:
                    vals = prof_scores[col_name].dropna()
                    topic_avgs[key.replace("_", " ").title()] = vals.mean() if len(vals) > 0 else 0
            if topic_avgs:
                st.plotly_chart(radar_chart(topic_avgs, selected), use_container_width=True)

    with col2:
        st.subheader("Reviews")
        for _, row in prof_reviews.iterrows():
            stars = "★" * row["star_rating"] + "☆" * (5 - row["star_rating"])
            course = row["course_name"] or "Unknown"
            st.markdown(f"**{stars}** — {course}")
            st.write(row["review_text"])
            st.divider()

# --- Tab 2: Recommend ---
with tab2:
    st.subheader("Set Your Preferences")

    cols = st.columns(3)
    weights = {}
    for i, topic in enumerate(TOPICS):
        key = TOPIC_KEYS[i]
        with cols[i % 3]:
            weights[key] = st.slider(topic, 0, 10, 5, key=f"w_{key}")

    min_rev = st.slider("Minimum reviews", 1, 20, 3)

    # Aggregate professor scores
    prof_agg = scores.groupby("professor_name").agg(
        num_reviews=("review_text", "count"),
        **{
            TOPIC_KEYS[i]: (f"topic_{TOPIC_KEYS[i]}_score", "mean")
            for i in range(len(TOPICS))
            if f"topic_{TOPIC_KEYS[i]}_score" in scores.columns
        },
    ).reset_index()
    for key in TOPIC_KEYS:
        if key not in prof_agg.columns:
            prof_agg[key] = 0.0
        prof_agg[key] = prof_agg[key].fillna(0.0)

    ranked = score_professors(prof_agg, weights)
    ranked = filter_results(ranked, min_reviews=min_rev)

    st.subheader("Recommendations")
    for _, row in ranked.head(10).iterrows():
        with st.expander(f"**{row['professor_name']}** — Score: {row['score']:.2f} ({int(row['num_reviews'])} reviews)"):
            topic_scores = {TOPICS[i]: row.get(TOPIC_KEYS[i], 0) for i in range(len(TOPICS))}
            st.plotly_chart(radar_chart(topic_scores, row["professor_name"]), use_container_width=True)

# --- Tab 3: Model Comparison ---
with tab3:
    st.subheader("Compare Approaches")

    if eval_results:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Topic Classification F1")
            topic_data = eval_results.get("topic_classification", {})
            comparison = pd.DataFrame({
                "Metric": ["F1 Macro", "F1 Micro"],
                "Zero-Shot": [
                    topic_data.get("zero_shot", {}).get("f1_macro", "N/A"),
                    topic_data.get("zero_shot", {}).get("f1_micro", "N/A"),
                ],
                "Fine-Tuned": [
                    topic_data.get("fine_tuned", {}).get("f1_macro", "N/A"),
                    topic_data.get("fine_tuned", {}).get("f1_micro", "N/A"),
                ],
            })
            st.dataframe(comparison, hide_index=True)

        with col2:
            st.markdown("### Sentiment Accuracy")
            sent_data = eval_results.get("sentiment", {})
            comparison = pd.DataFrame({
                "Metric": ["Accuracy", "F1 Macro"],
                "Star Baseline": [
                    sent_data.get("baseline_stars", {}).get("accuracy", "N/A"),
                    sent_data.get("baseline_stars", {}).get("f1_macro", "N/A"),
                ],
                "Zero-Shot": [
                    sent_data.get("zero_shot", {}).get("accuracy", "N/A"),
                    sent_data.get("zero_shot", {}).get("f1_macro", "N/A"),
                ],
                "Fine-Tuned": [
                    sent_data.get("fine_tuned", {}).get("accuracy", "N/A"),
                    sent_data.get("fine_tuned", {}).get("f1_macro", "N/A"),
                ],
            })
            st.dataframe(comparison, hide_index=True)

    # Live comparison on a single review
    st.markdown("### Try a Review")
    sample_text = st.text_area(
        "Paste a review to compare all approaches:",
        value=reviews["review_text"].iloc[0] if len(reviews) > 0 else "",
        height=100,
    )

    if st.button("Analyze") and sample_text:
        tc, sa = load_models()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Star Proxy**")
            st.info("N/A (needs star rating)")

        with col2:
            st.markdown("**Zero-Shot**")
            topics = tc.classify(sample_text)
            sentiment = sa.analyze(sample_text)
            st.write(f"Topics: {', '.join(topics) or 'none detected'}")
            st.write(f"Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")

        with col3:
            st.markdown("**Fine-Tuned**")
            if (MODELS_DIR / "topic_classifier").exists():
                ft_topics = predict_topics([sample_text])[0]
                ft_sent = predict_sentiment([sample_text])[0]
                st.write(f"Topics: {', '.join(ft_topics) or 'none detected'}")
                st.write(f"Sentiment: {ft_sent['label']} ({ft_sent['score']:.2f})")
            else:
                st.warning("Fine-tuned models not yet trained.")
```

- [ ] **Step 2: Test the app locally**

```bash
uv run streamlit run src/app/streamlit_app.py
```

Expected: app opens in browser with 3 tabs. Verify:
- Explore tab: professor dropdown works, radar chart renders, reviews display
- Recommend tab: sliders work, recommendations update, expanders open
- Model Comparison tab: metrics table shows (if eval results exist), live analysis works

- [ ] **Step 3: Commit**

```bash
git add src/app/streamlit_app.py
git commit -m "Add Streamlit app with explore, recommend, and comparison views"
```

---

### Task 13: Integration & Polish

**Files:**
- Modify: various files for final integration

- [ ] **Step 1: Create a run-all script**

Create `run_pipeline.py` at project root:

```python
#!/usr/bin/env python
"""Run the full Course Compass pipeline end-to-end."""
import asyncio
from pathlib import Path

from src.scraper.run import main as scrape
from src.scraper.preprocess import preprocess_reviews
from src.models.process import process_all

import pandas as pd


def main():
    raw_path = Path("data/processed/reviews.parquet")
    processed_dir = Path("data/processed")

    if not raw_path.exists():
        print("Step 1: Scraping RateMyProfessor...")
        asyncio.run(scrape())

    print("Step 2: Preprocessing...")
    df = pd.read_parquet(raw_path)
    df = preprocess_reviews(df)
    df.to_parquet(raw_path, index=False)
    print(f"  {len(df)} cleaned reviews")

    print("Step 3: Running zero-shot pipeline...")
    process_all(raw_path, processed_dir)

    print("\nPipeline complete!")
    print("Next steps:")
    print("  1. Label data:  python -m src.models.labeling")
    print("  2. Fine-tune:   see Task 9 commands")
    print("  3. Evaluate:    see Task 10 commands")
    print("  4. Launch app:  streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full pipeline**

```bash
uv run python run_pipeline.py
```

Verify all steps complete without errors.

- [ ] **Step 3: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add run_pipeline.py
git commit -m "Add end-to-end pipeline runner"
```
