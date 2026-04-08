# UNC Course Compass — Design Spec

ML-powered system that scrapes RateMyProfessor reviews for UNC Stats & Data Science, extracts topics and sentiment, and recommends courses based on student preferences.

**Course:** Data 522 — Practical Deep Learning  
**Scope:** UNC Statistics & Data Science department  
**Developer:** Kyle (solo)  
**Hardware:** MacBook Pro M5, 24GB RAM (PyTorch MPS backend), Google Colab as backup  

## Architecture Overview

**Approach C: Zero-Shot Baseline → Fine-Tune Comparison**

```
RateMyProfessor ──► Raw Reviews ──► Preprocessed Data
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
              Star Ratings         Zero-Shot Models     Fine-Tuned Models
              (Proxy Baseline)     (DL Baseline)        (DL Fine-Tuned)
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         ▼
                              Topic × Sentiment Scores
                                         │
                                         ▼
                              Recommendation Engine
                                         │
                                         ▼
                                   Streamlit App
```

Three-layer comparison for evaluation:
1. Star ratings as sentiment proxy (simple baseline)
2. Zero-shot text-based topic classification + sentiment (DL baseline)
3. Fine-tuned DistilBERT topic classification + sentiment (DL fine-tuned)

## 1. Data Collection

### Source
RateMyProfessor — scrape via their unofficial GraphQL API.

### Target
All professors listed under UNC Chapel Hill's Statistics & Data Science department.

### Expected Yield
~500–2000 reviews. Sufficient for fine-tuning DistilBERT and meaningful evaluation.

### Fields
| Field | Source | Notes |
|-------|--------|-------|
| review_text | RMP | Free-text review body |
| star_rating | RMP | 1–5 overall quality |
| difficulty_rating | RMP | 1–5 difficulty |
| would_take_again | RMP | Boolean |
| course_name | RMP | e.g. "STOR 435" |
| professor_name | RMP | Full name |
| date | RMP | Review date |
| thumbs_up / thumbs_down | RMP | Helpfulness votes |

### Storage
- Raw: `data/raw/` as JSON (one file per professor)
- Cleaned: `data/processed/reviews.parquet`

### Risks & Mitigations
- RMP API may change or rate-limit → implement retry logic, cache aggressively, keep raw backups
- Fallback: Kaggle RMP datasets exist if scraping fails entirely

## 2. Topic Classification

### Categories (6)
| Topic | Description | Example phrases |
|-------|-------------|-----------------|
| Workload | Homework volume, time commitment | "tons of homework", "manageable workload" |
| Grading | Fairness, curves, grade distribution | "harsh grader", "generous curve" |
| Teaching Quality | Lecture clarity, engagement, responsiveness | "explains well", "boring lectures" |
| Course Content | Relevance, interest, organization | "great material", "outdated content" |
| Accessibility | Office hours, approachability | "always available", "hard to reach" |
| Exam Difficulty | Test format, fairness, prep alignment | "exams are fair", "nothing like homework" |

### Multi-label
Each review can be tagged with 1+ topics. A review like "Great lectures but brutal exams" → Teaching Quality + Exam Difficulty.

### Zero-Shot Baseline
- Model: `facebook/bart-large-mnli`
- Method: Zero-shot classification with topic labels as candidates
- Threshold: Accept topics with confidence > 0.3 (tune on validation set)

### Fine-Tuned Model
- Model: `distilbert-base-uncased`
- Task: Multi-label classification (6 sigmoid outputs)
- Loss: Binary cross-entropy
- Training data: ~200–400 hand-corrected reviews (use zero-shot predictions as starting annotations, then manually fix)
- Training: HuggingFace Transformers, PyTorch MPS, ~5–10 min

## 3. Sentiment Analysis

### Three-Layer Comparison

**Layer 1 — Star Rating Proxy (Simple Baseline):**
- Map RMP star_rating to sentiment: 1–2 = negative, 3 = neutral, 4–5 = positive
- Map difficulty_rating similarly
- Per-topic sentiment = overall star sentiment (no topic differentiation)

**Layer 2 — Zero-Shot Text Sentiment (DL Baseline):**
- Model: `cardiffnlp/twitter-roberta-base-sentiment-latest` or similar
- Apply to full review text
- Aspect-based approach: split review into sentences, classify each sentence's topic(s) using the zero-shot topic model, then score sentiment on topic-matched sentences only
- Reviews with no topic match above threshold are tagged "general" and excluded from per-topic aggregation (still count toward overall sentiment)
- Output: sentiment per topic per review

**Layer 3 — Fine-Tuned Text Sentiment (DL Fine-Tuned):**
- Model: `distilbert-base-uncased` fine-tuned for 3-class sentiment (positive / neutral / negative)
- Training labels: hand-labeled subset, augmented with star ratings as noisy labels
- Applied per topic-relevant text spans

### Sentiment Normalization
All approaches output a normalized score in [-1, 1] per topic per professor:
- Aggregate across reviews: mean sentiment score per topic
- Track review count per topic for confidence weighting

## 4. Evaluation

### Labeled Test Set
- Hand-label ~100–150 reviews with ground-truth topics + per-topic sentiment
- Hold out from fine-tuning data
- This is the authoritative benchmark for all three approaches

### Metrics
| Metric | Applied To |
|--------|-----------|
| F1 (macro) | Topic classification |
| F1 (per-topic) | Topic classification |
| Accuracy | Sentiment (3-class) |
| F1 (macro) | Sentiment (3-class) |
| Confusion matrices | Both tasks |
| Cohen's kappa | Inter-approach agreement |

### Key Comparisons
1. Zero-shot vs. fine-tuned topic classification F1
2. Star proxy vs. zero-shot vs. fine-tuned sentiment accuracy
3. Per-topic sentiment accuracy (where does fine-tuning help most?)

## 5. Recommendation Engine

### Input
User sets preferences via sliders (0–10) for each of the 6 topics.

### Scoring
For each professor/course:
```
score = Σ (user_weight_i × topic_sentiment_i) / Σ user_weight_i
```
Where `topic_sentiment_i` is the normalized mean sentiment [-1, 1] for that topic.

### Ranking
- Sort by score descending
- Display top-N with score, review count, and per-topic breakdown

### Filters
- Minimum review count (exclude professors with < 3 reviews)
- Course level (100/200/300/400/500+)

## 6. Frontend — Streamlit App

### View 1: Explore
- Browse by professor or course
- Radar chart showing 6-topic sentiment profile
- Review list with topic/sentiment highlighting
- Star rating and difficulty shown alongside model predictions

### View 2: Recommend
- 6 sliders for topic preferences
- Ranked list of professors/courses
- Click to expand → see full profile (radar chart + reviews)

### View 3: Model Comparison
- Select a sample review
- Side-by-side: star proxy vs. zero-shot vs. fine-tuned predictions
- Highlighted topic spans with sentiment labels
- Aggregate metrics dashboard (F1, accuracy charts)

## 7. Project Structure

```
course_review/
├── data/
│   ├── raw/                # JSON per professor from RMP
│   ├── processed/          # reviews.parquet, labeled subsets
│   └── labels/             # hand-corrected annotations
├── notebooks/
│   ├── eda.ipynb           # data exploration
│   ├── labeling.ipynb      # annotation workflow
│   └── experiments.ipynb   # model experiments
├── src/
│   ├── scraper/            # RMP GraphQL scraping
│   │   ├── __init__.py
│   │   ├── client.py       # API client
│   │   └── parse.py        # response parsing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── zero_shot.py    # zero-shot topic + sentiment
│   │   ├── fine_tune.py    # fine-tuning pipeline
│   │   ├── baseline.py     # star rating proxy
│   │   └── evaluate.py     # metrics + comparison
│   ├── recommend/
│   │   ├── __init__.py
│   │   └── engine.py       # weighted scoring
│   └── app/
│       ├── __init__.py
│       └── streamlit_app.py
├── tests/
│   ├── test_scraper.py
│   ├── test_models.py
│   └── test_recommend.py
├── docs/
│   └── superpowers/specs/  # this spec
├── pyproject.toml
└── README.md
```

## 8. Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.11+ |
| ML Framework | PyTorch + HuggingFace Transformers |
| Data | pandas, pyarrow (parquet) |
| Scraping | httpx (async HTTP) |
| Frontend | Streamlit |
| Visualization | plotly (radar charts, bar charts) |
| Env management | pyenv + venv |
| Compute | Local MPS (M5), Colab backup |

## 9. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| RMP API changes / rate limits | Medium | Cache aggressively, implement retries, keep Kaggle backup |
| Small dataset limits fine-tuning gains | Medium | Data augmentation, few-shot techniques, present finding honestly |
| Zero-shot is "good enough" | Medium | This is a valid finding — frame as "when is fine-tuning worth it?" |
| Aspect-based sentiment extraction is noisy | High | Start with sentence-level splitting, iterate on span extraction |
| Scope creep | Medium | Stick to Stats & DS dept, resist adding features |

## 10. Implementation Order

1. **Scraping** — get data first, everything depends on it
2. **EDA + preprocessing** — understand the data
3. **Zero-shot pipeline** — working system, no training needed
4. **Hand-labeling** — annotate subset for fine-tuning + evaluation
5. **Fine-tuning** — train DistilBERT models
6. **Evaluation** — compare all three approaches
7. **Recommendation engine** — weighted scoring
8. **Streamlit app** — wire everything together
9. **Polish** — presentation prep, cleanup
