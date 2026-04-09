# UNC Course Compass — Project Report

**Course:** Data 522 — Practical Deep Learning  
**Developer:** Kyle  
**Date:** April 2026  

---

## 1. Project Overview

UNC Course Compass is an end-to-end ML system that scrapes student reviews from RateMyProfessor for UNC's Statistics and Data Science departments, performs **aspect-based sentiment analysis** — classifying what topics each review discusses and whether the sentiment toward each topic is positive, neutral, or negative — and recommends courses based on student preferences through a Streamlit web app.

The key insight: a single review can be positive about one aspect and negative about another. "Great lectures but brutal exams" is not a neutral review — it's positive about teaching quality and negative about exam difficulty. Our system captures this nuance.

### Architecture

```
RateMyProfessor GraphQL API
        │
        ▼
   2,429 raw reviews (117 professors)
        │
        ▼
   Text cleaning + deduplication
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
  Star Rating Proxy      Zero-Shot Models       Fine-Tuned Joint Model
  (no NLP)               (BART + RoBERTa)       (DistilBERT, 5×4-class)
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
            Per-topic sentiment: {topic → pos/neu/neg}
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
            Weighted Scoring         Evaluation
            (recommendations)     (per-topic F1, accuracy)
                    │
                    ▼
              Streamlit App
```

### Three-Layer Comparison

| Layer | Model | What It Does |
|-------|-------|-------------|
| 1. Star Baseline | None (rule-based) | Maps 1-5 stars to sentiment. No NLP, no topic differentiation. |
| 2. Zero-Shot | BART-mnli (406M) + RoBERTa (125M) | Pretrained models, no training on our data. Sentence-level aspect-based sentiment. |
| 3. Fine-Tuned | DistilBERT (66M) joint model | Trained on our labeled data. 5 independent 4-class heads for per-topic sentiment. |

---

## 2. Data Collection

### 2.1 Source: RateMyProfessor GraphQL API

RateMyProfessor is a React/Next.js app backed by a Relay-based GraphQL endpoint at `https://www.ratemyprofessors.com/graphql`. We reverse-engineered the API through a hybrid approach:

1. **Exploration:** Used web fetching to inspect RMP's page structure, discover the GraphQL endpoint and auth mechanism
2. **Verification:** Tested queries with curl to confirm field names, pagination, and auth
3. **Automation:** Wrote an async Python client (`src/scraper/client.py`) using httpx

**Technical details:**
- Auth: `Basic dGVzdDp0ZXN0` (base64 of "test:test") — a public token from RMP's frontend JS
- School ID: `U2Nob29sLTEyMzI=` (base64 of "School-1232", UNC Chapel Hill)
- Required browser-like `User-Agent` header — RMP returns 403 for python-httpx default UA
- Pagination: cursor-based, fetches all 4,838 UNC professors across 10 pages of 500, then filters to target departments

**Scraping process:**
1. Paginate through all UNC professors (4,838 total)
2. Filter to: Statistics (105), Biostatistics (22), Statistics & Ops Research (15) = **142 professors**
3. Fetch all reviews per professor with cursor-based pagination (20 per page)
4. 0.5s delay between requests, 3-attempt retry with exponential backoff
5. Cache raw JSON per professor in `data/raw/{legacy_id}.json`

### 2.2 Dataset Summary

| Metric | Value |
|--------|-------|
| Total reviews | 2,429 |
| Professors (with reviews) | 117 |
| Professors (total in RMP) | 142 |
| Departments | Statistics (105), Biostatistics (22), Stats & Ops Research (15) |
| Avg review length | 276 characters |

**Star rating distribution:**

| Stars | Count | Percentage |
|-------|-------|------------|
| 1 | 614 | 25.3% |
| 2 | 303 | 12.5% |
| 3 | 267 | 11.0% |
| 4 | 405 | 16.7% |
| 5 | 840 | 34.6% |

The distribution is bimodal — students leave reviews when they feel strongly. 5-star reviews dominate (34.6%), followed by 1-star (25.3%).

### 2.3 Top Professors by Review Count

| Professor | Reviews | Avg Stars | Department |
|-----------|---------|-----------|------------|
| Mario Giacomazzo | 157 | 2.7 | Statistics |
| Jeff McLean | 154 | 3.5 | Statistics |
| Chuanshu Ji | 137 | 2.0 | Statistics |
| Prairie Goodwin | 119 | 2.8 | Statistics |
| Gabor Pataki | 116 | 3.2 | Statistics |
| Oluremi Abayomi | 115 | 3.5 | Statistics & Ops Research |
| William Lassiter | 103 | 4.0 | Statistics |
| Teressa Bergland | 101 | 2.0 | Statistics & Ops Research |
| Jane Monaco | 82 | 4.2 | Biostatistics |
| Sayan Banerjee | 64 | 3.6 | Statistics |
| Robin Cunningham | 59 | 4.8 | Statistics |
| Nicolas Fraiman | 58 | 3.8 | Statistics |
| Kendall Thomas | 57 | 2.4 | Statistics |
| Guanting Chen | 54 | 2.5 | Statistics & Ops Research |
| Adrian Allen | 49 | 3.1 | Statistics |

---

## 3. Data Preprocessing

Minimal by design — transformer models handle messy text well.

**Steps:**
1. **Whitespace normalization:** `re.sub(r"\s+", " ", text).strip()`
2. **Empty review removal:** Drop reviews with no text after cleaning
3. **Deduplication:** Remove exact duplicates on `(review_text, professor_name, course_name)`

**Result:** 2,429 → 2,429 reviews (no duplicates found).

**Intentionally omitted:** lowercasing (handled by model tokenizer), stopword removal (attention handles this), stemming (subword tokenization makes it unnecessary), spell correction (risk of meaning change).

---

## 4. Labeling

### 4.1 Per-Topic Sentiment Labels

Each review is labeled with a **dict mapping discussed topics to their sentiment**:

```json
{"Teaching Quality": "positive", "Exam Difficulty": "negative"}
```

This captures that "Great lectures but brutal exams" is not neutral — it's positive about one thing and negative about another. Topics not in the dict are "not discussed."

### 4.2 Topic Taxonomy

| Topic | Description |
|-------|-------------|
| Workload | Homework volume, time commitment |
| Grading | Fairness, curves, grade distribution |
| Teaching Quality | Lecture clarity, engagement, responsiveness |
| Accessibility | Office hours, approachability |
| Exam Difficulty | Test format, fairness, prep alignment |

### 4.3 Labeling Process

All 2,429 reviews were labeled using **Claude Sonnet 4.6** as an automated annotator. We dispatched 8 parallel labeling agents, each processing ~300 reviews independently. Each agent read the raw review text (no model predictions shown) and assigned per-topic sentiment based on what the review clearly discusses.

### 4.4 Label Distribution

**Topic frequency:**

| Topic | Count | % of Reviews | Positive | Neutral | Negative |
|-------|-------|-------------|----------|---------|----------|
| Teaching Quality | 2,016 | 83.0% | 1,017 | 165 | 834 |
| Exam Difficulty | 1,161 | 47.8% | 411 | 139 | 611 |
| Grading | 875 | 36.0% | 431 | 28 | 416 |
| Workload | 687 | 28.3% | 296 | 38 | 353 |
| Accessibility | 624 | 25.7% | 421 | 5 | 198 |

- **Average topics per review:** 2.21
- **Reviews with no clear topic:** 74 (3.0%)
- **Total topic-sentiment assignments:** 5,363

Notable patterns: Teaching Quality is discussed in 83% of reviews. Accessibility is overwhelmingly positive (67% positive) — when students mention office hours, it's usually to praise them. Exam Difficulty is predominantly negative (53%) — exams are more often complained about than praised.

### 4.5 Train/Test Split

- **Train:** 1,700 reviews (70%)
- **Test:** 729 reviews (30%)

---

## 5. Model Architecture

### 5.1 Layer 1 — Star Rating Baseline

No NLP. Maps star ratings to sentiment:
- Stars 1–2 → negative (-1.0)
- Star 3 → neutral (0.0)
- Stars 4–5 → positive (+1.0)

Same sentiment for every topic. Exists as the "why do we need DL?" baseline.

### 5.2 Layer 2 — Zero-Shot Transformer Models

Two pretrained models used **without any fine-tuning on our data:**

#### Topic Classification: `facebook/bart-large-mnli` (406M params)

A BART model fine-tuned on Multi-NLI for natural language inference. Zero-shot classification works by reformulating topic detection as entailment: "Does this text entail 'homework volume and time commitment'?"

- Uses descriptive phrases as candidate labels (not just single words)
- `multi_label=True` — each topic scored independently
- Threshold: 0.3 confidence for topic assignment

#### Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest` (125M params)

RoBERTa fine-tuned on ~124M tweets for 3-class sentiment.

- Composite score: `positive_prob - negative_prob` → [-1, 1]
- Thresholds: >0.25 = positive, <-0.25 = negative, else neutral

#### Aspect-Based Sentiment (the key method)

1. Split review into sentences (regex on `.!?`)
2. Classify each sentence's topics with BART-mnli
3. Score each sentence's sentiment with RoBERTa
4. Assign sentence sentiment to its detected topics
5. Average per-topic across sentences

**Example:** *"The lectures were amazing but the exams were brutal"*
- Sentence 1: "The lectures were amazing" → Teaching Quality: +0.82
- Sentence 2: "the exams were brutal" → Exam Difficulty: -0.71
- Result: `{Teaching Quality: positive, Exam Difficulty: negative}`

### 5.3 Layer 3 — Fine-Tuned Joint DistilBERT (66M params)

#### Why a Joint Model?

Instead of separate topic classifier + sentiment classifier (the old approach), we use a **single model with 5 independent 4-class heads**, one per topic. Each head predicts: not_discussed (0), positive (1), neutral (2), or negative (3).

This is better because:
- It learns the **correlation** between topics and sentiments in a single forward pass
- A topic's sentiment depends on the full review context, not just isolated sentences
- One model instead of two = simpler, fewer parameters, faster inference

#### Architecture

```
Review Text
    │
    ▼
DistilBERT backbone (6 layers, 768-dim hidden)
    │
    ▼
[CLS] token → Pre-classifier (768→768) → ReLU → Dropout(0.1)
    │
    ├── Head 1: Linear(768→4) → Workload [not_discussed/pos/neu/neg]
    ├── Head 2: Linear(768→4) → Grading
    ├── Head 3: Linear(768→4) → Teaching Quality
    ├── Head 4: Linear(768→4) → Accessibility
    └── Head 5: Linear(768→4) → Exam Difficulty
```

#### Training Details

| Parameter | Value |
|-----------|-------|
| Base model | distilbert-base-uncased |
| Classification heads | 5 × Linear(768→4) |
| Loss | Average of 5 cross-entropy losses (one per head) |
| Optimizer | AdamW, lr=2e-5 |
| Scheduler | Linear warmup (0 warmup steps) |
| Batch size | 16 |
| Epochs | 5 |
| Max sequence length | 256 tokens |
| Training data | 1,700 reviews with per-topic sentiment labels |

**Training loss:** 1.003 → 0.518 over 5 epochs

#### Why Not Sigmoid (Multi-Label) for Per-Topic Sentiment?

The states within each topic are **mutually exclusive** — a review can't be both positive and negative about the same topic simultaneously. Softmax (via cross-entropy) enforces this constraint. Using sigmoid + BCE (as in the old multi-label topic classifier) would allow contradictory predictions.

---

## 6. Evaluation Results

### 6.1 Methodology

- **Test set:** 729 reviews (30% holdout)
- **Ground truth:** Sonnet 4.6 per-topic sentiment labels (independent of all models)
- **Per-topic sentiment accuracy:** evaluated only on reviews where the topic is present in ground truth

### 6.2 Per-Topic Sentiment

| Topic | n | ZS Accuracy | FT Accuracy | ZS F1 | FT F1 |
|-------|---|-------------|-------------|-------|-------|
| Teaching Quality | 611 | 0.755 | **0.827** | **0.487** | 0.445 |
| Exam Difficulty | 374 | **0.626** | 0.596 | **0.403** | 0.350 |
| Grading | 253 | 0.581 | **0.510** | **0.368** | 0.310 |
| Workload | 195 | **0.513** | 0.328 | **0.344** | 0.230 |
| Accessibility | 171 | **0.667** | 0.579 | 0.295 | **0.361** |

### 6.3 Topic Detection

| Metric | Zero-Shot | Fine-Tuned |
|--------|-----------|------------|
| F1 Macro | 0.600 | **0.769** |
| F1 Micro | 0.640 | **0.827** |

### 6.4 Overall Sentiment (majority vote from per-topic sentiments)

| Approach | Accuracy | F1 Macro |
|----------|----------|----------|
| Star Baseline | 0.793 | 0.641 |
| Zero-Shot | 0.716 | 0.607 |
| **Fine-Tuned** | **0.840** | **0.705** |

### 6.5 Agreement (Cohen's Kappa)

| Comparison | Kappa |
|------------|-------|
| Zero-shot vs Fine-tuned | 0.550 |
| Zero-shot vs Baseline | 0.559 |
| Fine-tuned vs Baseline | 0.650 |

---

## 7. Key Findings

### 7.1 Per-topic sentiment is a harder but more useful task

Overall sentiment classification is "easy" — the star baseline achieves 79.3%. Per-topic sentiment requires understanding what a review says about each specific aspect, which is genuinely challenging.

### 7.2 Fine-tuning excels at topic detection but struggles with sentiment nuance

The joint model achieves 82.7% micro F1 for detecting which topics are discussed (vs 64.0% for zero-shot). However, once it detects a topic, its sentiment accuracy is comparable to or worse than zero-shot for most topics. The model learns "what is being talked about" better than "how does the reviewer feel about it." Workload remains particularly challenging with only 195 training examples.

### 7.3 The star baseline remains competitive for overall sentiment

At 79.3% accuracy, mapping stars to sentiment is hard to beat with NLP alone. However, the baseline cannot differentiate between topics — it says a 3-star review is "neutral about everything" when in reality it might be "great lectures, terrible exams." This is where the NLP models provide value.

### 7.4 Zero-shot and fine-tuned models complement each other

An ensemble or routing strategy could use fine-tuned predictions for high-frequency topics (Teaching Quality, Exam Difficulty) and zero-shot for lower-data topics (Workload, Accessibility). This addresses the data scarcity problem without sacrificing performance on common topics.

### 7.5 Aspect-based sentiment reveals what star ratings hide

A professor with a 3.5 average could have excellent lectures but unfair exams, or boring lectures with generous grading. The radar chart visualization makes these differences visible and actionable for course selection — something a single average star rating can never show.

---

## 8. Recommendation Engine

### Aggregation

For each professor, compute mean sentiment score per topic across all their reviews (from zero-shot scored dataset).

### Scoring

User sets importance weights (0–10) for each topic:
```
score = Σ(weight_i × topic_score_i) / Σ(weight_i)
```

Produces a composite score in [-1, 1]. Professors ranked descending.

### Filtering

Minimum review count threshold (default: 3) excludes professors with insufficient data.

---

## 9. Frontend — Streamlit App

Three tabs:
1. **Explore** — browse professors, radar charts of per-topic sentiment, read reviews
2. **Recommend** — set topic preference sliders, get ranked professor list
3. **Model Comparison** — evaluation metrics + live per-topic sentiment comparison

---

## 10. Technical Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.12 |
| Package management | uv |
| ML Framework | PyTorch 2.11 (MPS backend) |
| Transformers | HuggingFace Transformers 5.5 |
| Data | pandas 2.2+, pyarrow |
| Metrics | scikit-learn 1.8+ |
| HTTP client | httpx 0.27+ (async) |
| Frontend | Streamlit 1.33+ |
| Visualization | plotly 5.20+ |
| Compute | MacBook Pro M5, 24GB RAM |
| Labeling | Claude Sonnet 4.6 (8 parallel agents) |

---

## 11. Project Structure

```
course_review/
├── pyproject.toml
├── run_pipeline.py             # end-to-end pipeline runner
├── src/
│   ├── scraper/
│   │   ├── client.py           # RMP GraphQL client (paginated, async, retry)
│   │   ├── parse.py            # normalize API responses → dicts/DataFrames
│   │   ├── preprocess.py       # text cleaning, dedup
│   │   └── run.py              # scraping entrypoint
│   ├── models/
│   │   ├── baseline.py         # star → sentiment mapping
│   │   ├── zero_shot.py        # BART-mnli topics + RoBERTa sentiment
│   │   ├── fine_tune.py        # joint DistilBERT (5×4-class heads)
│   │   ├── labeling.py         # annotation tools + train/test split
│   │   ├── process.py          # batch scoring all reviews
│   │   └── evaluate.py         # per-topic sentiment metrics + comparison
│   ├── recommend/
│   │   └── engine.py           # weighted scoring + filtering
│   └── app/
│       └── streamlit_app.py    # 3-tab web UI
├── tests/                      # 30 tests
├── data/
│   ├── raw/                    # cached API JSON
│   ├── processed/              # parquet files + evaluation results
│   └── labels/                 # Sonnet 4.6 per-topic sentiment annotations
└── models/
    └── joint_classifier/       # saved DistilBERT weights
```

---

## 12. Reproducibility

```bash
uv sync --extra dev                              # install deps
uv run python run_pipeline.py                    # scrape + clean + zero-shot score
uv run python -m src.models.labeling             # label data (interactive CLI)

# Train joint model
uv run python -c "
from pathlib import Path
from src.models.labeling import split_labeled_data
from src.models.fine_tune import train_joint_classifier
train, _ = split_labeled_data(Path('data/processed/reviews.parquet'))
train_joint_classifier(train['review_text'].tolist(), train['label_topics'].tolist())
"

# Evaluate
uv run python -c "
import json
from pathlib import Path
from src.models.labeling import split_labeled_data
from src.models.zero_shot import TopicClassifier, SentimentAnalyzer
from src.models.fine_tune import predict_joint
from src.models.evaluate import compare_approaches
_, test = split_labeled_data(Path('data/processed/reviews.parquet'))
texts = test['review_text'].tolist()
tc = TopicClassifier(); sa = SentimentAnalyzer()
zs = [sa.analyze_by_topic_flat(t, tc) for t in texts]
ft = predict_joint(texts)
results = compare_approaches(test, zs, ft)
Path('data/processed/evaluation_results.json').write_text(json.dumps(results, indent=2, default=str))
"

uv run streamlit run src/app/streamlit_app.py    # launch app
```
