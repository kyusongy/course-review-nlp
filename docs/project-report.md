# UNC Course Compass — Project Report

**Course:** Data 522 — Practical Deep Learning  
**Developer:** Kyle  
**Date:** April 2026  

---

## 1. Project Overview

UNC Course Compass is an end-to-end ML system that scrapes student reviews from RateMyProfessor for UNC's Statistics and Data Science departments, performs aspect-based topic classification and sentiment analysis using three approaches (star rating proxy, zero-shot transformers, fine-tuned DistilBERT), and recommends courses based on student preferences through a Streamlit web app.

### Architecture

```
RateMyProfessor GraphQL API
        │
        ▼
   1,163 raw reviews (22 professors)
        │
        ▼
   Text cleaning + deduplication
        │
        ├──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼
  Star Rating Proxy      Zero-Shot Models       Fine-Tuned DistilBERT
  (no NLP)               (BART + RoBERTa)       (trained on our data)
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
                 Per-topic sentiment scores [-1, 1]
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
            Weighted Scoring         Evaluation
            (recommendations)     (F1, accuracy, kappa)
                    │
                    ▼
              Streamlit App
```

---

## 2. Data Collection

### 2.1 Source: RateMyProfessor GraphQL API

RateMyProfessor is a React/Next.js app backed by a Relay-based GraphQL endpoint at `https://www.ratemyprofessors.com/graphql`. We reverse-engineered the API through a hybrid approach:

1. **Exploration:** Used web fetching to inspect RMP's page structure and discover the GraphQL endpoint, auth mechanism, and query schema.
2. **Verification:** Tested queries directly with curl to confirm field names, pagination behavior, and auth requirements.
3. **Automation:** Wrote an async Python client (`src/scraper/client.py`) using `httpx` to reproduce the same queries at scale.

**Key technical details:**
- Auth header: `Basic dGVzdDp0ZXN0` (base64 of "test:test") — a public token embedded in RMP's frontend JavaScript
- UNC's school ID: `U2Nob29sLTEyMzI=` (base64 of "School-1232")
- Required a browser-like `User-Agent` header — RMP returns 403 for `python-httpx` default UA
- Pagination: cursor-based, 20 ratings per page, with `hasNextPage` / `endCursor` in `pageInfo`

**Scraping process (`src/scraper/run.py`):**
1. One query fetches all 500 professors at UNC
2. Filter to target departments: Statistics (14), Statistics & Ops Research (4), Biostatistics (4)
3. For each of 22 professors, paginate through all their reviews
4. 0.5s delay between requests to avoid rate limiting
5. Retry logic: 3 attempts with exponential backoff on server errors
6. Raw JSON cached per professor in `data/raw/{legacy_id}.json`

### 2.2 Dataset Summary

| Metric | Value |
|--------|-------|
| Total reviews | 1,163 |
| Professors | 22 |
| Departments | Statistics (14), Statistics & Ops Research (4), Biostatistics (4) |
| Avg review length | 285 characters |
| Min / Max review length | 8 / 362 characters |

**Star rating distribution:**

| Stars | Count | Percentage |
|-------|-------|------------|
| 1 | 367 | 31.6% |
| 2 | 138 | 11.9% |
| 3 | 126 | 10.8% |
| 4 | 183 | 15.7% |
| 5 | 349 | 30.0% |

The distribution is bimodal — students tend to leave reviews when they feel strongly (positive or negative), with fewer moderate ratings.

**Fields collected per review:**
- `review_text` — free-text review body
- `star_rating` — 1–5 overall quality rating
- `difficulty_rating` — 1–5 difficulty rating
- `would_take_again` — boolean (or null)
- `course_name` — e.g. "STOR 435"
- `professor_name` — full name
- `date` — review timestamp
- `thumbs_up` / `thumbs_down` — helpfulness votes

### 2.3 Per-Professor Breakdown

| Professor | Reviews | Avg Stars | Department |
|-----------|---------|-----------|------------|
| Jeff McLean | 154 | 3.5 | Statistics |
| Chuanshu Ji | 137 | 2.0 | Statistics |
| Prairie Goodwin | 119 | 2.8 | Statistics |
| Gabor Pataki | 116 | 3.2 | Statistics |
| Oluremi Abayomi | 115 | 3.5 | Statistics & Ops Research |
| William Lassiter | 103 | 4.0 | Statistics |
| Teressa Bergland | 99 | 2.0 | Statistics & Ops Research |
| Jane Monaco | 82 | 4.2 | Biostatistics |
| Kendall Thomas | 57 | 2.4 | Statistics |
| Vladas Pipiras | 34 | 3.3 | Statistics |
| Kyung Kim | 29 | 1.1 | Statistics |
| Mariana Olvera-Cravioto | 26 | 3.5 | Statistics |
| Ali Nezhad | 22 | 2.8 | Statistics |
| William McCance | 19 | 2.2 | Statistics |
| Andrew Nobel | 18 | 2.0 | Statistics |
| Richard Smith | 10 | 4.1 | Statistics |
| Jianqing Jia | 8 | 5.0 | Statistics & Ops Research |
| Vincent Toups | 5 | 1.6 | Biostatistics |
| Kara McCormack | 5 | 4.8 | Biostatistics |
| Fan Yao | 2 | 4.5 | Statistics & Ops Research |
| Ishmael Benjamin Torres Aguilar | 2 | 3.5 | Statistics |
| Kinsey Helton | 1 | 4.0 | Biostatistics |

---

## 3. Data Preprocessing

Preprocessing was intentionally minimal (`src/scraper/preprocess.py`). Transformer models handle messy text well, so aggressive cleaning (lowercasing, stopword removal, stemming) would hurt more than help.

**Steps applied:**
1. **Whitespace normalization:** Collapsed all whitespace (newlines, tabs, multiple spaces) to single spaces using `re.sub(r"\s+", " ", text).strip()`
2. **Empty review removal:** Dropped reviews with no text after cleaning
3. **Deduplication:** Removed exact duplicates on `(review_text, professor_name, course_name)` — catches reviews submitted multiple times

**Result:** 1,163 → 1,163 reviews (no duplicates or empties found in this dataset — RMP data was already clean).

**What we intentionally did NOT do:**
- No lowercasing — BERT-family models have cased and uncased variants; we used uncased DistilBERT which handles this internally
- No stopword removal — transformers use attention mechanisms that learn to ignore irrelevant words
- No stemming/lemmatization — subword tokenization (WordPiece/BPE) makes this unnecessary
- No spell correction — would risk changing meaning, and models are robust to minor misspellings

---

## 4. Labeling

### 4.1 Topic Taxonomy

We defined 6 aspect categories for course reviews:

| Topic | Description | Example Phrases |
|-------|-------------|-----------------|
| Workload | Homework volume, time commitment | "tons of homework", "manageable workload" |
| Grading | Fairness, curves, grade distribution | "harsh grader", "generous curve" |
| Teaching Quality | Lecture clarity, engagement, responsiveness | "explains well", "boring lectures" |
| Course Content | Relevance, interest, organization | "great material", "outdated content" |
| Accessibility | Office hours, approachability | "always available", "hard to reach" |
| Exam Difficulty | Test format, fairness, prep alignment | "exams are fair", "nothing like homework" |

Each review is **multi-labeled** — a single review can discuss multiple topics (e.g., "Great lectures but brutal exams" → Teaching Quality + Exam Difficulty).

### 4.2 Labeling Process

All 1,163 reviews were labeled using **Claude Sonnet 4.6** as an automated annotator. We dispatched 4 parallel labeling agents, each processing ~290 reviews:

- Each agent read the raw review text (no model predictions shown — independent judgment)
- Assigned 0–6 topic labels per review based on what the review clearly discusses
- Assigned overall sentiment (positive / neutral / negative) based on overall tone

This approach provides genuine ground-truth labels that are independent of our zero-shot and fine-tuned models, enabling fair evaluation.

### 4.3 Label Distribution

**Sentiment:**

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Negative | 524 | 45.1% |
| Positive | 489 | 42.0% |
| Neutral | 150 | 12.9% |

**Topic frequency:**

| Topic | Count | % of Reviews |
|-------|-------|-------------|
| Teaching Quality | 970 | 83.4% |
| Exam Difficulty | 650 | 55.9% |
| Grading | 512 | 44.0% |
| Workload | 485 | 41.7% |
| Accessibility | 338 | 29.1% |
| Course Content | 235 | 20.2% |

**Average topics per review:** 2.74

Teaching Quality is discussed in 83% of reviews — students almost always comment on the professor's teaching. Course Content is the rarest (20%) — students are more likely to comment on how the course is run than on the material itself.

### 4.4 Train/Test Split

- **Train:** 814 reviews (70%)
- **Test:** 349 reviews (30%)
- Split is stratified to preserve sentiment class proportions

---

## 5. Model Architecture

### 5.1 Layer 1 — Star Rating Baseline (`src/models/baseline.py`)

The simplest possible approach. No NLP involved.

**Method:**
- Stars 1–2 → "negative" (score: -1.0)
- Star 3 → "neutral" (score: 0.0)
- Stars 4–5 → "positive" (score: +1.0)

**Purpose:** Establishes the floor — "can NLP models beat just looking at the number of stars?" The baseline assigns the same sentiment to every topic for a given review (no aspect differentiation).

### 5.2 Layer 2 — Zero-Shot Transformer Models (`src/models/zero_shot.py`)

Two pretrained models used **without any fine-tuning on our data**:

#### Topic Classification: `facebook/bart-large-mnli`

- **Architecture:** BART-large (406M parameters), a sequence-to-sequence transformer
- **Pretraining:** Fine-tuned on Multi-Genre Natural Language Inference (MNLI) — 433K sentence pairs labeled as entailment/contradiction/neutral
- **Zero-shot mechanism:** Reformulates topic classification as textual entailment. For each topic, the model asks: "Does this review text entail 'homework volume and time commitment'?" A high entailment score means the review discusses that topic.
- **Key detail:** We used descriptive phrases ("homework volume and time commitment") rather than single words ("Workload") as candidate labels — the model understands natural language descriptions much better than category names
- **Multi-label:** `multi_label=True` means each topic is scored independently, so a review can match multiple topics
- **Threshold:** 0.3 confidence — any topic above this is assigned

#### Sentiment Analysis: `cardiffnlp/twitter-roberta-base-sentiment-latest`

- **Architecture:** RoBERTa-base (125M parameters)
- **Pretraining:** Fine-tuned on ~124M tweets for 3-class sentiment classification
- **Output:** Probability distribution over negative/neutral/positive
- **Composite score:** `positive_prob - negative_prob` → range [-1, 1]
- **Classification thresholds:** > 0.25 = positive, < -0.25 = negative, else neutral
- **Input truncation:** 512 characters (model's max sequence length)

#### Aspect-Based Sentiment (`analyze_by_topic`)

This combines both models to get **sentiment per topic**:

1. Split review into sentences (regex split on `.!?`)
2. For each sentence, run topic classification → which topics does this sentence discuss?
3. For each sentence, run sentiment analysis → positive/neutral/negative with score
4. Assign each sentence's sentiment to its detected topics
5. Average sentiment scores per topic across all sentences in the review

**Example:** *"The lectures were amazing but the exams were brutal"*
- Sentence 1: "The lectures were amazing" → Topic: Teaching Quality, Score: +0.82
- Sentence 2: "the exams were brutal" → Topic: Exam Difficulty, Score: -0.71
- Result: `{Teaching Quality: +0.82, Exam Difficulty: -0.71}`

### 5.3 Layer 3 — Fine-Tuned DistilBERT (`src/models/fine_tune.py`)

#### Why DistilBERT?

DistilBERT is a distilled version of BERT — 66M parameters (6 transformer layers vs BERT's 12). It retains 97% of BERT's language understanding while being 60% faster and 40% smaller. This makes it practical to train on a MacBook Pro M5 with 24GB RAM using PyTorch's MPS (Metal Performance Shaders) backend.

#### Topic Classifier — Multi-Label Classification

- **Base model:** `distilbert-base-uncased`
- **Classification head:** Linear layer → 6 sigmoid outputs (one per topic)
- **Loss function:** Binary cross-entropy (each topic is an independent binary decision)
- **Why sigmoid + BCE, not softmax + CE?** Because a review can discuss multiple topics simultaneously. Softmax forces probabilities to sum to 1 (mutually exclusive), which is wrong for multi-label. Sigmoid treats each topic independently.
- **Training data:** 814 reviews with multi-hot topic labels from Sonnet 4.6 annotation
- **Tokenization:** DistilBERT WordPiece tokenizer, max_length=256 tokens, padding=True, truncation=True

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Optimizer | AdamW |
| Scheduler | Linear warmup (0 warmup steps) |
| Batch size | 16 |
| Epochs | 5 |
| Max sequence length | 256 tokens |

**Training loss:** 0.617 → 0.392 over 5 epochs

**Prediction threshold:** 0.5 — any topic with sigmoid output > 0.5 is assigned

#### Sentiment Classifier — Single-Label 3-Class

- **Base model:** `distilbert-base-uncased`
- **Classification head:** Linear layer → 3 softmax outputs
- **Loss function:** Cross-entropy (standard multiclass)
- **Labels:** positive (0), neutral (1), negative (2)
- **Same hyperparameters as topic classifier**

**Training loss:** 0.852 → 0.150 over 5 epochs (strong convergence — sentiment patterns are more consistent than topic patterns)

---

## 6. Batch Processing (`src/models/process.py`)

All 1,163 reviews were processed through both the baseline and zero-shot pipeline, producing a 25-column scored dataset:

**Columns:**
- `idx`, `professor_name`, `course_name`, `review_text`, `star_rating`
- `overall_sentiment` (zero-shot label), `overall_score` (zero-shot composite [-1,1])
- For each of 6 topics:
  - `topic_{key}_conf` — zero-shot classification confidence [0, 1]
  - `topic_{key}_sentiment` — zero-shot sentiment label (positive/neutral/negative or null)
  - `topic_{key}_score` — zero-shot sentiment score [-1, 1] (or null if topic not detected)

**Zero-shot overall sentiment distribution:**

| Sentiment | Count | Percentage |
|-----------|-------|------------|
| Negative | 560 | 48.2% |
| Positive | 516 | 44.4% |
| Neutral | 87 | 7.5% |

Note: The zero-shot model produces far fewer "neutral" labels (7.5%) compared to the Sonnet 4.6 ground truth (12.9%). The zero-shot model is more decisive — it tends to push ambiguous reviews toward positive or negative rather than neutral.

---

## 7. Evaluation Results

### 7.1 Methodology

- **Test set:** 349 reviews (30% holdout, stratified by sentiment)
- **Ground truth:** Sonnet 4.6 labels (independent of all three model approaches)
- All three approaches were evaluated on the same test set:
  1. Star rating baseline
  2. Zero-shot (BART-mnli + RoBERTa) — run fresh on test texts
  3. Fine-tuned DistilBERT — loaded from saved model, run on test texts

### 7.2 Topic Classification

| Topic | Zero-Shot F1 | Fine-Tuned F1 | Winner |
|-------|-------------|---------------|--------|
| Teaching Quality | 0.884 | **0.911** | Fine-tuned |
| Exam Difficulty | 0.777 | **0.915** | Fine-tuned |
| Workload | 0.669 | **0.793** | Fine-tuned |
| Grading | 0.566 | **0.809** | Fine-tuned |
| Accessibility | **0.470** | 0.173 | Zero-shot |
| Course Content | **0.350** | 0.127 | Zero-shot |
| **Macro F1** | 0.619 | 0.621 | ~Tied |
| **Micro F1** | 0.647 | **0.792** | Fine-tuned |

**Key observations:**
- Fine-tuning dramatically improves **high-frequency topics**: Grading (0.566 → 0.809), Exam Difficulty (0.777 → 0.915), Workload (0.669 → 0.793)
- Fine-tuning **hurts low-frequency topics**: Accessibility (0.470 → 0.173), Course Content (0.350 → 0.127)
- This is a classic **class imbalance problem** — with only 235 Course Content examples in the full dataset (~165 in training), the model doesn't see enough positive examples to learn the pattern
- Zero-shot is better for rare categories because it uses general language understanding rather than memorizing patterns from limited examples
- **Micro F1** (weighted by sample count) strongly favors fine-tuned (0.792 vs 0.647) because it does well on the topics that appear most often

### 7.3 Sentiment Analysis

| Approach | Accuracy | F1 Macro | F1 Negative | F1 Neutral | F1 Positive |
|----------|----------|----------|-------------|------------|-------------|
| Star Baseline | **0.871** | 0.747 | 0.927 | 0.384 | 0.929 |
| Zero-Shot | 0.840 | 0.690 | 0.902 | 0.269 | 0.899 |
| Fine-Tuned | 0.860 | **0.750** | **0.932** | **0.427** | 0.892 |

**Confusion matrices (rows = true, columns = predicted):**

**Star Baseline:**
```
              Predicted
              neg  neu  pos
True neg  [  140   14    0 ]
True neu  [    8   14   17 ]
True pos  [    0    6  150 ]
```

**Zero-Shot:**
```
              Predicted
              neg  neu  pos
True neg  [  142    9    3 ]
True neu  [   15    9   15 ]
True pos  [    4   10  142 ]
```

**Fine-Tuned:**
```
              Predicted
              neg  neu  pos
True neg  [  144    6    4 ]
True neu  [    9   16   14 ]
True pos  [    2   14  140 ]
```

**Key observations:**
- The star baseline is **surprisingly strong** (87.1% accuracy) — students' star ratings align well with their text sentiment
- All models struggle with **neutral** reviews (F1: 0.27–0.43) — the hardest class because neutral reviews are rare (11.2% of test set) and inherently ambiguous
- Fine-tuned has the **best F1 macro** (0.750) because it handles neutral better (0.427 vs baseline's 0.384 vs zero-shot's 0.269)
- The zero-shot model is the **worst at neutral** (0.269 F1) — it was trained on tweets, where neutral expressions may differ from academic course reviews
- The star baseline has **zero false negatives for positive** (never predicts a positive review as negative) but leaks 17 neutrals into positive — it can't detect mixed reviews

### 7.4 Inter-Approach Agreement (Cohen's Kappa)

| Comparison | Kappa |
|------------|-------|
| Zero-shot vs Fine-tuned | 0.738 |
| Zero-shot vs Star Baseline | 0.712 |
| Fine-tuned vs Star Baseline | 0.721 |

All pairs show **substantial agreement** (kappa 0.61–0.80 range). The highest agreement is between zero-shot and fine-tuned (0.738), which makes sense — the fine-tuned model was trained on labels from the same domain and learns similar patterns.

---

## 8. Recommendation Engine (`src/recommend/engine.py`)

### 8.1 Aggregation

For each professor, we compute the mean sentiment score per topic across all their reviews:

```
aggregate_professor_scores(reviews_df) →
  professor_name | num_reviews | workload | grading | teaching_quality | ...
  Jeff McLean    | 154         | -0.12    | -0.34   | 0.18             | ...
```

### 8.2 Scoring

User sets importance weights (0–10) for each of the 6 topics via sliders. The recommendation score for each professor is:

```
score = Σ(weight_i × topic_score_i) / Σ(weight_i)
```

This produces a composite score in [-1, 1]. Professors are ranked by score descending.

### 8.3 Filtering

- Minimum review count threshold (default: 3) to exclude professors with insufficient data
- Optional course-level prefix filter

---

## 9. Frontend — Streamlit App (`src/app/streamlit_app.py`)

Three-tab interface:

**Tab 1: Explore**
- Dropdown to select professor
- Left panel: review count, average star rating, average difficulty, plotly radar chart showing 6-topic sentiment profile
- Right panel: scrollable review list with star display and course names

**Tab 2: Recommend**
- 6 sliders for topic importance (0–10)
- Minimum review threshold slider
- Ranked list of top 10 professors with composite scores
- Expandable cards with radar charts per professor

**Tab 3: Model Comparison**
- Evaluation metrics tables (topic F1, sentiment accuracy)
- Live analysis: paste any review text, see side-by-side predictions from zero-shot vs fine-tuned models

---

## 10. Technical Stack

| Component | Tool | Version |
|-----------|------|---------|
| Language | Python | 3.12 |
| Package management | uv | — |
| ML Framework | PyTorch (MPS backend) | 2.11 |
| Transformers | HuggingFace Transformers | 5.5 |
| Data processing | pandas + pyarrow | 2.2+ |
| Evaluation metrics | scikit-learn | 1.8+ |
| HTTP client | httpx (async) | 0.27+ |
| Frontend | Streamlit | 1.33+ |
| Visualization | plotly | 5.20+ |
| Compute | MacBook Pro M5 (24GB, MPS) | — |

---

## 11. Project Structure

```
course_review/
├── pyproject.toml              # dependencies and project config
├── run_pipeline.py             # end-to-end pipeline runner
├── src/
│   ├── scraper/
│   │   ├── client.py           # RMP GraphQL API client (async, retry, cache)
│   │   ├── parse.py            # normalize API responses → standardized dicts
│   │   ├── preprocess.py       # text cleaning, dedup
│   │   └── run.py              # scraping entrypoint
│   ├── models/
│   │   ├── baseline.py         # star → sentiment mapping
│   │   ├── zero_shot.py        # BART-mnli topic + RoBERTa sentiment
│   │   ├── fine_tune.py        # DistilBERT training + prediction
│   │   ├── labeling.py         # annotation tools + train/test split
│   │   ├── process.py          # batch scoring all reviews
│   │   └── evaluate.py         # metrics + 3-way comparison
│   ├── recommend/
│   │   └── engine.py           # weighted scoring + filtering
│   └── app/
│       └── streamlit_app.py    # 3-tab web UI
├── tests/
│   ├── test_scraper.py         # 15 tests
│   ├── test_models.py          # 10 tests
│   └── test_recommend.py       # 3 tests
├── data/
│   ├── raw/                    # cached API JSON (per professor)
│   ├── processed/              # parquet files + evaluation JSON
│   └── labels/                 # Sonnet 4.6 annotations
├── models/                     # saved DistilBERT weights
└── docs/
    ├── project-report.md       # this file
    └── superpowers/
        ├── specs/              # design specification
        └── plans/              # implementation plan
```

---

## 12. Reproducibility

To reproduce from scratch:

```bash
# 1. Install dependencies
uv sync --extra dev

# 2. Run pipeline (scrape + clean + zero-shot scoring)
uv run python run_pipeline.py

# 3. Label data (interactive CLI, or use auto_label_from_zero_shot)
uv run python -m src.models.labeling

# 4. Fine-tune models
uv run python -c "
from pathlib import Path
from src.models.labeling import split_labeled_data
from src.models.fine_tune import train_topic_classifier, train_sentiment_classifier
train, test = split_labeled_data(Path('data/processed/zero_shot_scores.parquet'))
train_topic_classifier(train['review_text'].tolist(), train['label_topics'].tolist())
train_sentiment_classifier(train['review_text'].tolist(), train['label_sentiment'].tolist())
"

# 5. Run evaluation
uv run python -c "
import json
from pathlib import Path
from src.models.labeling import split_labeled_data
from src.models.zero_shot import TopicClassifier, SentimentAnalyzer
from src.models.fine_tune import predict_topics, predict_sentiment
from src.models.evaluate import compare_approaches
_, test = split_labeled_data(Path('data/processed/zero_shot_scores.parquet'))
texts = test['review_text'].tolist()
tc = TopicClassifier(); sa = SentimentAnalyzer()
zs_topics = [tc.classify(t) for t in texts]
zs_sents = [sa.analyze(t)['label'] for t in texts]
ft_topics = predict_topics(texts, 'models/topic_classifier')
ft_sents = [r['label'] for r in predict_sentiment(texts, 'models/sentiment_classifier')]
results = compare_approaches(test, zs_topics, zs_sents, ft_topics, ft_sents)
Path('data/processed/evaluation_results.json').write_text(json.dumps(results, indent=2, default=str))
"

# 6. Launch app
uv run streamlit run src/app/streamlit_app.py
```

---

## 13. Key Findings

1. **Fine-tuning helps most where data is abundant.** Topics with 400+ training examples (Teaching Quality, Exam Difficulty, Grading, Workload) saw F1 improvements of 5–24 percentage points. Topics with <300 examples (Accessibility, Course Content) degraded significantly.

2. **Zero-shot is the safer choice for rare categories.** General language understanding from BART-mnli outperforms a fine-tuned model that hasn't seen enough positive examples of a category.

3. **The star baseline is hard to beat on sentiment.** At 87.1% accuracy, simply mapping star ratings to sentiment labels performs remarkably well — students' numeric ratings align with their text sentiment. The fine-tuned model edges it out on F1 macro (0.750 vs 0.747) primarily by handling the neutral class better.

4. **Neutral is the hardest class.** All three models struggle with neutral reviews (best F1: 0.427 from fine-tuned). Neutral reviews are rare (12.9% of data) and inherently ambiguous — a review saying "it was fine" is hard to distinguish from a mildly positive one.

5. **Aspect-based sentiment reveals what star ratings hide.** A professor with a 3.5 average rating could have excellent lectures but unfair exams. The radar chart visualization makes these differences visible and actionable for course selection.

6. **Cohen's kappa shows all models substantially agree (0.71–0.74).** Despite different architectures and training, the three approaches converge on similar predictions for most reviews. Disagreements concentrate in the neutral class and edge cases.
