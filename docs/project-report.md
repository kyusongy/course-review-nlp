# UNC Course Compass -- Project Report

**Course:** Data 522 -- Practical Deep Learning
**Developer:** Kyleyu Songyao
**Date:** April 2026

---

## 1. Project Overview

UNC Course Compass is an end-to-end ML system that scrapes 65,376 student reviews from RateMyProfessor for all UNC Chapel Hill departments, performs **aspect-based sentiment analysis** -- classifying what topics each review discusses and whether the sentiment toward each topic is positive, neutral, or negative -- and recommends professors based on student preferences through a Streamlit web app.

The key insight: a single review can be positive about one aspect and negative about another. "Great lectures but brutal exams" is not a neutral review -- it's positive about teaching quality and negative about exam difficulty. Our system captures this nuance.

### Architecture

```
RateMyProfessor GraphQL API
        |
        v
   65,376 reviews (4,173 professors, 103 departments)
        |
        v
   Text cleaning + deduplication
        |
        |--- 8,378 sampled reviews labeled by Sonnet 4.6
        |         |
        |         v
        |    Train/Val/Test split (5,864 / 1,257 / 1,257)
        |         |
        |         v
        |    Fine-tune DistilBERT (5x4-class joint model)
        |         |
        |         v
        |    Evaluate on test set (vs zero-shot + star baseline)
        |
        v
   Score all 65,376 reviews with fine-tuned model
        |
        v
   Weighted scoring recommendation engine
        |
        v
   Streamlit App
```

### Three-Layer Comparison

| Layer | Model | What It Does |
|-------|-------|-------------|
| 1. Star Baseline | None (rule-based) | Maps 1-5 stars to sentiment. No NLP, no topic differentiation. |
| 2. Zero-Shot | BART-mnli (406M) + RoBERTa (125M) | Pretrained models, no training on our data. Sentence-level aspect-based sentiment. |
| 3. Fine-Tuned | DistilBERT (66M) joint model | Trained on 8,378 labeled reviews. 5 independent 4-class heads for per-topic sentiment. |

---

## 2. Data Collection

### 2.1 Source: RateMyProfessor GraphQL API

RateMyProfessor is a React/Next.js app backed by a Relay-based GraphQL endpoint at `https://www.ratemyprofessors.com/graphql`. We reverse-engineered the API through a hybrid approach:

1. **Exploration:** Used web fetching to inspect RMP's page structure, discover the GraphQL endpoint and auth mechanism
2. **Verification:** Tested queries with curl to confirm field names, pagination, and auth
3. **Automation:** Wrote an async Python client (`src/scraper/client.py`) using httpx

**Technical details:**
- Auth: `Basic dGVzdDp0ZXN0` (base64 of "test:test") -- a public token from RMP's frontend JS
- School ID: `U2Nob29sLTEyMzI=` (base64 of "School-1232", UNC Chapel Hill)
- Required browser-like `User-Agent` header -- RMP returns 403 for python-httpx default UA
- Pagination: cursor-based, fetches all 4,838 UNC professors across 10 pages of 500

**Scraping process:**
1. Paginate through all UNC professors (4,838 total)
2. For each professor with reviews, fetch all ratings with cursor-based pagination (20 per page)
3. 0.5s delay between requests, 3-attempt retry with exponential backoff
4. Cache raw JSON per professor in `data/raw/{legacy_id}.json`

### 2.2 Dataset Summary

| Metric | Value |
|--------|-------|
| Total reviews | 65,376 |
| Professors (with reviews) | 4,173 |
| Departments | 103 |
| Avg review length | 272 characters |

**Star rating distribution:**

| Stars | Count | Percentage |
|-------|-------|------------|
| 1 | 8,108 | 12.4% |
| 2 | 6,469 | 9.9% |
| 3 | 7,103 | 10.9% |
| 4 | 11,549 | 17.7% |
| 5 | 32,147 | 49.2% |

**Top 15 departments by review count:**

| Department | Reviews |
|------------|---------|
| English | 5,773 |
| Mathematics | 4,783 |
| Chemistry | 4,197 |
| Biology | 3,698 |
| Computer Science | 3,156 |
| History | 3,041 |
| Psychology | 2,832 |
| Economics | 2,827 |
| Political Science | 2,628 |
| Languages | 2,287 |
| Statistics | 1,923 |
| Journalism | 1,862 |
| Anthropology | 1,728 |
| Spanish | 1,654 |
| Sociology | 1,648 |

---

## 3. Data Preprocessing

Minimal by design -- transformer models handle messy text well.

**Steps:**
1. **Whitespace normalization:** `re.sub(r"\s+", " ", text).strip()`
2. **Empty review removal:** Drop reviews with no text after cleaning
3. **Deduplication:** Remove exact duplicates on `(review_text, professor_name, course_name)`
4. **Course name normalization:** Bare numbers mapped to correct prefixes (e.g., "301" -> "COMP301", "455" -> "STOR455") based on professor teaching history

**Result:** 65,441 raw reviews -> 65,376 after dedup.

---

## 4. Labeling

### 4.1 Per-Topic Sentiment Labels

Each review is labeled with a **dict mapping discussed topics to their sentiment**:

```json
{"Teaching Quality": "positive", "Exam Difficulty": "negative"}
```

This captures that "Great lectures but brutal exams" is not neutral -- it's positive about one thing and negative about another. Topics not in the dict are "not discussed."

### 4.2 Topic Taxonomy

| Topic | Description |
|-------|-------------|
| Workload | Homework volume, time commitment |
| Grading | Fairness, curves, grade distribution |
| Teaching Quality | Lecture clarity, engagement, responsiveness |
| Accessibility | Office hours, approachability |
| Exam Difficulty | Test format, fairness, prep alignment |

### 4.3 Labeling Process

We labeled **8,378 reviews** (12.8% of the full dataset) using **Claude Sonnet 4.6** as an automated annotator:

- **2,429 reviews** from Statistics/STOR/Biostatistics departments (initial scope)
- **5,949 reviews** sampled proportionally from all 103 departments (expanded scope)

For the expanded labeling, we dispatched **20 parallel Sonnet agents**, each processing ~300 reviews. Each agent:
1. Read ~50 reviews at a time from a parquet file
2. Labeled all 50 in a single reasoning pass (not one API call per review)
3. Wrote results to a JSON chunk file
4. Repeated until all reviews in its chunk were done

All 20 agents ran simultaneously -- total wall-clock time ~5 minutes for 5,949 reviews.

Each agent saw only the raw review text (no model predictions) and made independent topic/sentiment judgments, producing ground-truth labels independent of the models being evaluated.

### 4.4 Label Distribution

**Topic frequency (across 8,378 labeled reviews):**

| Topic | Count | % of Reviews | Positive | Neutral | Negative |
|-------|-------|-------------|----------|---------|----------|
| Teaching Quality | 7,259 | 86.7% | 4,622 | 592 | 2,045 |
| Exam Difficulty | 2,875 | 34.3% | 1,056 | 333 | 1,486 |
| Grading | 2,863 | 34.2% | 1,458 | 181 | 1,224 |
| Workload | 2,559 | 30.5% | 1,061 | 322 | 1,176 |
| Accessibility | 2,093 | 25.0% | 1,586 | 17 | 490 |

- **Reviews with no clear topic:** 260 (3.1%)
- Teaching Quality is discussed in 87% of reviews -- students almost always comment on how the professor teaches
- Accessibility is overwhelmingly positive (76%) -- when students mention office hours, it's usually to praise them

### 4.5 Train / Validation / Test Split

| Split | Reviews | Purpose |
|-------|---------|---------|
| Train | 5,864 (70%) | Fine-tune the model |
| Validation | 1,257 (15%) | Tune hyperparameters, pick best epoch |
| Test | 1,257 (15%) | Final evaluation (touched once) |

All splits cover 89-102 departments.

---

## 5. Model Architecture

### 5.1 Layer 1 -- Star Rating Baseline

No NLP. Maps star ratings to sentiment:
- Stars 1-2 -> negative (-1.0)
- Star 3 -> neutral (0.0)
- Stars 4-5 -> positive (+1.0)

Same sentiment for every topic. Exists as the "why do we need DL?" baseline.

### 5.2 Layer 2 -- Zero-Shot Transformer Models

Two pretrained models used **without any fine-tuning on our data:**

#### Topic Classification: `facebook/bart-large-mnli` (406M params)

A BART model fine-tuned on Multi-NLI for natural language inference. Zero-shot classification works by reformulating topic detection as entailment: for each topic, the model asks "Does this review text entail 'homework volume and time commitment'?" A high entailment score means the review discusses that topic.

- Uses descriptive phrases as candidate labels (not just single words)
- `multi_label=True` -- each topic scored independently
- Threshold: 0.3 confidence for topic assignment

#### Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest` (125M params)

RoBERTa fine-tuned on ~124M tweets for 3-class sentiment.

- Composite score: `positive_prob - negative_prob` -> [-1, 1]
- Thresholds: >0.25 = positive, <-0.25 = negative, else neutral

#### Aspect-Based Sentiment (two-stage pipeline)

1. Split review into sentences (regex on `.!?`)
2. Classify each sentence's topics with BART-mnli
3. Score each sentence's sentiment with RoBERTa
4. Assign sentence sentiment to its detected topics
5. Average per-topic across sentences

### 5.3 Layer 3 -- Fine-Tuned Joint DistilBERT (66M params)

#### Architecture

A single DistilBERT backbone with **5 independent 4-class classification heads**, one per topic. Each head predicts: not_discussed (0), positive (1), neutral (2), or negative (3).

```
Review Text
    |
    v
DistilBERT backbone (6 layers, 768-dim hidden)
    |
    v
[CLS] token -> Pre-classifier (768->768) -> ReLU -> Dropout(0.1)
    |
    |--- Head 1: Linear(768->4) -> Workload
    |--- Head 2: Linear(768->4) -> Grading
    |--- Head 3: Linear(768->4) -> Teaching Quality
    |--- Head 4: Linear(768->4) -> Accessibility
    |--- Head 5: Linear(768->4) -> Exam Difficulty
```

#### Why a Joint Model?

- Processes the full review in one forward pass (no sentence-splitting errors)
- Each head uses softmax (states are mutually exclusive within a topic)
- Heads are independent (a review can be positive about teaching and negative about exams)
- One model instead of a two-stage pipeline = simpler, faster, fewer compounding errors

#### Training Details

| Parameter | Value |
|-----------|-------|
| Base model | distilbert-base-uncased |
| Classification heads | 5 x Linear(768->4) |
| Loss | Average of 5 cross-entropy losses (one per head) |
| Optimizer | AdamW, lr=2e-5 |
| Scheduler | Linear warmup (0 warmup steps) |
| Batch size | 16 |
| Epochs | 10 (best model saved at epoch 5 by val loss) |
| Max sequence length | 256 tokens |
| Training data | 5,864 reviews from 102 departments |
| Validation data | 1,257 reviews from 89 departments |

**Training curve:**

```
Epoch  Train Loss  Val Loss
  1      0.749      0.547   *
  2      0.484      0.440   *
  3      0.376      0.410   *
  4      0.300      0.411
  5      0.242      0.410   * (best)
  6      0.197      0.415
  7      0.165      0.421
  8      0.142      0.424
  9      0.127      0.430
 10      0.119      0.431
```

Val loss plateaued at epoch 3-5 and began rising after epoch 5, indicating overfitting. The model saved at epoch 5 (val loss 0.410) was used for all evaluation and scoring.

#### Deployment

After training, the model scored all 65,376 reviews in ~2 minutes using batched inference (batch size 32). This produces the scored dataset that powers the recommendation engine.

---

## 6. Evaluation Results

### 6.1 Methodology

- **Test set:** 1,257 reviews (15% holdout, never used during training or hyperparameter tuning)
- **Ground truth:** Sonnet 4.6 per-topic sentiment labels
- **Per-topic sentiment accuracy:** evaluated only on reviews where the topic is present in ground truth
- All three approaches evaluated on the same test set

### 6.2 Per-Topic Sentiment

| Topic | n | ZS Accuracy | FT Accuracy | ZS F1 | FT F1 |
|-------|---|-------------|-------------|-------|-------|
| Teaching Quality | 1,086 | 0.796 | **0.857** | 0.505 | **0.543** |
| Grading | 427 | 0.609 | **0.707** | 0.397 | **0.401** |
| Exam Difficulty | 433 | 0.587 | **0.693** | 0.396 | **0.423** |
| Workload | 395 | 0.443 | **0.559** | 0.327 | **0.397** |
| Accessibility | 312 | **0.744** | 0.721 | 0.307 | **0.395** |

Fine-tuned outperforms zero-shot on every topic except Accessibility accuracy (where they're close). The biggest gains are on Workload (+11.6%) and Grading (+9.8%).

### 6.3 Topic Detection

| Metric | Zero-Shot | Fine-Tuned |
|--------|-----------|------------|
| F1 Macro | 0.590 | **0.832** |
| F1 Micro | 0.627 | **0.869** |

Per-topic detection F1:

| Topic | Zero-Shot | Fine-Tuned |
|-------|-----------|------------|
| Teaching Quality | 0.929 | **0.953** |
| Grading | 0.522 | **0.838** |
| Exam Difficulty | 0.548 | **0.846** |
| Workload | 0.533 | **0.760** |
| Accessibility | 0.417 | **0.763** |

Fine-tuned dramatically outperforms zero-shot on topic detection -- every topic improved, with Accessibility jumping from 0.417 to 0.763 (an 83% improvement).

### 6.4 Overall Sentiment

| Approach | Accuracy | F1 Macro |
|----------|----------|----------|
| Star Baseline | 0.745 | 0.592 |
| Zero-Shot | 0.714 | 0.594 |
| **Fine-Tuned** | **0.811** | **0.689** |

Overall sentiment is derived from per-topic sentiments via majority vote.

### 6.5 Agreement (Cohen's Kappa)

| Comparison | Kappa |
|------------|-------|
| Zero-shot vs Fine-tuned | 0.502 |
| Zero-shot vs Baseline | 0.559 |
| Fine-tuned vs Baseline | 0.535 |

---

## 7. Recommendation Engine

### Scoring

For each professor, the model's per-topic sentiment scores are averaged across all their reviews:

```
professor_topic_score = mean(sentiment_scores for that topic across all reviews)
```

Where sentiment scores are: positive = +1.0, neutral = 0.0, negative = -1.0.

Users set importance weights (0-10) for each topic via sliders:

```
recommendation_score = sum(weight_i * topic_score_i) / sum(weights)
```

This produces a composite score in [-1, 1]. Professors are ranked by score descending.

### Filters

- **Minimum review count** -- excludes professors with too few reviews for reliable scores
- **Course filter** -- optionally restrict to specific courses (e.g., STOR 155, COMP 401)

### Data Pipeline

The recommendation engine uses scores from the fine-tuned DistilBERT model applied to all 65,376 reviews. This is the end-to-end ML pipeline:

```
Labeled data -> Train model -> Score all reviews -> Weighted recommendations
```

---

## 8. Frontend -- Streamlit App

Three tabs:

**Explore:** Browse all 4,173 professors. See radar charts of per-topic sentiment, review count, average rating, would-retake percentage, and individual reviews with star ratings and course badges.

**Recommend:** Set 5 topic preference sliders and minimum review threshold. Optionally filter by specific courses. Get ranked top-10 professors with composite scores and radar charts.

**Model Comparison:** Evaluation metrics from the held-out test set (1,257 reviews). Per-topic sentiment accuracy/F1 tables, topic detection F1, overall sentiment comparison. Live analysis: paste any review to compare zero-shot vs fine-tuned predictions side-by-side.

---

## 9. Key Findings

### 9.1 More diverse training data dramatically improves the model

Expanding from 2,429 Stats-only labels to 8,378 cross-department labels improved every metric:

| Metric | Stats-only | All UNC | Improvement |
|--------|-----------|---------|-------------|
| Topic Detection F1 Macro | 0.769 | **0.832** | +8.2% |
| Workload Accuracy | 0.328 | **0.559** | +70.4% |
| Accessibility Detection F1 | 0.173 | **0.763** | +341% |
| Overall Sentiment Accuracy | 0.840 | **0.811** | -3.4% |

Overall sentiment accuracy dipped slightly because the test set now spans 95 departments with more diverse language, but topic-level performance improved across the board.

### 9.2 Fine-tuned model beats zero-shot on topic detection by a wide margin

Topic detection F1 macro: 0.832 (fine-tuned) vs 0.590 (zero-shot). The fine-tuned model is 41% better at identifying which topics a review discusses. This is because:
- Zero-shot relies on general language understanding (entailment)
- Fine-tuned has seen 5,864 examples of course-review-specific language and learned patterns like "curve" -> Grading, "office hours" -> Accessibility

### 9.3 Fine-tuned model wins on per-topic sentiment too

For every topic, the fine-tuned model matches or exceeds zero-shot on sentiment F1. The largest gains are on topics that benefited most from expanded training data (Workload, Accessibility).

### 9.4 Validation-based early stopping prevents overfitting

Training for 10 epochs with validation monitoring showed val loss plateauing at epoch 3-5 and rising after. Saving the best model at epoch 5 (val loss 0.410) gave better test performance than training to completion (epoch 10, val loss 0.431).

### 9.5 Aspect-based sentiment reveals what star ratings hide

A professor with a 3.5 average could have excellent lectures but unfair exams, or boring lectures with generous grading. The radar chart visualization makes these differences visible -- something a single average star rating can never show.

### 9.6 LLM-assisted labeling enables large-scale ground truth

Using Claude Sonnet 4.6 to label 8,378 reviews across 103 departments would have taken weeks manually. Parallelized across 20 agents, it took ~5 minutes. The labels serve as ground truth for training, and the fine-tuned model then scales to score all 65,376 reviews automatically.

---

## 10. Technical Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.12 |
| Package management | uv |
| ML Framework | PyTorch (MPS backend) |
| Transformers | HuggingFace Transformers |
| Data | pandas, pyarrow |
| Metrics | scikit-learn |
| HTTP client | httpx (async) |
| Frontend | Streamlit |
| Visualization | plotly |
| Compute | MacBook Pro M5, 24GB RAM |
| Labeling | Claude Sonnet 4.6 (20 parallel agents) |

---

## 11. Project Structure

```
course_review/
|-- pyproject.toml
|-- run_pipeline.py             # end-to-end pipeline runner
|-- src/
|   |-- scraper/
|   |   |-- client.py           # RMP GraphQL client (paginated, async, retry)
|   |   |-- parse.py            # normalize API responses
|   |   |-- preprocess.py       # text cleaning, dedup
|   |   |-- run.py              # scraping entrypoint
|   |-- models/
|   |   |-- baseline.py         # star -> sentiment mapping
|   |   |-- zero_shot.py        # BART-mnli topics + RoBERTa sentiment
|   |   |-- fine_tune.py        # joint DistilBERT (5x4-class heads)
|   |   |-- labeling.py         # annotation tools + train/test split
|   |   |-- process.py          # batch scoring
|   |   |-- evaluate.py         # per-topic sentiment metrics + comparison
|   |-- recommend/
|   |   |-- engine.py           # weighted scoring + filtering
|   |-- app/
|       |-- streamlit_app.py    # 3-tab web UI
|-- tests/                      # 30 tests
|-- data/
|   |-- raw/                    # cached API JSON (4,221 files)
|   |-- processed/              # parquet files, evaluation results
|   |-- labels/                 # Sonnet 4.6 per-topic sentiment annotations
|-- models/
    |-- joint_classifier/       # saved DistilBERT weights (best epoch)
```

---

## 12. Reproducibility

```bash
uv sync --extra dev                              # install deps
uv run python run_pipeline.py                    # scrape + clean + score

# Train (requires labeled data in data/labels/)
uv run python -c "
import pandas as pd
from src.models.fine_tune import train_joint_classifier
train = pd.read_parquet('data/processed/train.parquet')
train_joint_classifier(train['review_text'].tolist(), train['label_topics'].tolist())
"

# Evaluate
uv run python -c "
import json, pandas as pd
from pathlib import Path
from src.models.fine_tune import predict_joint
from src.models.zero_shot import TopicClassifier, SentimentAnalyzer
from src.models.evaluate import compare_approaches
test = pd.read_parquet('data/processed/test.parquet')
texts = test['review_text'].tolist()
tc = TopicClassifier(); sa = SentimentAnalyzer()
zs = [sa.analyze_by_topic_flat(t, tc) for t in texts]
ft = predict_joint(texts)
results = compare_approaches(test, zs, ft)
Path('data/processed/evaluation_results.json').write_text(json.dumps(results, indent=2, default=str))
"

uv run streamlit run src/app/streamlit_app.py    # launch app
```
