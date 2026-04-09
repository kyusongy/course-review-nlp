# UNC Course Compass

ML-powered course recommendation system for UNC Statistics & Data Science. Scrapes RateMyProfessor reviews, performs **aspect-based sentiment analysis** (per-topic positive/neutral/negative), and recommends professors based on student preferences.

## Quick Start

```bash
uv sync --extra dev          # install dependencies
uv run python run_pipeline.py      # scrape + clean + score reviews
uv run streamlit run src/app/streamlit_app.py  # launch app
```

## Pipeline

1. **Scrape** — RateMyProfessor GraphQL API → 2,429 reviews from 117 professors
2. **Label** — Claude Sonnet 4.6 labels per-topic sentiment for all reviews
3. **Train** — Fine-tune DistilBERT (5 independent 4-class heads) on labeled data
4. **Evaluate** — Compare star baseline vs zero-shot (BART+RoBERTa) vs fine-tuned
5. **Serve** — Streamlit app with explore, recommend, and model comparison tabs

## Topics

Each review is labeled with sentiment (positive/neutral/negative) per topic:

| Topic | Description |
|-------|-------------|
| Workload | Homework volume, time commitment |
| Grading | Fairness, curves, grade distribution |
| Teaching Quality | Lecture clarity, engagement, responsiveness |
| Accessibility | Office hours, approachability |
| Exam Difficulty | Test format, fairness, prep alignment |

## Models

| Model | Params | Role |
|-------|--------|------|
| BART-large-mnli | 406M | Zero-shot topic detection via entailment |
| RoBERTa-sentiment | 125M | Zero-shot sentiment per sentence |
| DistilBERT (ours) | 66M | Fine-tuned joint topic-sentiment classifier |

## Results (729 test reviews)

**Topic Detection F1:** Zero-shot 0.600 macro / Fine-tuned **0.769** macro

**Per-Topic Sentiment Accuracy:** Fine-tuned best on Teaching Quality (82.7%), zero-shot best on low-data topics (Workload, Accessibility)

**Overall Sentiment:** Star baseline 79.3% / Zero-shot 71.6% / Fine-tuned **84.0%**

## Project Structure

```
src/
  scraper/    # RMP GraphQL client, parser, preprocessor
  models/     # baseline, zero_shot, fine_tune, evaluate, labeling
  recommend/  # weighted scoring engine
  app/        # Streamlit frontend
tests/        # 30 tests
data/labels/  # Sonnet 4.6 per-topic sentiment annotations
```

## Tech Stack

Python 3.12, PyTorch (MPS), HuggingFace Transformers, httpx, pandas, Streamlit, plotly, uv

## Course

Data 522 — Practical Deep Learning, UNC Chapel Hill
