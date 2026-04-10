# UNC Course Compass

ML-powered course recommendation system for UNC Chapel Hill. Scrapes 65,376 RateMyProfessor reviews across 103 departments, performs **aspect-based sentiment analysis** (per-topic positive/neutral/negative), and recommends professors based on student preferences.

## Quick Start

```bash
uv sync --extra dev
uv run streamlit run src/app/streamlit_app.py
```

## Pipeline

1. **Scrape** -- RateMyProfessor GraphQL API -> 65,376 reviews, 4,173 professors, 103 departments
2. **Label** -- 8,378 reviews labeled with per-topic sentiment by Claude Sonnet 4.6 (20 parallel agents)
3. **Train** -- Fine-tune DistilBERT joint model (5 independent 4-class heads) with train/val/test split
4. **Evaluate** -- Compare fine-tuned vs zero-shot (BART+RoBERTa) vs star baseline on held-out test set
5. **Score** -- Apply fine-tuned model to all 65,376 reviews
6. **Serve** -- Streamlit app with explore, recommend, and model comparison tabs

## Topics

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

## Results (1,257 test reviews, 95 departments)

**Topic Detection F1:** Zero-shot 0.590 / Fine-tuned **0.832** macro

**Per-Topic Sentiment:** Fine-tuned beats zero-shot on all 5 topics (Teaching Quality 85.7%, Grading 70.7%, Exam Difficulty 69.3%, Workload 55.9%, Accessibility 72.1%)

**Overall Sentiment:** Star baseline 74.5% / Zero-shot 71.4% / Fine-tuned **81.1%**

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

Data 522 -- Practical Deep Learning, UNC Chapel Hill
