"""Fine-tune DistilBERT for topic and sentiment classification."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.models.zero_shot import TOPICS

MODEL_NAME = "distilbert-base-uncased"
MODELS_DIR = Path("models")
SENTIMENT_LABELS = ["positive", "neutral", "negative"]

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def create_topic_labels(topics: list[str]) -> list[float]:
    """Convert topic list to multi-hot vector aligned with TOPICS order."""
    return [1.0 if t in topics else 0.0 for t in TOPICS]


class TopicDataset(Dataset):
    def __init__(
        self, texts: list[str], topic_lists: list[list[str]], max_length: int = 256
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._encodings = self._tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self._labels = [create_topic_labels(t) for t in topic_lists]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self._encodings.items()}
        item["labels"] = torch.tensor(self._labels[idx], dtype=torch.float)
        return item


class SentimentDataset(Dataset):
    def __init__(self, texts: list[str], sentiments: list[str], max_length: int = 256):
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._encodings = self._tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self._labels = [SENTIMENT_LABELS.index(s) for s in sentiments]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self._encodings.items()}
        item["labels"] = torch.tensor(self._labels[idx], dtype=torch.long)
        return item


def _train_loop(model, loader, optimizer, scheduler, epochs: int) -> None:
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(_DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")


def train_topic_classifier(
    train_texts: list[str],
    train_topics: list[list[str]],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> None:
    """Fine-tune DistilBERT for multi-label topic classification."""
    save_dir = MODELS_DIR / "topic_classifier"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = TopicDataset(train_texts, train_topics)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(TOPICS),
        problem_type="multi_label_classification",
    ).to(_DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    _train_loop(model, loader, optimizer, scheduler, epochs)

    model.save_pretrained(save_dir)
    dataset._tokenizer.save_pretrained(save_dir)
    print(f"Saved topic classifier to {save_dir}")


def predict_topics(
    texts: list[str],
    model_dir: str | Path,
    threshold: float = 0.5,
) -> list[list[str]]:
    """Load saved topic model and predict topic lists for each text."""
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_DEVICE)
    model.eval()

    results = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            ).to(_DEVICE)
            logits = model(**enc).logits
            probs = torch.sigmoid(logits).squeeze().cpu().tolist()
            topics = [t for t, p in zip(TOPICS, probs) if p >= threshold]
            results.append(topics)
    return results


def train_sentiment_classifier(
    train_texts: list[str],
    train_sentiments: list[str],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> None:
    """Fine-tune DistilBERT for 3-class sentiment classification."""
    save_dir = MODELS_DIR / "sentiment_classifier"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = SentimentDataset(train_texts, train_sentiments)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(SENTIMENT_LABELS),
    ).to(_DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    _train_loop(model, loader, optimizer, scheduler, epochs)

    model.save_pretrained(save_dir)
    dataset._tokenizer.save_pretrained(save_dir)
    print(f"Saved sentiment classifier to {save_dir}")


def predict_sentiment(texts: list[str], model_dir: str | Path) -> list[dict]:
    """Load saved sentiment model and return list of {label, score} dicts."""
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_DEVICE)
    model.eval()

    results = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            ).to(_DEVICE)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
            best_idx = int(torch.argmax(torch.tensor(probs)).item())
            results.append(
                {"label": SENTIMENT_LABELS[best_idx], "score": probs[best_idx]}
            )
    return results
