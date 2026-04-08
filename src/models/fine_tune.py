"""Fine-tune DistilBERT for joint topic-sentiment classification.

Each of 6 topics has 4 possible states: not_discussed, positive, neutral, negative.
The model has 6 independent 4-class classification heads sharing a DistilBERT backbone.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.models.zero_shot import TOPICS

MODEL_NAME = "distilbert-base-uncased"
MODELS_DIR = Path("models")

# Per-topic states: 0=not_discussed, 1=positive, 2=neutral, 3=negative
SENTIMENT_TO_IDX = {"positive": 1, "neutral": 2, "negative": 3}
IDX_TO_SENTIMENT = {0: "not_discussed", 1: "positive", 2: "neutral", 3: "negative"}
NUM_STATES = 4

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def encode_topic_sentiments(topic_sentiments: dict[str, str]) -> list[int]:
    """Convert {topic: sentiment} dict to list of 6 ints (one per topic).

    Topics not in the dict get state 0 (not_discussed).
    """
    return [SENTIMENT_TO_IDX.get(topic_sentiments.get(t, ""), 0) for t in TOPICS]


def decode_topic_sentiments(states: list[int]) -> dict[str, str]:
    """Convert list of 6 state ints back to {topic: sentiment} dict.

    Skips topics with state 0 (not_discussed).
    """
    result = {}
    for topic, state in zip(TOPICS, states):
        if state != 0:
            result[topic] = IDX_TO_SENTIMENT[state]
    return result


# --- Dataset ---


class TopicSentimentDataset(Dataset):
    """Dataset for joint topic-sentiment classification."""

    def __init__(
        self,
        texts: list[str],
        topic_sentiments: list[dict[str, str]],
        max_length: int = 256,
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._encodings = self._tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        # Each label is a list of 6 ints (one 4-class label per topic)
        self._labels = [encode_topic_sentiments(ts) for ts in topic_sentiments]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self._encodings.items()}
        item["labels"] = torch.tensor(self._labels[idx], dtype=torch.long)
        return item


# --- Model ---


class TopicSentimentModel(nn.Module):
    """DistilBERT with 6 independent 4-class heads (one per topic)."""

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for distilbert-base
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        # 6 independent heads, each outputting 4 logits
        self.heads = nn.ModuleList([nn.Linear(hidden_size, NUM_STATES) for _ in TOPICS])

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        hidden = outputs.last_hidden_state[:, 0]  # (batch, hidden_size)
        hidden = self.pre_classifier(hidden)
        hidden = nn.ReLU()(hidden)
        hidden = self.dropout(hidden)

        # Each head produces (batch, 4) logits
        all_logits = [head(hidden) for head in self.heads]  # list of 6 × (batch, 4)

        loss = None
        if labels is not None:
            # labels shape: (batch, 6) — one 4-class label per topic
            loss = torch.tensor(0.0, device=input_ids.device)
            ce = nn.CrossEntropyLoss()
            for i, logits in enumerate(all_logits):
                loss += ce(logits, labels[:, i])
            loss /= len(TOPICS)  # average across topics

        # Stack logits: (batch, 6, 4)
        stacked_logits = torch.stack(all_logits, dim=1)
        return {"loss": loss, "logits": stacked_logits}


# --- Training ---


def train_joint_classifier(
    train_texts: list[str],
    train_topic_sentiments: list[dict[str, str]],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> None:
    """Fine-tune DistilBERT for joint topic-sentiment classification."""
    save_dir = MODELS_DIR / "joint_classifier"
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = TopicSentimentDataset(train_texts, train_topic_sentiments)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TopicSentimentModel().to(_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(_DEVICE) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")

    # Save model + tokenizer
    torch.save(model.state_dict(), save_dir / "model.pt")
    dataset._tokenizer.save_pretrained(save_dir)
    print(f"Saved joint classifier to {save_dir}")


def predict_joint(
    texts: list[str],
    model_dir: str | Path = MODELS_DIR / "joint_classifier",
) -> list[dict[str, str]]:
    """Predict per-topic sentiment for each text.

    Returns list of {topic: sentiment} dicts.
    """
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = TopicSentimentModel()
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location=_DEVICE, weights_only=True)
    )
    model.to(_DEVICE)
    model.eval()

    results = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256
            ).to(_DEVICE)
            outputs = model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            )
            logits = outputs["logits"].squeeze(0)  # (6, 4)
            states = logits.argmax(dim=-1).cpu().tolist()  # list of 6 ints
            results.append(decode_topic_sentiments(states))
    return results


# --- Legacy wrappers for backward compat ---


def create_topic_labels(topics) -> list[float]:
    """Convert topic list or dict to multi-hot vector (for evaluation compat)."""
    if isinstance(topics, dict):
        return [1.0 if t in topics else 0.0 for t in TOPICS]
    return [1.0 if t in topics else 0.0 for t in TOPICS]


def predict_topics(
    texts: list[str],
    model_dir: str | Path = MODELS_DIR / "joint_classifier",
    threshold: float = 0.5,
) -> list[list[str]]:
    """Predict topics using joint model (returns topic lists for backward compat)."""
    preds = predict_joint(texts, model_dir)
    return [list(p.keys()) for p in preds]


def predict_sentiment(
    texts: list[str], model_dir: str | Path = MODELS_DIR / "joint_classifier"
) -> list[dict]:
    """Predict overall sentiment using joint model (backward compat).

    Uses majority topic sentiment as overall sentiment.
    """
    preds = predict_joint(texts, model_dir)
    results = []
    for p in preds:
        if not p:
            results.append({"label": "neutral", "score": 0.0})
            continue
        sentiments = list(p.values())
        # Count votes
        from collections import Counter

        counts = Counter(sentiments)
        label = counts.most_common(1)[0][0]
        score = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}[label]
        results.append({"label": label, "score": score})
    return results
