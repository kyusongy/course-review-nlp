import json
import os
import sys
from pathlib import Path

# Suppress transformers file watcher spam
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Ensure project root is on sys.path so `src` imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.fine_tune import predict_joint
from src.models.zero_shot import TOPICS, SentimentAnalyzer, TopicClassifier
from src.recommend.engine import TOPIC_KEYS, filter_results, score_professors

st.set_page_config(page_title="UNC Course Compass", layout="wide")

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

SAMPLE_REVIEW = (
    "Professor Smith was incredibly helpful during office hours. "
    "The workload was heavy but fair, and exams were challenging but reflected what was taught."
)


@st.cache_data
def load_reviews() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "reviews.parquet")


@st.cache_data
def load_scores() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "finetuned_scores.parquet")


@st.cache_data
def load_eval_results() -> dict | None:
    path = DATA_DIR / "evaluation_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def aggregate_prof_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    return (
        scores_df.groupby("professor_name")
        .agg(
            num_reviews=("professor_name", "count"),
            **{k: (f"topic_{k}_score", "mean") for k in TOPIC_KEYS},
        )
        .fillna(0.0)
        .reset_index()
    )


@st.cache_resource
def load_zero_shot_models():
    tc = TopicClassifier()
    sa = SentimentAnalyzer()
    return tc, sa


def radar_chart(scores: dict[str, float], title: str = "") -> go.Figure:
    topics = list(scores.keys())
    values = list(scores.values())
    # close the loop
    topics_closed = topics + [topics[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values_closed,
            theta=topics_closed,
            fill="toself",
            line_color="royalblue",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
        showlegend=False,
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
    )
    return fig


def star_display(rating: int) -> str:
    r = int(rating)
    return "★" * r + "☆" * (5 - r)


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_explore, tab_recommend, tab_model = st.tabs(
    ["Explore", "Recommend", "Model Comparison"]
)

# ── Tab 1: Explore ────────────────────────────────────────────────────────────
with tab_explore:
    st.header("Explore Professors")

    reviews_df = load_reviews()
    scores_df = load_scores()

    professors = sorted(reviews_df["professor_name"].dropna().unique())
    selected_prof = st.selectbox("Select a professor", professors)

    prof_reviews = reviews_df[reviews_df["professor_name"] == selected_prof]
    prof_scores = scores_df[scores_df["professor_name"] == selected_prof]

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Summary")
        st.metric("Reviews", len(prof_reviews))
        if "star_rating" in prof_reviews.columns:
            avg_rating = prof_reviews["star_rating"].mean()
            st.metric("Avg Rating", f"{avg_rating:.2f} / 5")
        if "difficulty_rating" in prof_reviews.columns:
            avg_diff = prof_reviews["difficulty_rating"].mean()
            st.metric("Avg Difficulty", f"{avg_diff:.2f}")

        # Radar chart from zero_shot_scores
        if not prof_scores.empty:
            topic_means = {
                topic: prof_scores[f"topic_{key}_score"].mean()
                for topic, key in zip(TOPICS, TOPIC_KEYS)
            }
            st.plotly_chart(
                radar_chart(topic_means, title="Topic Sentiment"),
                use_container_width=True,
            )
        else:
            st.info("No scored reviews available for this professor.")

    with col_right:
        st.subheader("Reviews")
        for _, row in prof_reviews.iterrows():
            rating = int(row.get("star_rating", 0))
            course = row.get("course_name", "Unknown Course")
            text = row.get("review_text", "")
            st.markdown(f"**{star_display(rating)}** — {course}")
            st.write(text)
            st.divider()

# ── Tab 2: Recommend ──────────────────────────────────────────────────────────
with tab_recommend:
    st.header("Find Your Professor")

    col_sliders, col_results = st.columns([1, 2])

    with col_sliders:
        st.subheader("Priorities")
        weights = {
            key: st.slider(topic, 0, 10, 5) for topic, key in zip(TOPICS, TOPIC_KEYS)
        }
        min_reviews = st.slider("Minimum reviews", 1, 20, 3)

    with col_results:
        st.subheader("Top Professors")
        scores_df = load_scores()
        agg_df = aggregate_prof_scores(scores_df)

        ranked = score_professors(agg_df, {k: float(v) for k, v in weights.items()})
        filtered = filter_results(ranked, min_reviews=min_reviews)
        top10 = filtered.head(10)

        if top10.empty:
            st.warning("No professors match the current filters.")
        else:
            for _, row in top10.iterrows():
                with st.expander(
                    f"{row['professor_name']}  —  score: {row['score']:.3f}  "
                    f"({int(row['num_reviews'])} reviews)"
                ):
                    topic_scores = {
                        topic: row[key] for topic, key in zip(TOPICS, TOPIC_KEYS)
                    }
                    st.plotly_chart(
                        radar_chart(topic_scores),
                        use_container_width=True,
                        key=f"rec_{row['professor_name']}",
                    )

# ── Tab 3: Model Comparison ───────────────────────────────────────────────────
with tab_model:
    st.header("Model Comparison")

    eval_data = load_eval_results()

    if eval_data is None:
        st.warning(
            "Run evaluation first — `data/processed/evaluation_results.json` not found."
        )
    else:
        # Per-topic sentiment accuracy/F1
        st.subheader("Per-Topic Sentiment")
        pts_data = eval_data.get("per_topic_sentiment", {})
        if pts_data:
            rows = []
            for model_name, topics in pts_data.items():
                for topic, results in topics.items():
                    if topic.startswith("_"):
                        continue
                    rows.append(
                        {
                            "Model": model_name,
                            "Topic": topic,
                            "N": results.get("n", ""),
                            "Accuracy": round(results.get("accuracy", 0), 4),
                            "F1 Macro": round(results.get("f1_macro", 0), 4),
                        }
                    )
            if rows:
                st.dataframe(
                    pd.DataFrame(rows).set_index(["Model", "Topic"]),
                    use_container_width=True,
                )

        # Topic detection summary (from _topic_detection key)
        st.subheader("Topic Detection")
        if pts_data:
            rows = []
            for model_name, topics in pts_data.items():
                det = topics.get("_topic_detection", {})
                if det:
                    rows.append(
                        {
                            "Model": model_name,
                            "F1 Macro": round(det.get("f1_macro", 0), 4),
                            "F1 Micro": round(det.get("f1_micro", 0), 4),
                        }
                    )
            if rows:
                st.dataframe(
                    pd.DataFrame(rows).set_index("Model"), use_container_width=True
                )

        # Overall sentiment comparison
        st.subheader("Overall Sentiment")
        sent_data = eval_data.get("overall_sentiment", {})
        if sent_data:
            rows = []
            for model_name, results in sent_data.items():
                rows.append(
                    {
                        "Model": model_name,
                        "Accuracy": round(results.get("accuracy", 0), 4),
                        "F1 Macro": round(results.get("f1_macro", 0), 4),
                    }
                )
            st.dataframe(
                pd.DataFrame(rows).set_index("Model"), use_container_width=True
            )

    # Try a Review
    st.subheader("Try a Review")
    review_text = st.text_area("Review text", value=SAMPLE_REVIEW, height=120)

    if st.button("Analyze"):
        col_star, col_zero, col_fine = st.columns(3)

        with col_star:
            st.markdown("### Star Proxy")
            st.info("N/A — no star rating provided")

        with col_zero:
            st.markdown("### Zero-Shot")
            with st.spinner("Running zero-shot models..."):
                try:
                    tc, sa = load_zero_shot_models()
                    topic_sents = sa.analyze_by_topic_flat(review_text, tc)
                    if topic_sents:
                        for topic, sent in topic_sents.items():
                            st.write(f"  {topic}: {sent}")
                    else:
                        st.write("No topics detected")
                except Exception as e:
                    st.error(f"Zero-shot error: {e}")

        with col_fine:
            st.markdown("### Fine-Tuned")
            joint_dir = MODELS_DIR / "joint_classifier"
            if not joint_dir.exists():
                st.warning("Fine-tuned model not found. Train it first.")
            else:
                with st.spinner("Running fine-tuned model..."):
                    try:
                        ft_results = predict_joint([review_text])[0]
                        if ft_results:
                            for topic, sent in ft_results.items():
                                st.write(f"  {topic}: {sent}")
                        else:
                            st.write("No topics detected")
                    except Exception as e:
                        st.error(f"Fine-tuned error: {e}")
