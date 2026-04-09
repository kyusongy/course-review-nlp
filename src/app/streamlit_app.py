import json
import os
import sys
from pathlib import Path

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.fine_tune import predict_joint
from src.models.zero_shot import TOPICS, SentimentAnalyzer, TopicClassifier
from src.recommend.engine import TOPIC_KEYS, filter_results, score_professors

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="UNC Course Compass",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

# ── Theme ────────────────────────────────────────────────────────────────────

CAROLINA_BLUE = "#4B9CD3"
NAVY = "#13294B"
DARK_BG = "#0E1117"
CARD_BG = "#1A1D23"
ACCENT_GREEN = "#2ECC71"
ACCENT_RED = "#E74C3C"
ACCENT_AMBER = "#F39C12"
TEXT_PRIMARY = "#FAFAFA"
TEXT_MUTED = "#8B949E"
BORDER = "#30363D"

SENTIMENT_COLORS = {
    "positive": ACCENT_GREEN,
    "neutral": ACCENT_AMBER,
    "negative": ACCENT_RED,
}

TOPIC_ICONS = {
    "Workload": "",
    "Grading": "",
    "Teaching Quality": "",
    "Accessibility": "",
    "Exam Difficulty": "",
}

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
.stApp {{
    font-family: 'DM Sans', sans-serif;
}}

/* Hide default streamlit header */
header[data-testid="stHeader"] {{
    background: transparent;
}}

/* Hero banner */
.hero {{
    background: linear-gradient(135deg, {NAVY} 0%, #1a3a5c 50%, {CAROLINA_BLUE}33 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid {CAROLINA_BLUE}22;
    position: relative;
    overflow: hidden;
}}
.hero::before {{
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, {CAROLINA_BLUE}15 0%, transparent 70%);
    border-radius: 50%;
}}
.hero h1 {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    margin: 0 0 0.3rem 0;
    position: relative;
}}
.hero p {{
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: {TEXT_MUTED};
    margin: 0;
    position: relative;
}}
.hero .accent {{
    color: {CAROLINA_BLUE};
    font-weight: 600;
}}

/* Cards */
.metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}}
.metric-card .value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: {TEXT_PRIMARY};
}}
.metric-card .label {{
    font-size: 0.8rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}}

/* Sentiment pills */
.pill {{
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}
.pill-positive {{ background: {ACCENT_GREEN}22; color: {ACCENT_GREEN}; border: 1px solid {ACCENT_GREEN}44; }}
.pill-neutral {{ background: {ACCENT_AMBER}22; color: {ACCENT_AMBER}; border: 1px solid {ACCENT_AMBER}44; }}
.pill-negative {{ background: {ACCENT_RED}22; color: {ACCENT_RED}; border: 1px solid {ACCENT_RED}44; }}

/* Review card */
.review-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}}
.review-card .meta {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}}
.review-card .stars {{
    color: {ACCENT_AMBER};
    font-size: 1rem;
    letter-spacing: 2px;
}}
.review-card .course {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {CAROLINA_BLUE};
    background: {CAROLINA_BLUE}15;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
}}
.review-card .text {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.95rem;
    line-height: 1.6;
    color: {TEXT_PRIMARY}dd;
}}

/* Professor rank card */
.rank-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.2s;
}}
.rank-card:hover {{
    border-color: {CAROLINA_BLUE}66;
}}
.rank-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: {CAROLINA_BLUE};
    min-width: 2rem;
    text-align: center;
}}
.rank-info {{
    flex: 1;
}}
.rank-name {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}
.rank-meta {{
    font-size: 0.8rem;
    color: {TEXT_MUTED};
    margin-top: 0.15rem;
}}
.rank-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 500;
    color: {ACCENT_GREEN};
}}

/* Sidebar styling */
section[data-testid="stSidebar"] {{
    background: {CARD_BG};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.85rem;
}}

/* Topic analysis card */
.topic-result {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid {BORDER};
}}
.topic-result:last-child {{ border-bottom: none; }}
.topic-icon {{ font-size: 1.2rem; }}
.topic-name {{
    font-weight: 500;
    color: {TEXT_PRIMARY};
    flex: 1;
}}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    letter-spacing: 0.02em;
}}

/* Section headers */
.section-header {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid {CAROLINA_BLUE}44;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ── Data Loading ─────────────────────────────────────────────────────────────

SCORE_SOURCES = {
    "Sonnet 4.6 Labels (ground truth)": "labeled_scores.parquet",
    "Fine-Tuned DistilBERT": "finetuned_scores.parquet",
    "Zero-Shot (BART + RoBERTa)": "zero_shot_scores.parquet",
}


@st.cache_data
def load_reviews() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "reviews.parquet")


@st.cache_data
def load_scores(source_file: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / source_file)


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


# ── Components ───────────────────────────────────────────────────────────────


def radar_chart(scores: dict[str, float], height: int = 320) -> go.Figure:
    topics = list(scores.keys())
    values = list(scores.values())
    topics_closed = topics + [topics[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=topics_closed,
            fill="toself",
            fillcolor=f"rgba(75, 156, 211, 0.15)",
            line=dict(color=CAROLINA_BLUE, width=2.5),
            marker=dict(size=6, color=CAROLINA_BLUE),
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[-1, 1],
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "", "0", "", "+1"],
                gridcolor="#30363D",
                linecolor="#30363D",
                tickfont=dict(size=10, color=TEXT_MUTED),
            ),
            angularaxis=dict(
                gridcolor="#30363D",
                linecolor="#30363D",
                tickfont=dict(size=11, color=TEXT_PRIMARY, family="DM Sans"),
            ),
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=20, b=20),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def sentiment_pill(sentiment: str) -> str:
    return f'<span class="pill pill-{sentiment}">{sentiment}</span>'


def metric_card(value: str, label: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
    </div>"""


def review_card(text: str, rating: int, course: str) -> str:
    stars = "★" * rating + "☆" * (5 - rating)
    return f"""
    <div class="review-card">
        <div class="meta">
            <span class="stars">{stars}</span>
            <span class="course">{course}</span>
        </div>
        <div class="text">{text}</div>
    </div>"""


def topic_result_html(topic: str, sentiment: str) -> str:
    icon = TOPIC_ICONS.get(topic, "")
    return f"""
    <div class="topic-result">
        <span class="topic-icon">{icon}</span>
        <span class="topic-name">{topic}</span>
        {sentiment_pill(sentiment)}
    </div>"""


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"""<div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-family: 'Source Serif 4', serif; font-size: 1.3rem; font-weight: 700; color: {TEXT_PRIMARY};">
            Course Compass
        </div>
        <div style="font-size: 0.78rem; color: {TEXT_MUTED}; margin-top: 0.2rem;">
            UNC Statistics & Operations Research
        </div>
    </div>""",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        f'<div style="font-size: 0.75rem; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Data Source</div>',
        unsafe_allow_html=True,
    )
    score_source = st.radio(
        "source",
        list(SCORE_SOURCES.keys()),
        index=0,
        label_visibility="collapsed",
    )
    score_file = SCORE_SOURCES[score_source]

    st.divider()
    st.markdown(
        f"""<div style="font-size: 0.75rem; color: {TEXT_MUTED}; line-height: 1.5; padding: 0.5rem 0;">
        <strong style="color: {TEXT_PRIMARY};">2,429</strong> reviews<br>
        <strong style="color: {TEXT_PRIMARY};">117</strong> professors<br>
        <strong style="color: {TEXT_PRIMARY};">5</strong> sentiment topics
    </div>""",
        unsafe_allow_html=True,
    )


# ── Hero ─────────────────────────────────────────────────────────────────────

st.markdown(
    """<div class="hero">
    <h1>UNC Course Compass</h1>
    <p>Aspect-based sentiment analysis for <span class="accent">Statistics & Operations Research</span> course reviews</p>
</div>""",
    unsafe_allow_html=True,
)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_explore, tab_recommend, tab_model = st.tabs(
    ["Explore", "Recommend", "Model Comparison"]
)

# ── Tab 1: Explore ───────────────────────────────────────────────────────────

with tab_explore:
    reviews_df = load_reviews()
    scores_df = load_scores(score_file)

    professors = sorted(reviews_df["professor_name"].dropna().unique())
    selected_prof = st.selectbox(
        "Search professor",
        professors,
        index=0,
        placeholder="Type a name...",
    )

    prof_reviews = reviews_df[reviews_df["professor_name"] == selected_prof]
    prof_scores = scores_df[scores_df["professor_name"] == selected_prof]

    col_left, col_right = st.columns([2, 3], gap="large")

    with col_left:
        # Metrics row
        n_reviews = len(prof_reviews)
        avg_rating = prof_reviews["star_rating"].mean() if n_reviews else 0
        avg_diff = prof_reviews["difficulty_rating"].mean() if n_reviews else 0
        wta = prof_reviews["would_take_again"].dropna()
        wta_pct = f"{wta.mean() * 100:.0f}%" if len(wta) > 0 else "N/A"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(metric_card(str(n_reviews), "Reviews"), unsafe_allow_html=True)
        with c2:
            st.markdown(
                metric_card(f"{avg_rating:.1f}", "Avg Rating"), unsafe_allow_html=True
            )
        with c3:
            st.markdown(metric_card(wta_pct, "Would Retake"), unsafe_allow_html=True)

        st.markdown("", unsafe_allow_html=True)

        # Radar chart
        if not prof_scores.empty:
            topic_means = {}
            for topic, key in zip(TOPICS, TOPIC_KEYS):
                col_name = f"topic_{key}_score"
                vals = prof_scores[col_name].dropna()
                topic_means[topic] = vals.mean() if len(vals) > 0 else 0
            st.plotly_chart(
                radar_chart(topic_means), width="stretch", key="explore_radar"
            )
        else:
            st.info("No scored reviews for this professor.")

        # Per-topic sentiment breakdown
        if not prof_scores.empty:
            st.markdown(
                f'<div class="section-header">Topic Breakdown</div>',
                unsafe_allow_html=True,
            )
            for topic, key in zip(TOPICS, TOPIC_KEYS):
                col_name = f"topic_{key}_sentiment"
                vals = prof_scores[col_name].dropna()
                if len(vals) == 0:
                    continue
                pos = (vals == "positive").sum()
                neg = (vals == "negative").sum()
                neu = (vals == "neutral").sum()
                total = len(vals)
                dominant = "positive" if pos >= neg else "negative"
                if neu > pos and neu > neg:
                    dominant = "neutral"
                icon = TOPIC_ICONS.get(topic, "")
                bar_pct = pos / total * 100 if total > 0 else 0
                st.markdown(
                    f"""<div style="display:flex; align-items:center; gap:0.6rem; padding:0.4rem 0; border-bottom:1px solid {BORDER};">
                    <span style="font-size:1.1rem;">{icon}</span>
                    <span style="flex:1; font-weight:500; color:{TEXT_PRIMARY};">{topic}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:{TEXT_MUTED};">{total} mentions</span>
                    {sentiment_pill(dominant)}
                </div>""",
                    unsafe_allow_html=True,
                )

    with col_right:
        st.markdown(
            f'<div class="section-header">Reviews</div>', unsafe_allow_html=True
        )
        for _, row in prof_reviews.head(20).iterrows():
            rating = int(row.get("star_rating", 0))
            course = row.get("course_name", "")
            text = row.get("review_text", "")
            st.markdown(review_card(text, rating, course), unsafe_allow_html=True)

        if len(prof_reviews) > 20:
            st.caption(f"Showing 20 of {len(prof_reviews)} reviews.")


# ── Tab 2: Recommend ─────────────────────────────────────────────────────────

with tab_recommend:
    st.markdown(
        f'<div class="section-header">Set Your Priorities</div>',
        unsafe_allow_html=True,
    )

    slider_cols = st.columns(5)
    weights = {}
    for i, (topic, key) in enumerate(zip(TOPICS, TOPIC_KEYS)):
        with slider_cols[i]:
            icon = TOPIC_ICONS.get(topic, "")
            weights[key] = st.slider(f"{icon} {topic}", 0, 10, 5, key=f"w_{key}")

    min_reviews = st.slider("Minimum reviews", 1, 30, 5, key="min_rev")

    scores_df = load_scores(score_file)
    agg_df = aggregate_prof_scores(scores_df)
    ranked = score_professors(agg_df, {k: float(v) for k, v in weights.items()})
    filtered = filter_results(ranked, min_reviews=min_reviews)
    top = filtered.head(10)

    st.markdown(
        f'<div class="section-header">Top Recommendations</div>',
        unsafe_allow_html=True,
    )

    if top.empty:
        st.warning("No professors match the current filters.")
    else:
        for rank_idx, (_, row) in enumerate(top.iterrows()):
            score_val = row["score"]
            score_color = (
                ACCENT_GREEN
                if score_val > 0.2
                else ACCENT_AMBER
                if score_val > -0.2
                else ACCENT_RED
            )

            with st.container():
                col_info, col_chart = st.columns([3, 2], gap="medium")

                with col_info:
                    st.markdown(
                        f"""<div class="rank-card">
                        <div class="rank-num">#{rank_idx + 1}</div>
                        <div class="rank-info">
                            <div class="rank-name">{row["professor_name"]}</div>
                            <div class="rank-meta">{int(row["num_reviews"])} reviews</div>
                        </div>
                        <div class="rank-score" style="color: {score_color};">{score_val:+.2f}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

                with col_chart:
                    topic_scores = {
                        topic: row[key] for topic, key in zip(TOPICS, TOPIC_KEYS)
                    }
                    st.plotly_chart(
                        radar_chart(topic_scores, height=220),
                        width="stretch",
                        key=f"rec_{rank_idx}",
                    )


# ── Tab 3: Model Comparison ──────────────────────────────────────────────────

with tab_model:
    eval_data = load_eval_results()

    if eval_data is None:
        st.warning("Run evaluation first.")
    else:
        # Overall sentiment comparison — the headline metric
        st.markdown(
            f'<div class="section-header">Overall Sentiment Accuracy</div>',
            unsafe_allow_html=True,
        )
        sent_data = eval_data.get("overall_sentiment", {})
        if sent_data:
            cols = st.columns(3)
            labels = {
                "baseline": "Star Baseline",
                "zero_shot": "Zero-Shot",
                "finetuned": "Fine-Tuned",
            }
            for i, (model_key, label) in enumerate(labels.items()):
                data = sent_data.get(model_key, {})
                acc = data.get("accuracy", 0)
                f1 = data.get("f1_macro", 0)
                with cols[i]:
                    st.markdown(
                        metric_card(f"{acc:.1%}", label),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="text-align:center; font-size:0.78rem; color:{TEXT_MUTED}; margin-top:0.3rem;">F1 Macro: <span style="color:{TEXT_PRIMARY};">{f1:.3f}</span></div>',
                        unsafe_allow_html=True,
                    )

        # Per-topic sentiment
        st.markdown(
            f'<div class="section-header">Per-Topic Sentiment</div>',
            unsafe_allow_html=True,
        )
        pts_data = eval_data.get("per_topic_sentiment", {})
        if pts_data:
            rows = []
            for model_name, topics in pts_data.items():
                for topic, results in topics.items():
                    if topic.startswith("_"):
                        continue
                    acc = results.get("accuracy", 0)
                    f1 = results.get("f1_macro", 0)
                    rows.append(
                        {
                            "Model": "Zero-Shot"
                            if model_name == "zero_shot"
                            else "Fine-Tuned",
                            "Topic": topic,
                            "N": results.get("n", ""),
                            "Accuracy": f"{acc:.1%}" if acc else "0%",
                            "F1 Macro": f"{f1:.3f}" if f1 else "0.000",
                        }
                    )
            if rows:
                st.dataframe(
                    pd.DataFrame(rows).set_index(["Model", "Topic"]),
                    width="stretch",
                )

        # Topic detection
        st.markdown(
            f'<div class="section-header">Topic Detection F1</div>',
            unsafe_allow_html=True,
        )
        if pts_data:
            det_cols = st.columns(2)
            for i, model_name in enumerate(["zero_shot", "finetuned"]):
                det = pts_data.get(model_name, {}).get("_topic_detection", {})
                label = "Zero-Shot" if model_name == "zero_shot" else "Fine-Tuned"
                with det_cols[i]:
                    macro = det.get("f1_macro", 0)
                    micro = det.get("f1_micro", 0)
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="value">{macro:.3f}</div>
                        <div class="label">{label} &mdash; Macro</div>
                        <div style="font-size:0.8rem; color:{TEXT_MUTED}; margin-top:0.5rem;">Micro: <span style="color:{TEXT_PRIMARY};">{micro:.3f}</span></div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

    # Try a Review
    st.markdown(
        f'<div class="section-header">Live Analysis</div>', unsafe_allow_html=True
    )
    review_text = st.text_area(
        "Paste a review to compare models side-by-side:",
        value="Great professor, explains concepts clearly. But the exams were brutal and nothing like the homework.",
        height=100,
        label_visibility="visible",
    )

    if st.button("Analyze", type="primary", width="stretch"):
        col_zero, col_fine = st.columns(2, gap="large")

        with col_zero:
            st.markdown(
                f'<div style="font-family:Source Serif 4,serif; font-size:1.1rem; font-weight:600; color:{CAROLINA_BLUE}; margin-bottom:0.8rem;">Zero-Shot Pipeline</div>',
                unsafe_allow_html=True,
            )
            with st.spinner("Running BART + RoBERTa..."):
                try:
                    tc, sa = load_zero_shot_models()
                    topic_sents = sa.analyze_by_topic_flat(review_text, tc)
                    if topic_sents:
                        html = ""
                        for topic, sent in topic_sents.items():
                            html += topic_result_html(topic, sent)
                        st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.caption("No topics detected.")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_fine:
            st.markdown(
                f'<div style="font-family:Source Serif 4,serif; font-size:1.1rem; font-weight:600; color:{ACCENT_GREEN}; margin-bottom:0.8rem;">Fine-Tuned DistilBERT</div>',
                unsafe_allow_html=True,
            )
            joint_dir = MODELS_DIR / "joint_classifier"
            if not joint_dir.exists():
                st.warning("Fine-tuned model not found.")
            else:
                with st.spinner("Running joint classifier..."):
                    try:
                        ft_results = predict_joint([review_text])[0]
                        if ft_results:
                            html = ""
                            for topic, sent in ft_results.items():
                                html += topic_result_html(topic, sent)
                            st.markdown(html, unsafe_allow_html=True)
                        else:
                            st.caption("No topics detected.")
                    except Exception as e:
                        st.error(f"Error: {e}")
