import json
import os
import re
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
from src.recommend.engine import (
    TOPIC_KEYS,
    aggregate_professor_scores,
    filter_results,
    score_professors,
)

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
CARD_BG_HOVER = "#20242B"
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

# Per-topic accent colors — playful but cohesive with Carolina Blue primary.
TOPIC_COLORS = {
    "Workload": "#F59E0B",  # amber
    "Grading": "#A78BFA",  # violet
    "Teaching Quality": CAROLINA_BLUE,
    "Accessibility": "#10B981",  # mint
    "Exam Difficulty": "#F87171",  # coral
}

# Lucide-style inline SVG icons. `currentColor` means they inherit CSS color.
_SVG_OPEN = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
_SVG_CLOSE = "</svg>"

TOPIC_ICONS = {
    # Clock — workload/time commitment
    "Workload": f'{_SVG_OPEN}<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>{_SVG_CLOSE}',
    # Pen line — grading
    "Grading": f'{_SVG_OPEN}<path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>{_SVG_CLOSE}',
    # Graduation cap — teaching quality
    "Teaching Quality": f'{_SVG_OPEN}<path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>{_SVG_CLOSE}',
    # Speech bubble — accessibility/office hours
    "Accessibility": f'{_SVG_OPEN}<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>{_SVG_CLOSE}',
    # Clipboard with check — exam difficulty
    "Exam Difficulty": f'{_SVG_OPEN}<rect x="8" y="2" width="8" height="4" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="m9 14 2 2 4-4"/>{_SVG_CLOSE}',
}

# Compass icon for the hero title
COMPASS_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>'

# Model-family icons (star = baseline, zap = zero-shot, sparkles = fine-tuned)
STAR_ICON = f'{_SVG_OPEN}<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>{_SVG_CLOSE}'
ZAP_ICON = (
    f'{_SVG_OPEN}<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>{_SVG_CLOSE}'
)
SPARK_ICON = f'{_SVG_OPEN}<path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5z"/><path d="M19 14l.75 2.25L22 17l-2.25.75L19 20l-.75-2.25L16 17l2.25-.75z"/>{_SVG_CLOSE}'


def topic_icon(topic: str, color: str | None = None, size: int = 18) -> str:
    """Return SVG markup for a topic icon, tinted with the topic accent color."""
    svg = TOPIC_ICONS.get(topic, "")
    if not svg:
        return ""
    c = color or TOPIC_COLORS.get(topic, CAROLINA_BLUE)
    return (
        f'<span style="display:inline-flex;color:{c};width:{size}px;height:{size}px;'
        f'align-items:center;justify-content:center;">{svg}</span>'
    )


# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ─────────────────────────────────────── */
.stApp {{ font-family: 'DM Sans', sans-serif; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.5rem; }}

/* Smooth tab content fade-in */
.stTabs [data-baseweb="tab-panel"] {{ animation: fade-in 0.4s ease-out; }}
@keyframes fade-in {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ── Hero banner ─────────────────────────────────── */
.hero {{
    background: linear-gradient(135deg, {NAVY} 0%, #1a3a5c 45%, {CAROLINA_BLUE}40 100%);
    border-radius: 22px;
    padding: 2.5rem 2.4rem;
    margin-bottom: 1.75rem;
    border: 1px solid {CAROLINA_BLUE}33;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 40px -18px {CAROLINA_BLUE}55, 0 2px 6px -2px #0008;
}}
.hero::before {{
    content: '';
    position: absolute;
    top: -45%; right: -15%;
    width: 460px; height: 460px;
    background: radial-gradient(circle, {CAROLINA_BLUE}22 0%, transparent 70%);
    border-radius: 50%;
}}
.hero::after {{
    content: '';
    position: absolute;
    top: 0; left: -60%;
    width: 45%; height: 100%;
    background: linear-gradient(105deg, transparent 30%, {CAROLINA_BLUE}18 50%, transparent 70%);
    animation: hero-sweep 5.5s ease-in-out infinite;
}}
@keyframes hero-sweep {{
    0%, 70%, 100% {{ transform: translateX(0); opacity: 0; }}
    25% {{ opacity: 1; }}
    50% {{ transform: translateX(320%); opacity: 0; }}
}}
.hero-title-row {{
    display: flex; align-items: center; gap: 0.75rem;
    position: relative;
}}
.hero-compass {{
    color: {CAROLINA_BLUE};
    display: inline-flex;
    animation: compass-spin 18s linear infinite;
    filter: drop-shadow(0 0 12px {CAROLINA_BLUE}44);
}}
@keyframes compass-spin {{
    from {{ transform: rotate(0deg); }}
    to   {{ transform: rotate(360deg); }}
}}
.hero h1 {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    margin: 0;
    letter-spacing: -0.01em;
}}
.hero p.tagline {{
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: {TEXT_MUTED};
    margin: 0.4rem 0 1.2rem 0;
    position: relative;
}}
.hero .accent {{ color: {CAROLINA_BLUE}; font-weight: 600; }}

.hero-stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    position: relative;
    margin-top: 0.4rem;
}}
.hero-chip {{
    display: inline-flex;
    align-items: baseline;
    gap: 0.45rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid {CAROLINA_BLUE}44;
    border-radius: 999px;
    padding: 0.45rem 0.95rem;
    backdrop-filter: blur(6px);
    font-size: 0.88rem;
    color: {TEXT_MUTED};
    transition: border-color 0.2s, background 0.2s;
}}
.hero-chip:hover {{
    border-color: {CAROLINA_BLUE};
    background: rgba(75,156,211,0.08);
}}
.hero-chip b {{
    font-family: 'JetBrains Mono', monospace;
    color: {TEXT_PRIMARY};
    font-weight: 600;
    font-size: 1rem;
}}

/* Count-up uses CSS @property + counter-reset trick (Chromium/Safari; Firefox 128+). */
@property --cu-r {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
@property --cu-p {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
@property --cu-d {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
@property --cu-t {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}

/* ── Metric cards ────────────────────────────────── */
.metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 1.3rem 1.1rem;
    text-align: center;
    transition: transform 0.18s, border-color 0.18s, box-shadow 0.18s;
    box-shadow: 0 1px 3px #0003;
}}
.metric-card:hover {{
    transform: translateY(-2px);
    border-color: {CAROLINA_BLUE}66;
    box-shadow: 0 10px 24px -12px {CAROLINA_BLUE}55, 0 2px 4px #0004;
}}
.metric-card .icon {{
    display: inline-flex;
    margin-bottom: 0.55rem;
    padding: 0.55rem;
    border-radius: 14px;
    background: {CAROLINA_BLUE}14;
    color: {CAROLINA_BLUE};
}}
.metric-card .value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: {TEXT_PRIMARY};
    line-height: 1.1;
}}
.metric-card .label {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.35rem;
}}

/* ── Sentiment pills ─────────────────────────────── */
.pill {{
    display: inline-block;
    padding: 0.22rem 0.75rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.pill-positive {{ background: {ACCENT_GREEN}22; color: {ACCENT_GREEN}; border: 1px solid {ACCENT_GREEN}55; }}
.pill-neutral  {{ background: {ACCENT_AMBER}22; color: {ACCENT_AMBER}; border: 1px solid {ACCENT_AMBER}55; }}
.pill-negative {{ background: {ACCENT_RED}22;   color: {ACCENT_RED};   border: 1px solid {ACCENT_RED}55; }}

.pill-confidence {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    color: {ACCENT_AMBER};
    background: {ACCENT_AMBER}18;
    border: 1px dashed {ACCENT_AMBER}66;
    text-transform: uppercase;
}}

/* ── Review cards ────────────────────────────────── */
.review-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s;
}}
.review-card:hover {{
    transform: translateY(-1px);
    border-color: {CAROLINA_BLUE}55;
    box-shadow: 0 6px 18px -10px {CAROLINA_BLUE}55;
}}
.review-card .meta {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.55rem;
}}
.review-card .stars {{ color: {ACCENT_AMBER}; font-size: 1rem; letter-spacing: 2px; }}
.review-card .course {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: {CAROLINA_BLUE};
    background: {CAROLINA_BLUE}15;
    padding: 0.18rem 0.6rem;
    border-radius: 8px;
}}
.review-card .text {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.95rem;
    line-height: 1.6;
    color: {TEXT_PRIMARY}dd;
}}

/* ── Professor rank cards ────────────────────────── */
.rank-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: transform 0.18s, border-color 0.18s, box-shadow 0.18s;
    box-shadow: 0 1px 3px #0003;
}}
.rank-card:hover {{
    transform: translateY(-2px);
    border-color: {CAROLINA_BLUE}88;
    box-shadow: 0 12px 28px -14px {CAROLINA_BLUE}66, 0 2px 4px #0004;
}}
.rank-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    min-width: 2.4rem;
    text-align: center;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.4rem;
    height: 2.4rem;
    border-radius: 14px;
}}
/* Podium treatment for top 3 */
.rank-1 .rank-num {{
    background: linear-gradient(135deg, #FFD70033, #FFA50022);
    color: #FFD700;
    box-shadow: 0 0 18px #FFD70033;
}}
.rank-2 .rank-num {{
    background: linear-gradient(135deg, #E8E8E833, #B0B0B022);
    color: #E0E0E0;
}}
.rank-3 .rank-num {{
    background: linear-gradient(135deg, #CD7F3233, #8B451322);
    color: #CD7F32;
}}
.rank-other .rank-num {{
    color: {CAROLINA_BLUE};
    background: {CAROLINA_BLUE}14;
}}
.rank-info {{ flex: 1; min-width: 0; }}
.rank-name {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.12rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    display: flex; align-items: center; gap: 0.55rem; flex-wrap: wrap;
}}
.rank-meta {{
    font-size: 0.8rem;
    color: {TEXT_MUTED};
    margin-top: 0.2rem;
}}
.rank-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
}}

/* ── Sidebar ─────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {CARD_BG};
    border-right: 1px solid {BORDER};
}}
.sb-brand {{
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.3rem 0 0.9rem 0;
}}
.sb-brand .compass {{ color: {CAROLINA_BLUE}; }}
.sb-brand-title {{
    font-family: 'Source Serif 4', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1.1;
}}
.sb-brand-sub {{
    font-size: 0.75rem;
    color: {TEXT_MUTED};
    margin-top: 0.15rem;
}}
.sb-section-label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXT_MUTED};
    margin: 0.8rem 0 0.4rem 0;
}}
.sb-stat-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.35rem 0;
    font-size: 0.83rem;
    color: {TEXT_MUTED};
}}
.sb-stat-row b {{ color: {TEXT_PRIMARY}; font-weight: 600; }}
.sb-credit {{
    font-size: 0.7rem;
    color: {TEXT_MUTED};
    line-height: 1.5;
    padding-top: 0.4rem;
}}
.sb-credit code {{
    background: transparent;
    color: {CAROLINA_BLUE};
    font-size: 0.7rem;
}}

/* ── Topic result rows (Live Analysis) ───────────── */
.topic-result {{
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.35rem;
    border-radius: 12px;
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-left: 3px solid var(--topic-color, {CAROLINA_BLUE});
}}
.topic-name {{
    font-weight: 500;
    color: {TEXT_PRIMARY};
    flex: 1;
}}

/* Live-analysis verdict badge */
.verdict {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.55rem 1rem;
    border-radius: 14px;
    font-weight: 600;
    font-size: 0.9rem;
    margin: 0.3rem 0 0.8rem 0;
}}
.verdict-agree {{
    color: {ACCENT_GREEN};
    background: {ACCENT_GREEN}14;
    border: 1px solid {ACCENT_GREEN}55;
}}
.verdict-differ {{
    color: {ACCENT_AMBER};
    background: {ACCENT_AMBER}14;
    border: 1px solid {ACCENT_AMBER}55;
}}

/* ── Tabs ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.35rem;
    background: transparent;
    border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    letter-spacing: 0.02em;
    border-radius: 10px 10px 0 0;
    padding: 0.5rem 1rem;
}}
.stTabs [aria-selected="true"] {{ color: {CAROLINA_BLUE} !important; }}

/* ── Section headers ─────────────────────────────── */
.section-header {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin: 1.6rem 0 0.9rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid {CAROLINA_BLUE}55;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* ── Streamlit slider playful tint ───────────────── */
[data-testid="stSlider"] [role="slider"] {{
    box-shadow: 0 0 0 4px {CAROLINA_BLUE}22 !important;
}}

/* Preset chip buttons — secondary buttons only, keeps Analyze untouched */
.stButton > button[kind="secondary"] {{
    border-radius: 999px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: all 0.18s !important;
    border-color: {BORDER} !important;
    color: {TEXT_MUTED} !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: {CAROLINA_BLUE} !important;
    color: {CAROLINA_BLUE} !important;
    background: {CAROLINA_BLUE}14 !important;
    transform: translateY(-1px);
}}
</style>
""",
    unsafe_allow_html=True,
)

# ── Data Loading ─────────────────────────────────────────────────────────────


@st.cache_data
def load_reviews() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "reviews_all.parquet")


@st.cache_data
def load_scores() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "scored_all.parquet")


@st.cache_data
def load_eval_results() -> dict | None:
    path = DATA_DIR / "evaluation_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def aggregate_prof_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_professor_scores(scores_df)


def normalize_course(name: str) -> str:
    """Normalize course names: 'STOR-155' / 'STOR 155' / 'STOR155' → 'STOR 155'."""
    name = name.strip().upper().replace("-", "").replace(" ", "")
    m = re.match(r"^([A-Z]+)(\d+[A-Z]?)$", name)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return name


@st.cache_data
def get_course_list(reviews_df: pd.DataFrame) -> list[str]:
    """Get sorted unique normalized course names with a letter prefix."""
    courses = reviews_df["course_name"].dropna().apply(normalize_course)
    counts = courses.value_counts()
    valid = counts[counts >= 3].index.tolist()
    # Exclude bare numbers (no department prefix)
    valid = [c for c in valid if re.match(r"^[A-Z]", c)]
    return sorted(valid)


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


def metric_card(
    value: str,
    label: str,
    icon_svg: str | None = None,
    icon_color: str = CAROLINA_BLUE,
) -> str:
    icon_html = (
        f'<div class="icon" style="color:{icon_color};background:{icon_color}14;">{icon_svg}</div>'
        if icon_svg
        else ""
    )
    return (
        f'<div class="metric-card">{icon_html}'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>'
    )


def review_card(text: str, rating: int, course: str) -> str:
    stars = "★" * rating + "☆" * (5 - rating)
    return (
        f'<div class="review-card">'
        f'<div class="meta"><span class="stars">{stars}</span>'
        f'<span class="course">{course}</span></div>'
        f'<div class="text">{text}</div></div>'
    )


def topic_result_html(topic: str, sentiment: str) -> str:
    icon = topic_icon(topic)
    color = TOPIC_COLORS.get(topic, CAROLINA_BLUE)
    return (
        f'<div class="topic-result" style="--topic-color:{color};">'
        f'{icon}<span class="topic-name">{topic}</span>'
        f"{sentiment_pill(sentiment)}</div>"
    )


# ── Global counts (computed once; feed both sidebar and hero) ────────────────

_scores_global = load_scores()
N_REVIEWS = int(len(_scores_global))
N_PROFS = int(_scores_global["professor_name"].nunique())
N_DEPTS = int(_scores_global["department"].nunique())
N_TOPICS = len(TOPICS)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"""<div class="sb-brand">
            <span class="compass">{COMPASS_SVG.replace('width="38"', 'width="26"').replace('height="38"', 'height="26"')}</span>
            <div>
                <div class="sb-brand-title">Course Compass</div>
                <div class="sb-brand-sub">UNC Chapel Hill</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="sb-section-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sb-stat-row"><span>Reviews</span><b>{N_REVIEWS:,}</b></div>'
        f'<div class="sb-stat-row"><span>Professors</span><b>{N_PROFS:,}</b></div>'
        f'<div class="sb-stat-row"><span>Departments</span><b>{N_DEPTS}</b></div>'
        f'<div class="sb-stat-row"><span>Topics</span><b>{N_TOPICS}</b></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sb-section-label">Models</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sb-stat-row"><span>Fine-tuned</span><b style="color:{CAROLINA_BLUE};">DistilBERT</b></div>'
        f'<div class="sb-stat-row"><span>Zero-shot</span><b>BART + RoBERTa</b></div>'
        f'<div class="sb-stat-row"><span>Labeled by</span><b>Sonnet 4.6</b></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-section-label">Built with</div>', unsafe_allow_html=True
    )
    st.markdown(
        f"""<div class="sb-credit">
            <code>PyTorch</code> · <code>Transformers</code><br>
            <code>Streamlit</code> · <code>Plotly</code>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Hero ─────────────────────────────────────────────────────────────────────

# Inject dynamic count-up keyframes targeting the actual data counts.
st.markdown(
    f"""<style>
    .cu-reviews {{ animation: ku-r 1.8s cubic-bezier(.2,.9,.3,1) forwards; counter-reset: n var(--cu-r); }}
    .cu-reviews::after {{ content: counter(n); }}
    @keyframes ku-r {{ from {{ --cu-r: 0; }} to {{ --cu-r: {N_REVIEWS}; }} }}

    .cu-profs   {{ animation: ku-p 1.6s cubic-bezier(.2,.9,.3,1) 0.15s forwards; counter-reset: n var(--cu-p); }}
    .cu-profs::after   {{ content: counter(n); }}
    @keyframes ku-p {{ from {{ --cu-p: 0; }} to {{ --cu-p: {N_PROFS}; }} }}

    .cu-depts   {{ animation: ku-d 1.2s cubic-bezier(.2,.9,.3,1) 0.3s forwards; counter-reset: n var(--cu-d); }}
    .cu-depts::after   {{ content: counter(n); }}
    @keyframes ku-d {{ from {{ --cu-d: 0; }} to {{ --cu-d: {N_DEPTS}; }} }}

    .cu-topics  {{ animation: ku-t 0.9s cubic-bezier(.2,.9,.3,1) 0.45s forwards; counter-reset: n var(--cu-t); }}
    .cu-topics::after  {{ content: counter(n); }}
    @keyframes ku-t {{ from {{ --cu-t: 0; }} to {{ --cu-t: {N_TOPICS}; }} }}
    </style>""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""<div class="hero">
    <div class="hero-title-row">
        <span class="hero-compass">{COMPASS_SVG}</span>
        <h1>UNC Course Compass</h1>
    </div>
    <p class="tagline">Aspect-based sentiment analysis across every <span class="accent">UNC Chapel Hill</span> course review</p>
    <div class="hero-stats">
        <span class="hero-chip"><b class="cu-reviews"></b>reviews</span>
        <span class="hero-chip"><b class="cu-profs"></b>professors</span>
        <span class="hero-chip"><b class="cu-depts"></b>departments</span>
        <span class="hero-chip"><b class="cu-topics"></b>sentiment topics</span>
    </div>
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
    scores_df = load_scores()

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
                color = TOPIC_COLORS.get(topic, CAROLINA_BLUE)
                st.markdown(
                    f"""<div class="topic-result" style="--topic-color:{color};">
                    {topic_icon(topic)}
                    <span class="topic-name">{topic}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:{TEXT_MUTED};">{total} mentions</span>
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
        '<div class="section-header">How important is each aspect to you?</div>',
        unsafe_allow_html=True,
    )

    # Preset quick-picks — values in TOPIC_KEYS order: workload, grading, teaching, access, exam
    PRESETS = {
        "Balanced": (5, 5, 5, 5, 5),
        "Easy A": (8, 10, 4, 4, 10),
        "Great teachers": (3, 3, 10, 9, 3),
        "Light workload": (10, 5, 5, 4, 8),
    }
    # Pre-seed slider state so widgets don't carry an explicit `value=` default
    # (Streamlit warns when a keyed widget has both a default and API-set state).
    for _key in TOPIC_KEYS:
        st.session_state.setdefault(f"w_{_key}", 5)

    def _apply_preset(name: str) -> None:
        # Callbacks run before the rerun and only touch weight keys, so
        # other widgets (course filter, min_reviews) keep their state.
        for k, v in zip(TOPIC_KEYS, PRESETS[name]):
            st.session_state[f"w_{k}"] = v

    st.markdown(
        f'<div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:{TEXT_MUTED};margin:0.2rem 0 0.45rem 0;">Quick picks</div>',
        unsafe_allow_html=True,
    )
    preset_cols = st.columns(len(PRESETS) + 1)
    for i, (name, vals) in enumerate(PRESETS.items()):
        with preset_cols[i]:
            st.button(
                name,
                key=f"preset_{name}",
                use_container_width=True,
                on_click=_apply_preset,
                args=(name,),
            )
    st.markdown(
        f'<div style="font-size:0.75rem;color:{TEXT_MUTED};margin:0.8rem 0 0.3rem 0;">Or tune each aspect yourself:</div>',
        unsafe_allow_html=True,
    )

    slider_cols = st.columns(5)
    weights = {}
    for i, (topic, key) in enumerate(zip(TOPICS, TOPIC_KEYS)):
        with slider_cols[i]:
            color = TOPIC_COLORS.get(topic, CAROLINA_BLUE)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.4rem;font-size:0.88rem;font-weight:500;color:{TEXT_PRIMARY};margin-bottom:-0.25rem;">'
                f"{topic_icon(topic, color=color)}<span>{topic}</span></div>",
                unsafe_allow_html=True,
            )
            weights[key] = st.slider(
                topic,
                min_value=0,
                max_value=10,
                key=f"w_{key}",
                label_visibility="collapsed",
            )
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:0.68rem;color:{TEXT_MUTED};margin-top:-0.65rem;letter-spacing:0.02em;">'
                f"<span>Not important</span><span>Must have</span></div>",
                unsafe_allow_html=True,
            )

    min_reviews = st.slider(
        "Minimum reviews",
        1,
        30,
        3,
        key="min_rev",
        help="Scores for low-review profs are shrunk toward their department mean — lower this slider to surface new profs without letting 1-2 lucky reviews dominate.",
    )

    scores_df = load_scores()
    reviews_df_rec = load_reviews()

    # Course filter
    all_courses = get_course_list(reviews_df_rec)
    selected_courses = st.multiselect(
        "Filter by course (optional)",
        all_courses,
        default=[],
        placeholder="All courses",
        key="course_filter",
    )

    # If courses selected, filter scores to only reviews from those courses
    filtered_scores = scores_df
    if selected_courses:
        filtered_scores = scores_df.copy()
        filtered_scores["_norm_course"] = filtered_scores["course_name"].apply(
            normalize_course
        )
        filtered_scores = filtered_scores[
            filtered_scores["_norm_course"].isin(selected_courses)
        ]
        filtered_scores = filtered_scores.drop(columns=["_norm_course"])

    agg_df = aggregate_prof_scores(filtered_scores)
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

            rank_position = rank_idx + 1
            rank_class = f"rank-{rank_position}" if rank_position <= 3 else "rank-other"
            n_rev = int(row["num_reviews"])
            dept = row.get("department", "")
            conf_pill = (
                f'<span class="pill-confidence">Based on {n_rev} review{"" if n_rev == 1 else "s"}</span>'
                if n_rev < 5
                else ""
            )
            dept_badge = (
                f'<span style="font-size:0.72rem;color:{TEXT_MUTED};">· {dept}</span>'
                if dept
                else ""
            )

            with st.container():
                col_info, col_chart = st.columns([3, 2], gap="medium")

                with col_info:
                    st.markdown(
                        f"""<div class="rank-card {rank_class}">
                        <div class="rank-num">{rank_position}</div>
                        <div class="rank-info">
                            <div class="rank-name">{row["professor_name"]}{conf_pill}</div>
                            <div class="rank-meta">{n_rev} review{"" if n_rev == 1 else "s"} {dept_badge}</div>
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
        pts_data = eval_data.get("per_topic_sentiment", {})
        sent_data = eval_data.get("overall_sentiment", {})

        # Overall sentiment — headline cards
        st.markdown(
            f'<div class="section-header">Overall Sentiment Accuracy</div>',
            unsafe_allow_html=True,
        )
        if sent_data:
            cols = st.columns(3)
            model_specs = [
                ("baseline", "Star Baseline", STAR_ICON, ACCENT_AMBER),
                ("zero_shot", "Zero-Shot", ZAP_ICON, TEXT_MUTED),
                ("finetuned", "Fine-Tuned", SPARK_ICON, ACCENT_GREEN),
            ]
            # Find the winner so we can highlight it
            best_key = max(
                (k for k, _, _, _ in model_specs),
                key=lambda k: sent_data.get(k, {}).get("accuracy", 0),
            )
            for i, (model_key, label, icon, color) in enumerate(model_specs):
                data = sent_data.get(model_key, {})
                acc = data.get("accuracy", 0)
                f1 = data.get("f1_macro", 0)
                is_winner = model_key == best_key
                winner_badge = (
                    f'<div style="position:absolute;top:-10px;right:-10px;background:{ACCENT_GREEN};color:#fff;font-size:0.65rem;font-weight:700;letter-spacing:0.08em;padding:0.25rem 0.6rem;border-radius:999px;text-transform:uppercase;box-shadow:0 4px 12px {ACCENT_GREEN}66;">Best</div>'
                    if is_winner
                    else ""
                )
                with cols[i]:
                    st.markdown(
                        f'<div style="position:relative;">{winner_badge}{metric_card(f"{acc:.1%}", label, icon_svg=icon, icon_color=color)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="text-align:center; font-size:0.78rem; color:{TEXT_MUTED}; margin-top:0.3rem;">F1 Macro: <span style="color:{TEXT_PRIMARY};">{f1:.3f}</span></div>',
                        unsafe_allow_html=True,
                    )

        # Per-topic: side-by-side bar chart
        st.markdown(
            f'<div class="section-header">Per-Topic Sentiment Accuracy</div>',
            unsafe_allow_html=True,
        )
        if pts_data:
            zs_accs = []
            ft_accs = []
            topic_labels = []
            for topic in TOPICS:
                zs_info = pts_data.get("zero_shot", {}).get(topic, {})
                ft_info = pts_data.get("finetuned", {}).get(topic, {})
                zs_acc = (zs_info.get("accuracy", 0) or 0) * 100
                ft_acc = (ft_info.get("accuracy", 0) or 0) * 100
                n = ft_info.get("n", 0)
                topic_labels.append(f"{topic} (n={n})")
                zs_accs.append(zs_acc)
                ft_accs.append(ft_acc)

            ft_colors = [TOPIC_COLORS[t] for t in TOPICS]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="Zero-Shot",
                    x=topic_labels,
                    y=zs_accs,
                    marker=dict(color="#4B5563", line=dict(width=0)),
                    text=[f"{v:.1f}%" for v in zs_accs],
                    textposition="outside",
                    textfont=dict(color=TEXT_MUTED, size=11),
                )
            )
            fig.add_trace(
                go.Bar(
                    name="Fine-Tuned",
                    x=topic_labels,
                    y=ft_accs,
                    marker=dict(color=ft_colors, line=dict(width=0)),
                    text=[f"<b>{v:.1f}%</b>" for v in ft_accs],
                    textposition="outside",
                    textfont=dict(color=TEXT_PRIMARY, size=12),
                )
            )
            fig.update_layout(
                barmode="group",
                bargap=0.3,
                bargroupgap=0.08,
                height=400,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_PRIMARY, family="DM Sans"),
                yaxis=dict(
                    title="Accuracy %",
                    range=[0, 105],
                    gridcolor="#30363D",
                    zeroline=False,
                ),
                xaxis=dict(gridcolor="#30363D"),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                margin=dict(l=40, r=20, t=40, b=80),
            )
            st.plotly_chart(fig, width="stretch", key="topic_acc_bar")

        # Topic detection — side-by-side cards
        st.markdown(
            f'<div class="section-header">Topic Detection F1</div>',
            unsafe_allow_html=True,
        )
        if pts_data:
            # Per-topic detection bar chart
            zs_det = pts_data.get("zero_shot", {}).get("_topic_detection", {})
            ft_det = pts_data.get("finetuned", {}).get("_topic_detection", {})
            zs_per = zs_det.get("f1_per_topic", {})
            ft_per = ft_det.get("f1_per_topic", {})

            det_cols = st.columns(2)
            with det_cols[0]:
                st.markdown(
                    metric_card(
                        f"{zs_det.get('f1_macro', 0):.3f}", "Zero-Shot -- Macro"
                    ),
                    unsafe_allow_html=True,
                )
            with det_cols[1]:
                st.markdown(
                    metric_card(
                        f"{ft_det.get('f1_macro', 0):.3f}", "Fine-Tuned -- Macro"
                    ),
                    unsafe_allow_html=True,
                )

            ft_colors_det = [TOPIC_COLORS[t] for t in TOPICS]

            fig2 = go.Figure()
            fig2.add_trace(
                go.Bar(
                    name="Zero-Shot",
                    x=TOPICS,
                    y=[zs_per.get(t, 0) for t in TOPICS],
                    marker=dict(color="#4B5563", line=dict(width=0)),
                    text=[f"{zs_per.get(t, 0):.2f}" for t in TOPICS],
                    textposition="outside",
                    textfont=dict(color=TEXT_MUTED, size=11),
                )
            )
            fig2.add_trace(
                go.Bar(
                    name="Fine-Tuned",
                    x=TOPICS,
                    y=[ft_per.get(t, 0) for t in TOPICS],
                    marker=dict(color=ft_colors_det, line=dict(width=0)),
                    text=[f"<b>{ft_per.get(t, 0):.2f}</b>" for t in TOPICS],
                    textposition="outside",
                    textfont=dict(color=TEXT_PRIMARY, size=12),
                )
            )
            fig2.update_layout(
                barmode="group",
                bargap=0.3,
                bargroupgap=0.08,
                height=370,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_PRIMARY, family="DM Sans"),
                yaxis=dict(
                    title="F1 Score",
                    range=[0, 1.08],
                    gridcolor="#30363D",
                    zeroline=False,
                ),
                xaxis=dict(gridcolor="#30363D"),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                margin=dict(l=40, r=20, t=40, b=60),
            )
            st.plotly_chart(fig2, width="stretch", key="topic_det_bar")

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
        # Run both models first so we can compute the verdict before rendering.
        zs_topic_sents: dict[str, str] | None = None
        zs_error: str | None = None
        with st.spinner("Running BART + RoBERTa..."):
            try:
                tc, sa = load_zero_shot_models()
                zs_topic_sents = sa.analyze_by_topic_flat(review_text, tc)
            except Exception as e:
                zs_error = str(e)

        ft_results: dict[str, str] | None = None
        ft_error: str | None = None
        joint_dir = MODELS_DIR / "joint_classifier"
        if not joint_dir.exists():
            ft_error = "Fine-tuned model not found."
        else:
            with st.spinner("Running joint classifier..."):
                try:
                    ft_results = predict_joint([review_text])[0]
                except Exception as e:
                    ft_error = str(e)

        # Verdict: how many detected topics agree in sentiment?
        if zs_topic_sents and ft_results:
            shared = set(zs_topic_sents) & set(ft_results)
            if shared:
                agree = sum(1 for t in shared if zs_topic_sents[t] == ft_results[t])
                total = len(shared)
                if agree == total:
                    verdict_html = (
                        f'<div class="verdict verdict-agree">'
                        f"Models agree on all {total} shared topic{'s' if total != 1 else ''}"
                        f"</div>"
                    )
                else:
                    diffs = [
                        f"<b style='color:{TOPIC_COLORS.get(t, CAROLINA_BLUE)};'>{t}</b>"
                        for t in shared
                        if zs_topic_sents[t] != ft_results[t]
                    ]
                    verdict_html = (
                        f'<div class="verdict verdict-differ">'
                        f"Models disagree on {len(diffs)}/{total}: {', '.join(diffs)}"
                        f"</div>"
                    )
                st.markdown(verdict_html, unsafe_allow_html=True)
            # Also flag topic-coverage differences
            only_ft = set(ft_results) - set(zs_topic_sents)
            only_zs = set(zs_topic_sents) - set(ft_results)
            if only_ft or only_zs:
                extras = []
                if only_ft:
                    extras.append(f"Fine-Tuned caught {', '.join(sorted(only_ft))}")
                if only_zs:
                    extras.append(f"Zero-Shot caught {', '.join(sorted(only_zs))}")
                st.markdown(
                    f'<div style="font-size:0.8rem;color:{TEXT_MUTED};margin-bottom:0.8rem;">{" · ".join(extras)}</div>',
                    unsafe_allow_html=True,
                )

        col_zero, col_fine = st.columns(2, gap="large")
        with col_zero:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;font-family:Source Serif 4,serif;font-size:1.1rem;font-weight:600;color:{TEXT_MUTED};margin-bottom:0.8rem;">'
                f'<span style="display:inline-flex;color:{TEXT_MUTED};">{ZAP_ICON}</span>'
                f"Zero-Shot Pipeline</div>",
                unsafe_allow_html=True,
            )
            if zs_error:
                st.error(f"Error: {zs_error}")
            elif zs_topic_sents:
                st.markdown(
                    "".join(topic_result_html(t, s) for t, s in zs_topic_sents.items()),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No topics detected.")

        with col_fine:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;font-family:Source Serif 4,serif;font-size:1.1rem;font-weight:600;color:{ACCENT_GREEN};margin-bottom:0.8rem;">'
                f'<span style="display:inline-flex;color:{ACCENT_GREEN};">{SPARK_ICON}</span>'
                f"Fine-Tuned DistilBERT</div>",
                unsafe_allow_html=True,
            )
            if ft_error:
                st.warning(ft_error)
            elif ft_results:
                st.markdown(
                    "".join(topic_result_html(t, s) for t, s in ft_results.items()),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No topics detected.")
