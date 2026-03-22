import os
import json
import logging
import requests
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import streamlit as st
from groq import Groq
from datetime import datetime, timezone
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
NEWS_API_URL         = 'https://newsapi.org/v2/everything'
MAX_ARTICLES         = 20
REQUEST_TIMEOUT      = 10
SENTIMENT_MODEL      = 'ProsusAI/finbert'           # trained on financial/tech news
SUMMARIZATION_MODEL  = 'sshleifer/distilbart-cnn-12-6'
GROQ_MODEL           = 'llama-3.3-70b-versatile'
SENTIMENT_BATCH_SIZE = 8
SUMMARY_TOP_N        = 10
SUMMARY_MAX_LEN      = 180
SUMMARY_MIN_LEN      = 50
COLORS               = {'positive': '#66b3ff', 'negative': '#ff9999', 'neutral': '#d3d3d3'}
IMPACT_SECTORS       = ['Global Markets', 'Politics', 'Technology', 'Energy', 'Public Health', 'Trade & Economy']

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title='News Sentiment Dashboard', page_icon='📰', layout='wide')

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

    /* ── Global dark theme ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #070b14 !important;
        color: #c8d8e8 !important;
    }
    [data-testid="stMain"], .main, .block-container {
        background-color: #070b14 !important;
        padding-top: 1.8rem !important;
        font-family: 'Exo 2', sans-serif !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0b1120 !important;
        border-right: 1px solid #0ff2 !important;
    }
    [data-testid="stSidebar"] * { color: #a0c4d8 !important; font-family: 'Exo 2', sans-serif !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: 1px solid #0ff4 !important;
        color: #00e5ff !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.05em !important;
        transition: all 0.2s !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #00e5ff18 !important;
        border-color: #00e5ff !important;
        box-shadow: 0 0 12px #00e5ff44 !important;
    }

    /* ── Primary Analyze button ── */
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #00b4d8, #00e5ff) !important;
        border: none !important;
        color: #070b14 !important;
        font-weight: 700 !important;
        font-family: 'Exo 2', sans-serif !important;
        letter-spacing: 0.08em !important;
        box-shadow: 0 0 20px #00e5ff55 !important;
        transition: all 0.3s !important;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 0 35px #00e5ff88 !important;
        transform: translateY(-1px) !important;
    }

    /* ── Text inputs & sliders ── */
    [data-testid="stTextInput"] input {
        background: #0f1a2e !important;
        border: 1px solid #0ff3 !important;
        color: #00e5ff !important;
        font-family: 'Share Tech Mono', monospace !important;
        border-radius: 4px !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #00e5ff !important;
        box-shadow: 0 0 10px #00e5ff33 !important;
    }
    [data-testid="stSlider"] * { color: #00e5ff !important; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #0b1627 !important;
        border: 1px solid #0ff2 !important;
        border-radius: 8px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 0 20px #00e5ff0a !important;
        transition: box-shadow 0.3s !important;
    }
    [data-testid="metric-container"]:hover {
        box-shadow: 0 0 30px #00e5ff22 !important;
        border-color: #00e5ff44 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Share Tech Mono', monospace !important;
        color: #00e5ff !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #5d8a9e !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 1px solid #0ff2 !important;
        gap: 0 !important;
    }
    [data-testid="stTabs"] button[role="tab"] {
        background: transparent !important;
        color: #4a7a8e !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.1em !important;
        border: none !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #00e5ff !important;
        border-bottom: 2px solid #00e5ff !important;
        text-shadow: 0 0 10px #00e5ff88 !important;
    }

    /* ── Expanders (article cards) ── */
    [data-testid="stExpander"] {
        background: #0b1627 !important;
        border: 1px solid #0ff2 !important;
        border-radius: 6px !important;
        margin-bottom: 0.5rem !important;
        transition: border-color 0.2s !important;
    }
    [data-testid="stExpander"]:hover { border-color: #00e5ff44 !important; }
    [data-testid="stExpander"] summary {
        color: #c8d8e8 !important;
        font-family: 'Exo 2', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.93rem !important;
    }
    [data-testid="stExpander"] summary:hover { color: #00e5ff !important; }

    /* ── Dividers ── */
    hr { border-color: #0ff1 !important; }

    /* ── Status / progress box ── */
    [data-testid="stStatus"] {
        background: #0b1627 !important;
        border: 1px solid #0ff2 !important;
        color: #a0c4d8 !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
        background: #0b1627 !important;
        border: 1px solid #00e5ff44 !important;
        color: #a0c4d8 !important;
        font-family: 'Exo 2', sans-serif !important;
    }

    /* ── Radio buttons ── */
    [data-testid="stRadio"] label { color: #7aaec0 !important; font-size: 0.85rem !important; }
    [data-testid="stRadio"] [aria-checked="true"] + div { color: #00e5ff !important; }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] button {
        background: #0b1627 !important;
        border: 1px solid #00e5ff66 !important;
        color: #00e5ff !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 0.1em !important;
        transition: all 0.3s !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: #00e5ff15 !important;
        box-shadow: 0 0 20px #00e5ff44 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #070b14; }
    ::-webkit-scrollbar-thumb { background: #00e5ff44; border-radius: 2px; }

    /* ── Custom components ── */
    .main-title {
        font-family: 'Exo 2', sans-serif !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.02em !important;
        color: #ffffff !important;
        text-shadow: 0 0 30px #00e5ff66 !important;
        margin-bottom: 0 !important;
    }
    .subtitle {
        font-family: 'Share Tech Mono', monospace !important;
        color: #3a6a7e !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        margin-top: 0.2rem !important;
    }
    .summary-box {
        background: #0b1627;
        border-left: 3px solid #00e5ff;
        padding: 1.1rem 1.4rem;
        border-radius: 0 6px 6px 0;
        font-family: 'Exo 2', sans-serif;
        font-size: 0.96rem;
        line-height: 1.75;
        color: #c8d8e8;
        box-shadow: inset 0 0 40px #00e5ff08, 0 0 20px #00000044;
    }
    .impact-card {
        background: #0c1828;
        border-left: 3px solid #00b4d8;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.5rem;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.82rem;
        line-height: 1.65;
        color: #a0c4d8;
        transition: border-color 0.2s;
    }
    .impact-card:hover { border-left-color: #00e5ff; }
    .article-meta {
        font-family: 'Share Tech Mono', monospace;
        color: #3a6a7e;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }
    .badge-pos {
        background: #00e5ff18;
        color: #00e5ff;
        border: 1px solid #00e5ff44;
        border-radius: 3px;
        padding: 1px 8px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
    }
    .badge-neg {
        background: #ff3c5218;
        color: #ff6b8a;
        border: 1px solid #ff3c5244;
        border-radius: 3px;
        padding: 1px 8px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
    }

    /* ── Subheaders ── */
    h2, h3, [data-testid="stSubheader"] {
        font-family: 'Exo 2', sans-serif !important;
        color: #c8d8e8 !important;
        letter-spacing: 0.04em !important;
        border-bottom: 1px solid #0ff1 !important;
        padding-bottom: 0.3rem !important;
    }
    p, li, .stMarkdown { color: #8aaec0 !important; }

    /* ── Caption ── */
    [data-testid="stCaptionContainer"] p {
        font-family: 'Share Tech Mono', monospace !important;
        color: #2a5a6e !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'results' not in st.session_state:
    st.session_state.results = None

# ─── API Keys ────────────────────────────────────────────────────────────────
NEWS_API_KEY  = os.environ.get('NEWS_API_KEY', '').strip()
GROQ_API_KEY  = os.environ.get('GROQ_API_KEY', '').strip()

if not NEWS_API_KEY:
    st.error(' NEWS_API_KEY is not set. Add it to your .env file.')
    st.stop()

if not GROQ_API_KEY:
    st.warning(' GROQ_API_KEY is not set — Impact Analysis will be disabled. Add it to your .env file.')

# ─── Model Loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline('sentiment-analysis', model=SENTIMENT_MODEL,
                    device=device, truncation=True, max_length=512)

@st.cache_resource(show_spinner=False)
def load_summarization_model():
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL)
    model.eval()
    return tokenizer, model

# ─── Core Functions ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_articles(api_key: str, query: str, page_size: int) -> list:
    params = {
        'q': query, 'language': 'en',
        'pageSize': page_size, 'sortBy': 'publishedAt', 'apiKey': api_key
    }
    r = requests.get(NEWS_API_URL, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if data.get('status') != 'ok':
        raise RuntimeError(data.get('message', 'Unknown API error'))
    raw = data.get('articles', [])
    return [a for a in raw if a.get('title') and '[Removed]' not in a['title']]


def clean_articles(articles: list) -> pd.DataFrame:
    df = pd.DataFrame(articles)
    df['source_name'] = df['source'].apply(
        lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown'
    )
    out = df[['title', 'description', 'publishedAt', 'url', 'source_name']].copy()
    out['title']       = out['title'].str.split(' - ').str[0].str.strip()
    out['description'] = out['description'].fillna('').str.strip().replace('', 'No description available.')
    out['publishedAt'] = pd.to_datetime(out['publishedAt'], utc=True, errors='coerce')
    out = out.dropna(subset=['publishedAt'])
    return out.sort_values('publishedAt', ascending=False).reset_index(drop=True)


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    model = load_sentiment_model()
    # Combine title + description for better context (FinBERT understands news language)
    texts = (
        df['title'] + '. ' +
        df['description'].replace('No description available.', '')
    ).str.strip('. ').tolist()
    results = []
    with torch.no_grad():
        for i in range(0, len(texts), SENTIMENT_BATCH_SIZE):
            results.extend(model(texts[i:i + SENTIMENT_BATCH_SIZE]))
    # FinBERT returns lowercase labels: positive / negative / neutral
    df['sentiment_label'] = [r['label'].lower() for r in results]
    df['sentiment_score'] = [round(r['score'], 4) for r in results]
    return df


def run_summarization(df: pd.DataFrame) -> str:
    tokenizer, model = load_summarization_model()
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    top      = df.head(SUMMARY_TOP_N)
    combined = ' . '.join(
        f"{row['title']}: {row['description']}"
        for _, row in top.iterrows()
        if row['description'] != 'No description available.'
    )
    if not combined.strip():
        return ''
    inputs = tokenizer(combined, max_length=1024, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        ids = model.generate(
            inputs['input_ids'], attention_mask=inputs['attention_mask'],
            max_length=SUMMARY_MAX_LEN, min_length=SUMMARY_MIN_LEN,
            length_penalty=2.0, num_beams=4, early_stopping=True
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def run_impact_analysis(topic: str, summary: str, sentiment_ratio: float) -> dict | None:
    """Use Groq (free) to analyse how this news impacts related sectors."""
    if not GROQ_API_KEY:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""You are an expert analyst. Given this news topic and summary, analyse the potential impact on each of these sectors: {', '.join(IMPACT_SECTORS)}.

Topic: {topic}
Summary: {summary}
Overall sentiment: {'Mostly positive' if sentiment_ratio >= 0.5 else 'Mostly negative'} ({sentiment_ratio:.0%} positive)

Respond ONLY with a valid JSON object — no preamble, no explanation, no markdown code fences. Format exactly:
{{
  "sectors": {{
    "Global Markets": {{"score": 7, "direction": "negative", "reason": "one sentence explanation"}},
    "Politics": {{"score": 4, "direction": "positive", "reason": "one sentence explanation"}},
    "Technology": {{"score": 2, "direction": "neutral", "reason": "one sentence explanation"}},
    "Energy": {{"score": 8, "direction": "negative", "reason": "one sentence explanation"}},
    "Public Health": {{"score": 1, "direction": "neutral", "reason": "one sentence explanation"}},
    "Trade & Economy": {{"score": 6, "direction": "negative", "reason": "one sentence explanation"}}
  }},
  "headline_impact": "One sentence overall impact statement.",
  "most_affected": "Sector name"
}}

Score is 0-10 (how strongly impacted). Direction must be: positive, negative, or neutral."""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=800,
            temperature=0.3,
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if model adds them anyway
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        log.error('Impact analysis failed: %s', e)
        return None


# ─── Chart Functions ──────────────────────────────────────────────────────────
CHART_BG    = '#070b14'
CHART_PANEL = '#0b1627'
CHART_TEXT  = '#5d8a9e'
CHART_GRID  = '#0f2035'
CYAN        = '#00e5ff'
CORAL       = '#ff6b8a'
MUTED       = '#3a5a6e'

def _apply_dark_style(fig, ax_list):
    """Apply consistent dark theme to any matplotlib figure."""
    fig.patch.set_facecolor(CHART_BG)
    for ax in (ax_list if isinstance(ax_list, (list, np.ndarray)) else [ax_list]):
        ax.set_facecolor(CHART_PANEL)
        ax.tick_params(colors=CHART_TEXT, labelsize=8)
        ax.xaxis.label.set_color(CHART_TEXT)
        ax.yaxis.label.set_color(CHART_TEXT)
        ax.title.set_color('#c8d8e8')
        for spine in ax.spines.values():
            spine.set_edgecolor(CHART_GRID)
        ax.grid(color=CHART_GRID, linewidth=0.5)


def chart_sentiment_pie_and_hist(df: pd.DataFrame, topic: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=CHART_BG)
    fig.suptitle(f'SENTIMENT OVERVIEW — {topic.upper()}', fontsize=11,
                 fontweight='bold', color='#c8d8e8', fontfamily='monospace')

    counts       = df['sentiment_label'].value_counts()
    dark_colors  = {'positive': CYAN, 'negative': CORAL, 'neutral': MUTED}
    chart_colors = [dark_colors.get(l, '#445566') for l in counts.index]

    axes[0].set_facecolor(CHART_BG)
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=[l.upper() for l in counts.index],
        autopct='%1.1f%%', colors=chart_colors,
        startangle=140, wedgeprops={'edgecolor': CHART_BG, 'linewidth': 2}
    )
    for t in texts: t.set_color(CHART_TEXT); t.set_fontsize(8)
    for at in autotexts: at.set_color('#070b14'); at.set_fontweight('bold'); at.set_fontsize(8)
    axes[0].set_title('SENTIMENT SPLIT', fontsize=9, color='#c8d8e8', fontfamily='monospace', pad=10)

    axes[1].set_facecolor(CHART_PANEL)
    sns.histplot(df['sentiment_score'], bins=10, kde=True, color=CYAN, ax=axes[1],
                 alpha=0.3, line_kws={'color': CYAN, 'linewidth': 2})
    axes[1].set_title('CONFIDENCE DISTRIBUTION', fontsize=9, color='#c8d8e8', fontfamily='monospace')
    axes[1].set_xlabel('Confidence Score', color=CHART_TEXT, fontsize=8)
    axes[1].set_ylabel('Articles', color=CHART_TEXT, fontsize=8)
    axes[1].tick_params(colors=CHART_TEXT, labelsize=8)
    for spine in axes[1].spines.values(): spine.set_edgecolor(CHART_GRID)
    axes[1].grid(color=CHART_GRID, linewidth=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def chart_timeline(df: pd.DataFrame):
    daily         = df.copy()
    daily['date'] = daily['publishedAt'].dt.date
    timeline      = daily.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)
    for col in ['positive', 'negative', 'neutral']:
        if col not in timeline.columns:
            timeline[col] = 0

    fig, ax = plt.subplots(figsize=(11, 3.5))
    _apply_dark_style(fig, ax)

    ax.plot(timeline.index, timeline['positive'], marker='o', color=CYAN,    label='POSITIVE', linewidth=2, markersize=4)
    ax.plot(timeline.index, timeline['negative'], marker='o', color=CORAL,   label='NEGATIVE', linewidth=2, markersize=4)
    ax.plot(timeline.index, timeline['neutral'],  marker='o', color=MUTED,   label='NEUTRAL',  linewidth=2, markersize=4)
    ax.fill_between(timeline.index, timeline['positive'], alpha=0.08, color=CYAN)
    ax.fill_between(timeline.index, timeline['negative'], alpha=0.08, color=CORAL)
    ax.fill_between(timeline.index, timeline['neutral'],  alpha=0.06, color=MUTED)

    ax.set_title('SENTIMENT TREND OVER TIME', fontsize=9, color='#c8d8e8', fontfamily='monospace')
    ax.set_xlabel('Date', color=CHART_TEXT, fontsize=8)
    ax.set_ylabel('Articles', color=CHART_TEXT, fontsize=8)
    legend = ax.legend(facecolor=CHART_PANEL, edgecolor=CHART_GRID, labelcolor=CHART_TEXT, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=30, color=CHART_TEXT)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def chart_by_source(df: pd.DataFrame, topic: str):
    src = df.groupby(['source_name', 'sentiment_label']).size().unstack(fill_value=0)
    for col in ['negative', 'neutral', 'positive']:
        if col not in src.columns:
            src[col] = 0
    src = src[['negative', 'neutral', 'positive']]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    _apply_dark_style(fig, ax)

    src.plot(kind='bar', color=[CORAL, MUTED, CYAN], edgecolor=CHART_BG, ax=ax, width=0.65)
    ax.set_title(f'SENTIMENT BY SOURCE — {topic.upper()}', fontsize=9, color='#c8d8e8', fontfamily='monospace')
    ax.set_xlabel('Source', color=CHART_TEXT, fontsize=8)
    ax.set_ylabel('Articles', color=CHART_TEXT, fontsize=8)
    legend = ax.legend(facecolor=CHART_PANEL, edgecolor=CHART_GRID, labelcolor=CHART_TEXT, fontsize=8)
    plt.xticks(rotation=40, ha='right', color=CHART_TEXT, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def chart_radar(impact_data: dict):
    sectors     = list(impact_data['sectors'].keys())
    scores      = [impact_data['sectors'][s]['score'] for s in sectors]
    N           = len(sectors)
    angles      = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles      = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), facecolor=CHART_BG)
    ax.set_facecolor(CHART_PANEL)
    ax.plot(angles, scores_plot, 'o-', linewidth=2, color=CYAN)
    ax.fill(angles, scores_plot, alpha=0.15, color=CYAN)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(sectors, fontsize=7.5, color=CHART_TEXT, fontfamily='monospace')
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=6, color=CHART_TEXT)
    ax.grid(color=CHART_GRID, linewidth=0.8)
    ax.spines['polar'].set_color(CHART_GRID)
    ax.set_title('IMPACT BY SECTOR\n0 = none  ·  10 = major',
                 fontsize=8, color='#c8d8e8', fontfamily='monospace', pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─── Article Cards ─────────────────────────────────────────────────────────────
def render_articles(df: pd.DataFrame):
    for _, row in df.iterrows():
        badge = (f'<span class="badge-pos"> POSITIVE {row["sentiment_score"]:.0%}</span>'
                 if row['sentiment_label'] == 'positive'
                 else f'<span class="badge-neg"> NEGATIVE {row["sentiment_score"]:.0%}</span>'
                 if row['sentiment_label'] == 'negative'
                 else f'<span style="background:#f0f0f0;color:#555;border-radius:4px;padding:1px 7px;font-size:0.78rem;"> NEUTRAL {row["sentiment_score"]:.0%}</span>')
        date_str = row['publishedAt'].strftime('%b %d, %Y')
        with st.expander(f"{row['title']}"):
            st.markdown(
                f'<p class="article-meta">{row["source_name"]} · {date_str} · {badge}</p>',
                unsafe_allow_html=True
            )
            st.write(row['description'])
            st.markdown(f"[Read full article →]({row['url']})")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('### ⬡ AI SENTIMENT ANALYSER')
    st.divider()
    topic      = st.text_input('Search Topic', placeholder='e.g. War, Tesla, AI...')
    max_art    = st.slider('Number of Articles', 5, 20, 15, 5)
    run_button = st.button(' Analyze', use_container_width=True, type='primary')
    st.divider()

    if st.session_state.history:
        st.markdown('** Search History**')
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f" {h['topic']}", key=f'hist_{i}', use_container_width=True):
                    st.session_state.results = h
            with col2:
                st.caption(h['time'])

    st.divider()
    st.caption('Sentiment: FinBERT (news-trained)')
    st.caption('Summary: DistilBART')
    st.caption('Impact: Groq LLaMA 3.3 (free)')
    st.caption('News: NewsAPI')

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title"> ⬡ UNVEILING SENTIMENTS</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle"> ◈ deep learning nlp · real-time news decoding </p>', unsafe_allow_html=True)
st.divider()

# ─── Run Analysis ─────────────────────────────────────────────────────────────
if run_button:
    if not topic.strip():
        st.warning('Please enter a topic.')
        st.stop()

    steps = st.status('Analyzing...', expanded=True)
    with steps:
        st.write(' Fetching articles...')
        try:
            articles = fetch_articles(NEWS_API_KEY, topic.strip(), max_art)
        except RuntimeError as e:
            st.error(f'API Error: {e}')
            st.stop()
        except requests.exceptions.Timeout:
            st.error('Request timed out. Check your internet.')
            st.stop()
        except Exception as e:
            st.error(f'Unexpected error: {e}')
            st.stop()

        if not articles:
            st.warning('No articles found. Try a different topic.')
            st.stop()

        st.write(' Cleaning data...')
        df = clean_articles(articles)

        st.write(' Running sentiment analysis...')
        df = run_sentiment(df)

        st.write(' Generating summary...')
        summary = ''
        try:
            summary = run_summarization(df)
        except Exception as e:
            log.error('Summarization failed: %s', e)

        impact_data = None
        if GROQ_API_KEY and summary:
            st.write(' Analysing broader impact with Groq LLaMA...')
            pos_ratio   = (df['sentiment_label'] == 'positive').mean()
            impact_data = run_impact_analysis(topic.strip(), summary, pos_ratio)

    result = {
        'topic':       topic.strip(),
        'time':        datetime.now().strftime('%H:%M'),
        'df':          df,
        'summary':     summary,
        'impact_data': impact_data,
        'pos':         int((df['sentiment_label'] == 'positive').sum()),
        'neg':         int((df['sentiment_label'] == 'negative').sum()),
        'neu':         int((df['sentiment_label'] == 'neutral').sum()),
        'avg_score':   float(df['sentiment_score'].mean()),
    }
    st.session_state.results = result
    if not any(h['topic'].lower() == topic.strip().lower() for h in st.session_state.history):
        st.session_state.history.append(result)

# ─── Display Results ──────────────────────────────────────────────────────────
res = st.session_state.results
if res:
    df          = res['df']
    summary     = res['summary']
    impact_data = res['impact_data']
    rtopic      = res['topic']

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(' Articles',        len(df))
    c2.metric(' Positive',        int((df['sentiment_label'] == 'positive').sum()))
    c3.metric(' Negative',        int((df['sentiment_label'] == 'negative').sum()))
    c4.metric(' Neutral',         int((df['sentiment_label'] == 'neutral').sum()))
    c5.metric(' Avg Confidence',  f"{res['avg_score']:.0%}")
    st.divider()

    if summary:
        st.subheader(' Executive Summary')
        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
        st.divider()

    if impact_data:
        st.subheader(' Broader Impact Analysis')
        st.info(f"**{impact_data.get('headline_impact', '')}**")
        left, right = st.columns([1, 1])
        with left:
            chart_radar(impact_data)
        with right:
            st.markdown(f"**Most affected sector:** `{impact_data.get('most_affected', 'N/A')}`")
            st.markdown('')
            for sector, info in impact_data['sectors'].items():
                direction_icon = {'positive': '🟢', 'negative': '🔴', 'neutral': '⚪'}.get(info['direction'], '⚪')
                score_bar = '█' * info['score'] + '░' * (10 - info['score'])
                st.markdown(
                    f'<div class="impact-card">'
                    f'<b>{direction_icon} {sector}</b> &nbsp; <code>{score_bar}</code> {info["score"]}/10<br>'
                    f'{info["reason"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        st.divider()

    st.subheader(' Visual Insights')
    tab1, tab2, tab3 = st.tabs(['Sentiment Split', 'Trend Over Time', 'By Source'])
    with tab1:
        chart_sentiment_pie_and_hist(df, rtopic)
    with tab2:
        chart_timeline(df)
    with tab3:
        chart_by_source(df, rtopic)
    st.divider()

    st.subheader(' Articles')
    filter_col, _ = st.columns([2, 3])
    with filter_col:
        sentiment_filter = st.radio('Filter by sentiment',
                                    ['All', 'Positive only', 'Negative only', 'Neutral only'], horizontal=True)
    filtered = df.copy()
    if sentiment_filter == 'Positive only':
        filtered = df[df['sentiment_label'] == 'positive'].reset_index(drop=True)
    elif sentiment_filter == 'Negative only':
        filtered = df[df['sentiment_label'] == 'negative'].reset_index(drop=True)
    elif sentiment_filter == 'Neutral only':
        filtered = df[df['sentiment_label'] == 'neutral'].reset_index(drop=True)

    st.caption(f'Showing {len(filtered)} of {len(df)} articles')
    render_articles(filtered)
    st.divider()

    export = df.copy()
    export['summary']     = summary
    export['topic']       = rtopic
    export['exported_at'] = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    csv        = export.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    safe_topic = rtopic.replace(' ', '_').replace('/', '-')
    ts         = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

    st.download_button(
        '⬇ Download Full Results as CSV',
        data=csv,
        file_name=f'news_sentiment_{safe_topic}_{ts}.csv',
        mime='text/csv',
        use_container_width=True
    )
