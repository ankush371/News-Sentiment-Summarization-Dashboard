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
FETCH_TTL_SECONDS    = 300
SENTIMENT_MODEL      = 'ProsusAI/finbert'           # trained on financial/tech news
SUMMARIZATION_MODEL  = 'sshleifer/distilbart-cnn-12-6'
GROQ_MODEL           = 'llama-3.3-70b-versatile'
SENTIMENT_BATCH_SIZE = 8
SUMMARY_TOP_N        = 10
SUMMARY_MAX_LEN      = 180
SUMMARY_MIN_LEN      = 50
COLORS               = {'positive': '#66b3ff', 'negative': '#ff9999', 'neutral': '#d3d3d3'}
ACTION_ITEMS_LIMIT   = 4

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
def get_secret(name: str) -> str:
    # Prefer environment variables locally, then Streamlit secrets in cloud.
    value = os.environ.get(name, '').strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, '')).strip()
    except Exception:
        return ''


NEWS_API_KEY = get_secret('NEWS_API_KEY')
GROQ_API_KEY = get_secret('GROQ_API_KEY')

if not NEWS_API_KEY:
    st.error(' NEWS_API_KEY is not set. Add it in Streamlit Secrets or local environment variables.')
    st.stop()

if not GROQ_API_KEY:
    st.warning(' GROQ_API_KEY is not set — AI action guidance will be disabled. Add it in Streamlit Secrets or local environment variables.')

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


@st.cache_resource(show_spinner=False)
def get_http_session():
    """Reuse TCP connections for NewsAPI calls to reduce request overhead."""
    return requests.Session()

# ─── Core Functions ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=FETCH_TTL_SECONDS)
def fetch_articles(api_key: str, query: str, page_size: int) -> list:
    # Cached fetch avoids repeated API calls when users rerun the same query.
    params = {
        'q': query, 'language': 'en',
        'pageSize': page_size, 'sortBy': 'publishedAt', 'apiKey': api_key
    }
    session = get_http_session()
    r = session.get(NEWS_API_URL, params=params, timeout=REQUEST_TIMEOUT)
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
    # Remove near-duplicate stories syndicated across sources.
    out = out.drop_duplicates(subset=['title', 'source_name']).reset_index(drop=True)
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
    # Feed one compact context string from top stories for a stable summary.
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


def run_extractive_summary(df: pd.DataFrame) -> str:
    """Fast fallback summary for better responsiveness on CPU-only machines."""
    top = df.head(5)
    bullets = []
    for _, row in top.iterrows():
        sentiment = row['sentiment_label'].capitalize()
        bullets.append(f"- {row['title']} ({sentiment}, {row['sentiment_score']:.0%} confidence)")
    return '\n'.join(bullets)


def run_actionable_guidance(topic: str, summary: str, df: pd.DataFrame) -> dict | None:
    """Generate customer-facing actions rather than broad sector categorization."""
    if not GROQ_API_KEY:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        pos = int((df['sentiment_label'] == 'positive').sum())
        neg = int((df['sentiment_label'] == 'negative').sum())
        neu = int((df['sentiment_label'] == 'neutral').sum())

        prompt = f"""You are a pragmatic advisor for a business user.

Topic: {topic}
Summary: {summary}
Sentiment counts -> positive: {pos}, negative: {neg}, neutral: {neu}

Return ONLY valid JSON (no markdown), using this exact schema:
{{
    "executive_brief": "1-2 sentence plain-English takeaway for decision makers",
    "risk_level": "low|medium|high",
    "what_it_means": [
        "exactly 3 concise bullet points"
    ],
    "suggested_actions": [
        "3 to 4 concrete actions the customer can take in the next 24-72 hours"
    ]
}}

Rules:
- Keep language specific and practical.
- Avoid sector-level categorization.
- Suggested actions must be actionable and measurable."""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=550,
            temperature=0.3,
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if model adds them anyway
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]

        parsed = json.loads(raw.strip())
        # Defensive normalization keeps UI/export stable even with imperfect model output.
        if not isinstance(parsed.get('what_it_means'), list):
            parsed['what_it_means'] = []
        if not isinstance(parsed.get('suggested_actions'), list):
            parsed['suggested_actions'] = []
        parsed['what_it_means'] = parsed['what_it_means'][:3]
        parsed['suggested_actions'] = parsed['suggested_actions'][:ACTION_ITEMS_LIMIT]
        parsed['risk_level'] = str(parsed.get('risk_level', 'medium')).lower()
        if parsed['risk_level'] not in {'low', 'medium', 'high'}:
            parsed['risk_level'] = 'medium'
        return parsed
    except Exception as e:
        log.error('Action guidance failed: %s', e)
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
    topic      = st.text_input('Search Topic', placeholder='e.g. War,Petrol, AI...')
    max_art    = st.slider('Number of Articles', 5, 20, 15, 5)
    analysis_mode = st.selectbox('Analysis Mode', ['Fast', 'Deep'], help='Fast mode is quicker. Deep mode uses transformer summarization + AI guidance.')
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
    st.caption('Summary: Fast extractive or Deep DistilBART')
    st.caption('Guidance: Groq LLaMA 3.3 (optional)')
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
        if analysis_mode == 'Fast':
            # Fast mode skips transformer generation to reduce latency.
            summary = run_extractive_summary(df)
        else:
            try:
                summary = run_summarization(df)
            except Exception as e:
                log.error('Summarization failed: %s', e)
                # Always fall back so users still get a result.
                summary = run_extractive_summary(df)

        guidance_data = None
        if GROQ_API_KEY and summary:
            st.write(' Generating customer action guidance...')
            guidance_data = run_actionable_guidance(topic.strip(), summary, df)

    result = {
        'topic':       topic.strip(),
        'time':        datetime.now().strftime('%H:%M'),
        'df':          df,
        'summary':     summary,
        'guidance_data': guidance_data,
        'analysis_mode': analysis_mode,
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
    guidance_data = res.get('guidance_data')
    rtopic      = res['topic']

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(' Articles',        len(df))
    c2.metric(' Positive',        res['pos'])
    c3.metric(' Negative',        res['neg'])
    c4.metric(' Neutral',         res['neu'])
    c5.metric(' Avg Confidence',  f"{res['avg_score']:.0%}")
    st.divider()

    if summary:
        st.subheader(' Executive Summary')
        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
        st.divider()

    if guidance_data:
        st.subheader(' Customer Action Brief')
        risk_color = {'low': '#00d084', 'medium': '#f5c542', 'high': '#ff6b6b'}.get(guidance_data.get('risk_level', 'medium'), '#f5c542')
        st.info(f"**{guidance_data.get('executive_brief', '')}**")
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown('**What this means**')
            for item in guidance_data.get('what_it_means', []):
                st.markdown(f'- {item}')
        with right:
            st.markdown('**Risk Level**')
            st.markdown(
                f'<div style="padding:0.5rem 0.8rem;border-left:3px solid {risk_color};background:#0c1828;border-radius:0 6px 6px 0;text-transform:uppercase;letter-spacing:0.08em;">{guidance_data.get("risk_level", "medium")}</div>',
                unsafe_allow_html=True
            )
            st.markdown('')
            st.markdown('**Suggested actions (24-72h)**')
            for action in guidance_data.get('suggested_actions', []):
                st.markdown(f'- {action}')
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
    filter_col, source_col, sort_col = st.columns([2, 2, 2])
    with filter_col:
        sentiment_filter = st.radio('Filter by sentiment',
                                    ['All', 'Positive only', 'Negative only', 'Neutral only'], horizontal=True)
    with source_col:
        source_options = sorted(df['source_name'].dropna().unique().tolist())
        source_filter = st.multiselect('Filter by source', source_options, default=source_options)
    with sort_col:
        sort_by = st.selectbox('Sort by', ['Newest first', 'Highest confidence', 'Most positive', 'Most negative'])

    filtered = df.copy()
    if sentiment_filter == 'Positive only':
        filtered = df[df['sentiment_label'] == 'positive'].reset_index(drop=True)
    elif sentiment_filter == 'Negative only':
        filtered = df[df['sentiment_label'] == 'negative'].reset_index(drop=True)
    elif sentiment_filter == 'Neutral only':
        filtered = df[df['sentiment_label'] == 'neutral'].reset_index(drop=True)

    filtered = filtered[filtered['source_name'].isin(source_filter)].reset_index(drop=True)
    # Explicit rank avoids alphabetical ordering artifacts when sorting sentiments.
    sentiment_rank = {'negative': 0, 'neutral': 1, 'positive': 2}
    filtered['sentiment_rank'] = filtered['sentiment_label'].map(sentiment_rank).fillna(1)

    if sort_by == 'Highest confidence':
        filtered = filtered.sort_values('sentiment_score', ascending=False).reset_index(drop=True)
    elif sort_by == 'Most positive':
        filtered = filtered.sort_values(['sentiment_rank', 'sentiment_score'], ascending=[False, False]).reset_index(drop=True)
    elif sort_by == 'Most negative':
        filtered = filtered.sort_values(['sentiment_rank', 'sentiment_score'], ascending=[True, False]).reset_index(drop=True)
    else:
        filtered = filtered.sort_values('publishedAt', ascending=False).reset_index(drop=True)
    filtered = filtered.drop(columns=['sentiment_rank'])

    st.caption(f'Showing {len(filtered)} of {len(df)} articles')
    render_articles(filtered)
    st.divider()

    export = df.copy()
    export['summary']     = summary
    export['topic']       = rtopic
    export['analysis_mode'] = res.get('analysis_mode', 'Deep')
    if guidance_data:
        export['executive_brief'] = guidance_data.get('executive_brief', '')
        export['risk_level']      = guidance_data.get('risk_level', '')
        # Flatten list for CSV compatibility.
        export['suggested_actions'] = ' | '.join(guidance_data.get('suggested_actions', []))
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
