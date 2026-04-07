# News Sentiment Dashboard

A Streamlit app that fetches live news articles for a topic, runs sentiment analysis with FinBERT, generates summaries, and provides optional AI-driven action guidance.

## Features

- Fetches recent English-language news from NewsAPI.
- Cleans and deduplicates article data.
- Runs sentiment analysis with `ProsusAI/finbert`.
- Supports two summary modes:
  - `Fast`: extractive bullet-style summary.
  - `Deep`: transformer summary with DistilBART.
- Optional customer action brief using Groq (Llama 3.3 70B).
- Interactive charts (sentiment split, timeline, by source).
- Article filtering, sorting, and CSV export.

## Tech Stack

- Python 3.12 (see `runtime.txt`)
- Streamlit
- PyTorch (CPU build)
- Transformers
- Pandas, NumPy
- Matplotlib, Seaborn
- NewsAPI + Groq API

## Project Structure

- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies
- `runtime.txt` - runtime Python version

## Prerequisites

- Python 3.12 recommended
- A NewsAPI key (required): https://newsapi.org/
- A Groq API key (optional for action guidance): https://console.groq.com/

## Setup

1. Create and activate a virtual environment.

   Windows (PowerShell):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:

   ```env
   NEWS_API_KEY=your_newsapi_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Notes:
   - `NEWS_API_KEY` is required.
   - `GROQ_API_KEY` is optional. If omitted, AI action guidance is disabled.

## Run

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. Enter a topic in the sidebar.
2. Select article count and analysis mode (`Fast` or `Deep`).
3. Click `Analyze`.
4. Explore metrics, charts, summaries, and article cards.
5. Download full results as CSV.

## Notes

- The app is optimized for CPU inference by using a CPU PyTorch wheel.
- First run may take longer while transformer models download and cache.
- NewsAPI free tier limits and rate limits may affect results.
