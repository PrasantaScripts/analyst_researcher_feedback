# BRD — Project Guide for Claude Code

## Project Purpose

BRD is a 3-agent AI pipeline that discovers, enriches, and scores Indian listed companies
(BSE/NSE) as B2B IT/software sales prospects. A single `python main.py` run produces a
filtered, scored HTML dashboard of ~64 companies with personalized outreach email drafts,
stored under a timestamped run ID in S3.

---

## How to Run

```bash
# Run the full pipeline (research + analysis + reporting)
python main.py

# Re-run a single agent against the latest S3 run (for debugging)
python agents/analyst.py
python agents/reporter.py
```

---

## Environment Setup

Required `.env` variables:

```
BEDROCK_MODEL_ID=qwen.qwen3-coder-30b-a3b-v1:0   # Bedrock model for all agents
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
TAVILY_API_KEY=...
S3_BUCKET=prospect-ai-data
```

The pipeline was tuned for the Qwen3 Coder model's JSON compliance. Switching models may
require prompt adjustments.

---

## Architecture

```
main.py — run_pipeline()
  |
  +-- Agent 1: agents/researcher.py — run_researcher()
  |     Uses: tools/company_scraper.py, Tavily API, yfinance, screener.in
  |     Output: all_sectors.json (unified company schema per sector)
  |
  +-- Agent 2: agents/analyst.py — run_analyst()
  |     Uses: agents/researcher.tavily_search(), tools/company_scraper.py
  |     Output: analyst_output.json (scored global leaderboard)
  |
  +-- Agent 3: agents/reporter.py — run_reporter()
        Uses: templates/report_template.html
        Output: report.html (self-contained filterable dashboard)

Supporting modules:
  config.py           — all tunable constants
  utils/logger.py     — log() / warn() / error() wrappers
  utils/s3_storage.py — S3 read/write helpers + run-ID management
  utils/token_tracker.py — per-call token usage tracking and cost estimation
```

---

## Unified Company Schema

Every agent reads and writes the same company dict shape. Each agent adds its own namespace:

| Namespace          | Written by   | Contains                                              |
|--------------------|--------------|-------------------------------------------------------|
| `identifiers`      | Researcher   | yahoo_ticker, screener_symbol, name_match_confidence  |
| `profile`          | Researcher   | description, long_business_summary, website, employees|
| `financials`       | Researcher   | revenue_ttm_crore, market_cap_crore, revenue_quarters |
| `documents`        | Researcher   | announcements, annual_reports, concalls, credit_ratings, annual_report_excerpt |
| `news`             | Researcher   | hiring: [{title, url, content}]                       |
| `llm_seed`         | Researcher   | tech_signals, recent_news, key_quotes, ticker_guess   |
| `provenance`       | Researcher   | researched_at, failed_stages[], errors[]              |
| `computed_features`| Analyst      | revenue_growth_pct_4q, has_recent_cxo_change, etc.   |
| `analyst`          | Analyst      | score, top_signals, reasoning, global_rank            |
| `outreach`         | Reporter     | email_subject, email_body, drafted_at                 |

No agent strips another's namespace. The full record passes forward at each stage.

---

## Coding Conventions

### Naming
- **Classes**: `PascalCase` — `CompanyDataFetcher`, `TokenTracker`, `CallRecord`
- **Functions and variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE` — always defined in `config.py`, never hard-coded in agent files
- **Local variables**: descriptive names — no single-letter or abbreviated names
  - Exception: loop indices (`i`, `c`) where the type is obvious from context
  - Exception: `s` in `_parse_screener_date()` — stripping a date string

### Function size
Keep functions under 40 lines. Extract helpers with names that describe their responsibility
(what they return or what they change, not their mechanics).

### Comments
Write "why", not "what". One-line `# reason for this design choice` only.
Architecture decisions live in `docs/` — code comments add only what the docs don't cover.

### Error handling
Every external call (Tavily, yfinance, screener.in, PDF download) is wrapped in `try/except`.
Failures are recorded in `provenance.failed_stages` and logged. The pipeline continues.

### Single Responsibility
Each function does one thing. If a function calls an API AND parses the response AND merges
results, split it into three functions.

---

## Key Config Values

| Constant                         | Default | Effect                                         |
|----------------------------------|---------|------------------------------------------------|
| `SECTORS`                        | 8 sectors | Which markets to research                    |
| `REVENUE_MIN_CRORE`              | 500     | Lower bound of prospect revenue filter         |
| `REVENUE_MAX_CRORE`              | 2000    | Upper bound of prospect revenue filter         |
| `MAX_COMPANIES_PER_SECTOR`       | 8       | How many companies the LLM discovers           |
| `MAX_SEARCH_CHARS`               | 4000    | Tavily text budget per sector (~1k tokens)     |
| `NAME_MATCH_CONFIDENCE_THRESHOLD`| 0.6     | Below this, yfinance result is rejected        |
| `BLIND_MATCH_CONFIDENCE`         | 0.5     | Assigned when yfinance returns no longname     |
| `CXO_CHANGE_WINDOW_DAYS`         | 180     | Lookback for CXO change signals                |
| `CONCALL_RECENCY_DAYS`           | 90      | Lookback for concall signals                   |
| `PDF_MDNA_START_PAGE`            | 8       | First page of MD&A in annual report PDFs       |
| `PDF_MDNA_END_PAGE`              | 20      | Last page (exclusive) of MD&A section          |
| `PDF_FALLBACK_PAGES`             | 5       | Pages to extract when PDF is short             |
| `TOP_COMPANIES_FOR_REPORT`       | 100     | Companies that receive email drafts            |

---

## Adding New Features

### New sector
Add a string to the `SECTORS` list in `config.py`. No other changes needed.

### New enrichment source
1. Add a method to `CompanyDataFetcher` in `tools/company_scraper.py`
2. Call it from `_enrich_company()` in `agents/researcher.py` (Stage 4 or as a new Stage 7)
3. Write results into the company dict under an appropriate namespace key

### New computed feature
Add the computation to `_compute_features()` in `agents/analyst.py`.
The new key flows automatically into the LLM prompt payload via `_build_prompt_payload()`.

---

## S3 Artifact Layout

```
s3://{S3_BUCKET}/runs/{run_id}/
    research.json   — all sectors with enriched company records
    analysis.json   — scored global leaderboard
    report.html     — self-contained filterable dashboard
    metrics.json    — token usage, timing, cost breakdown
```

Retrieve programmatically:
```python
from utils.s3_storage import get_run_artifacts, list_runs
runs = list_runs()          # newest-first list of run IDs
data = get_run_artifacts(runs[0])  # {'research': {...}, 'analysis': {...}, ...}
```

---

## Failure Modes and Debugging

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: S3 preflight failed` | Bad credentials or wrong bucket | Check `.env` AWS keys and `S3_BUCKET` |
| `ValueError: TAVILY_API_KEY not set` | Missing key | Add `TAVILY_API_KEY` to `.env` |
| Company `score: 0`, reasoning "not found on BSE/NSE" | Unlisted or delisted company | Expected — `provenance.failed_stages` has `name_resolution` |
| LLM JSON parse failed (logged) | Model returned markdown fences or extra text | Check `logs/run_log.txt` for raw response; adjust system prompt |
| `annual_report_pdf` in `failed_stages` | Screener had no PDF, Tavily recovery also failed | Analyst will retry via Tavily in `_recover_annual_reports()` |

Logs: `logs/run_log.txt` (timestamped), token usage: `logs/token_usage.json`
