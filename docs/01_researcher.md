# Agent 1: The Researcher

## What It Does

The Researcher is the **discovery and enrichment** engine. It answers one question:
*"Which Indian listed companies in sector X have revenue in the 500-2000 Cr range
and might buy IT/software services?"*

It does this in two phases:
1. **Discovery** — Use Tavily web search + an LLM to identify company names
2. **Enrichment** — For each company, replace the LLM's guesses with real data
   from yfinance, screener.in, Tavily hiring news, and annual report PDFs

The output is a fully structured, multi-source enriched company record.

---

## Architecture

```
                         ┌──────────────────────────────────┐
                         │         RESEARCHER AGENT         │
                         │         (agents/researcher.py)   │
                         └──────────┬───────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────────┐
              │                     │                         │
     ┌────────▼────────┐  ┌────────▼────────┐  ┌─────────────▼──────────┐
     │ SECTOR DISCOVERY │  │  PER-COMPANY    │  │  OUTPUT & STORAGE      │
     │                  │  │  ENRICHMENT     │  │                        │
     │ tavily x2/sector │  │                 │  │ Unified schema dict    │
     │ LLM call x1      │  │ 5 data sources  │  │ S3 upload (research)   │
     │ JSON parse        │  │ per company     │  │ S3 upload (run-based)  │
     └──────────────────┘  └────────┬────────┘  └────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
           ┌────────▼──┐    ┌──────▼──────┐  ┌─────▼──────┐
           │ yfinance   │    │ screener.in │  │ Tavily     │
           │            │    │             │  │ (hiring)   │
           │ Ticker     │    │ BSE/NSE     │  │            │
           │ resolve    │    │ documents   │  │ 1 query    │
           │ Financials │    │ section     │  │ per company│
           │ Quarterly  │    │             │  │            │
           │ revenue    │    │ announce-   │  │ 4 results  │
           │            │    │ ments       │  │ max        │
           │ 16 fields  │    │ annual rpts │  │            │
           │ from .info │    │ concalls    │  └────────────┘
           │            │    │ credit rtgs │
           └────────────┘    └─────────────┘
                                    │
                             ┌──────▼──────┐
                             │ PDF Extract │
                             │ (annual rpt)│
                             │ pdfplumber  │
                             │ pages 8-19  │
                             └─────────────┘
```

---

## Phase 1: Sector Discovery

### Theory

The LLM is good at knowing *which companies exist* in a sector, but terrible at
knowing their exact ticker symbols or precise revenue figures. So we use it only
for discovery — "give me names" — and verify everything with real data sources.

We feed the LLM two Tavily web search results to ground it:
- **q1** (financial): Companies in sector with revenue 500-2000 Cr
- **q2** (tech): Companies adopting digital transformation / AI

Both searches are wrapped in try/except so a Tavily 429 doesn't kill the sector.

### Code Flow

```python
def research_sector(sector: str) -> dict:
    # 1. Web search for grounding
    q1 = f"Indian listed BSE NSE {sector} revenue 500-2000 crore 2024 2025"
    q2 = f"Indian {sector} digital transformation AI technology 2024 2025"

    try:
        r1 = tavily_search(q1, 6)          # 6 results for financial context
    except Exception:
        r1 = []                            # graceful fallback
    try:
        r2 = tavily_search(q2, 4)          # 4 results for tech context
    except Exception:
        r2 = []

    # 2. Compact search results to stay under token budget
    search_text = _compact_search(r1, config.MAX_SEARCH_CHARS)      # 4000 chars
    tech_text   = _compact_search(r2, config.MAX_SEARCH_CHARS // 2) # 2000 chars

    # 3. LLM call — returns JSON with company names + basic metadata
    prompt = f'List EXACTLY {config.MAX_COMPANIES_PER_SECTOR} companies...'
    response = _call_agent(agent, prompt, sector)

    # 4. Parse JSON (graceful on failure)
    result = json.loads(response_text[start:end])

    # 5. Enrich each company with real data
    for company in result["companies"]:
        _enrich_company(company, sector_query=sector)
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| Two Tavily calls per sector (not per company) | Cost control: 2 calls vs 30+ |
| `_compact_search` truncates to 4000 chars | Token budget: LLM input costs money |
| `try/except` around each Tavily call | One API failure shouldn't kill a whole sector |
| LLM told to return "EXACTLY N" companies | Without this, LLMs return 1-2 lazily |
| JSON-only system prompt, no markdown fences | Reduces parse failures from ```json wrappers |

---

## Phase 2: Company Enrichment

### Theory

The LLM gives us a name like "Transport Corporation of India Ltd" with a guessed
ticker like "TCIL" and hallucinated revenue figures. Enrichment replaces *every*
guess with authoritative data from 5 sources:

```
LLM Output (unreliable)          Enriched Output (authoritative)
─────────────────────────         ─────────────────────────────────
ticker: "TCIL"            ──→    yahoo_ticker: "TCI.BO"  (yfinance)
revenue: "₹1200 Cr"       ──→    revenue_ttm_crore: 4790.68  (yfinance)
description: "..."         ──→    long_business_summary: "..."  (yfinance)
                           ──→    announcements: [{...}]  (screener.in)
                           ──→    hiring: [{...}]  (Tavily)
                           ──→    annual_report_excerpt: "..."  (PDF)
```

### The Name Resolution Problem

Yahoo Finance doesn't find "Shipping Corporation of India Ltd" — it needs
"Shipping Corporation of India" (without "Ltd"). The LLM always adds formal
suffixes. Solution: **progressive name shortening**.

```python
def _build_search_queries(name: str) -> list[str]:
    """
    Input:  "Shipping Corporation of India Ltd"
    Output: [
      "Shipping Corporation of India Ltd",     # try full name first
      "Shipping Corporation of India",          # strip Ltd/Limited
      "Shipping Corporation of",                # first 3 words
      "Shipping Corporation",                   # first 2 words
    ]
    """
    queries = [name]
    stripped = name
    for token in ("Limited", "Ltd.", "Ltd", "Pvt.", "Pvt"):
        stripped = stripped.replace(token, "")
    stripped = stripped.strip(" ,.")
    if stripped != name:
        queries.append(stripped)
    words = stripped.split()
    if len(words) > 3: queries.append(" ".join(words[:3]))
    if len(words) > 2: queries.append(" ".join(words[:2]))
    return queries   # de-duped
```

Each query is tried in order. First one that returns `.NS` or `.BO` results wins.

### The Name Similarity Guard

Even when yfinance returns a result, it might be the wrong company. "ABB India"
could resolve to "ABB Ltd Switzerland". We use `difflib.SequenceMatcher`:

```
"shipping corporation of india ltd"  vs  "The Shipping Corporation of India Limited"
                                          → confidence = 0.89  ✓ ACCEPT

"abb india limited"                  vs  "ABB Ltd"
                                          → confidence = 0.42  ✗ REJECT (< 0.6)
```

If no quote has a `longname` field (rare), we accept with confidence 0.5 ("blind match").

### The 5 Enrichment Stages

```
 _enrich_company(company, sector_query)
 │
 ├─ Stage 1: TICKER RESOLUTION
 │   yf.Search(name) → resolved ticker (e.g. "TCI.BO")
 │   + SequenceMatcher confidence check
 │   ↓ failed? → failed_stages.append("name_resolution"), skip stages 2-3
 │
 ├─ Stage 2: YFINANCE FINANCIALS
 │   yf.Ticker(ticker).info → 16 fields (revenue, margins, employees, etc.)
 │   yf.Ticker(ticker).quarterly_financials → last 4 quarters
 │   ↓ These replace the LLM's hallucinated revenue_quarters
 │
 ├─ Stage 3: SCREENER.IN DOCUMENTS
 │   HTTP GET screener.in/company/{symbol}/
 │   BeautifulSoup → 4 document lists:
 │     announcements (with CXO context!)
 │     annual_reports (newest-first, with PDF URLs)
 │     concalls (transcripts + PPTs)
 │     credit_ratings (CRISIL, ICRA, CARE)
 │
 ├─ Stage 4: TAVILY HIRING NEWS
 │   1 query: '"{name}" hiring layoffs headcount India 2024 2025'
 │   4 results max, 0.4s sleep
 │   ↓ This is the ONLY Tavily call per company. Everything else uses
 │     screener (announcements, financials, annual reports) or yfinance.
 │
 └─ Stage 5: ANNUAL REPORT PDF
     Stage 5a: Look for FY24/FY25 in screener's annual_reports list
     Stage 5b: Fall back to annual_reports[0]
     Extract pages 8-19 (MD&A section, skip cover/index)
     3000 char cap
```

### Unified Schema Output

After enrichment, the company dict is **cleared and rebuilt** into the unified schema:

```python
company.clear()
company["name"]        = name                    # "Transport Corporation of India Limited"
company["exchange"]    = exchange_hint            # "BSE"
company["sector_query"]= sector_query            # "travel logistics and supply chain"
company["identifiers"] = {
    "yahoo_ticker": "TCI.BO",
    "screener_symbol": "TCI",
    "name_match_confidence": 1.0,
}
company["profile"]     = { description, long_business_summary, website, employees, industry }
company["financials"]  = { revenue_ttm_crore, market_cap_crore, margins, growth,
                           revenue_quarters: [{quarter, revenue_crore, source}] }
company["documents"]   = { screener_url, announcements, annual_reports, concalls,
                           credit_ratings, annual_report_excerpt }
company["news"]        = { hiring: [{title, url, content}] }
company["llm_seed"]    = { tech_signals, recent_news, key_quotes, ticker_guess }
company["provenance"]  = { researched_at, failed_stages: [], errors: [] }
```

**Rule: nothing from the LLM's original output survives at the top level.** It's
all moved to `llm_seed` for the analyst to use as qualitative context. The real
data lives in `financials`, `profile`, `documents`.

---

## Failure Handling

The researcher is designed to **never crash on a single company failure**.

```
                     Failure Type                         Behavior
 ─────────────────────────────────────────   ────────────────────────────────
 Tavily API key missing                      ValueError → pipeline crashes (correct)
 Tavily 429/500 on sector search             try/except → r1=[], sector continues
 Tavily 429/500 on hiring search             try/except per topic → empty list
 yfinance.Search returns no Indian stocks    (None, 0.0, None) → skip yfinance
 yfinance name match < 0.6                   rejected → skip yfinance + screener
 yfinance.Ticker.info throws                 failed_stages.append("yfinance")
 Screener HTTP error                         failed_stages.append("screener")
 PDF download > 30MB                         return None (skip)
 PDF Content-Type not pdf                    return None (skip)
 PDF page extraction fails                   per-page try/except, continue
 LLM returns invalid JSON                    empty companies[], sector continues
 S3 upload fails                             raises → pipeline stops (intentional)
```

Every failure is tracked in `provenance.failed_stages` so the analyst knows what
data is missing per company.

---

## Cost Analysis

For 1 sector with 5 companies:

```
 Source             Calls    Cost/call    Total
 ──────────────    ──────   ──────────   ──────
 Tavily (sector)     2      ~$0.005     $0.01
 Tavily (hiring)     5      ~$0.005     $0.025
 LLM (Bedrock)       1      ~$0.02      $0.02
 yfinance            5       free        $0
 screener.in         5       free        $0
 PDF download        5       free        $0
 ──────────────────────────────────────────────
 Total per sector                       ~$0.055
 Full run (8 sectors × 15 companies)    ~$1.50
```

The design intentionally shifted work from Tavily (paid) to screener.in (free).
The original design used 5 Tavily topics per company (600 calls/run). The new
design uses 1 topic (120 calls/run) — an 80% cost reduction.

---

## How to Extend

### Adding a new enrichment source

1. Add a fetcher method to `CompanyDataFetcher` in `tools/company_scraper.py`
2. Call it from `_enrich_company()` at the appropriate stage
3. Map the output to a field in the unified schema
4. Add the stage name to `failed_stages` on failure

### Adding a new sector

Edit `config.py`:
```python
SECTORS = [
    "travel logistics and supply chain",
    "healthcare and pharma",
    "your new sector here",    # ← add here
]
```

### Changing the revenue target band

```python
REVENUE_MIN_CRORE = 500    # lower bound
REVENUE_MAX_CRORE = 2000   # upper bound
```

This affects the LLM prompt and `computed_features.in_revenue_target_band`.
