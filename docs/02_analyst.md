# Agent 2: The Analyst

## What It Does

The Analyst is the **scoring and intelligence** engine. It answers one question:
*"How likely is this company to buy IT/software/AI services in the next 6-12 months?"*

It scores each company on a 0-100 scale using a hybrid approach:
1. **Deterministic features** — computed in Python (revenue growth, CXO changes,
   concall recency, revenue band, employee size). These are ground truth.
2. **Qualitative scoring** — an LLM interprets signals, news, and announcements
   to score narrative dimensions (tech adoption, strategic moves, recency).

The key insight: **never let the LLM do arithmetic.** Python computes numbers;
the LLM interprets stories.

---

## Architecture

```
              ┌──────────────────────────────────────────────┐
              │              ANALYST AGENT                   │
              │           (agents/analyst.py)                │
              └───────────────────┬──────────────────────────┘
                                  │
      ┌───────────────────────────┼───────────────────────────┐
      │                           │                           │
┌─────▼──────────┐    ┌──────────▼──────────┐    ┌───────────▼──────────┐
│  PRE-FILTER    │    │ PYTHON FEATURES     │    │ LLM SCORING          │
│                │    │                     │    │                      │
│ is_scoreable() │    │ _compute_features() │    │ Bedrock / Qwen       │
│                │    │                     │    │                      │
│ name_resolution│    │ revenue_growth_4q   │    │ Receives:            │
│ failed?        │    │ has_cxo_change      │    │  - computed_features │
│   → score = 0  │    │ has_concall_90d     │    │  - tech_signals      │
│   → skip LLM   │    │ in_revenue_band     │    │  - announcements     │
│                │    │ employee_band       │    │  - hiring_news       │
│ Saves tokens   │    │                     │    │                      │
│ and time       │    │ Deterministic,      │    │ Returns:             │
│                │    │ no LLM needed       │    │  - score (0-100)     │
└────────────────┘    └─────────────────────┘    │  - top_signals       │
                                                  │  - reasoning         │
                                                  │  - risk_factors      │
                                                  │  - outreach_angle    │
                                                  └───────────┬──────────┘
                                                              │
                                               ┌──────────────▼──────────┐
                                               │  MERGE & RECOVERY       │
                                               │                        │
                                               │ Merge scores by idx    │
                                               │ Fallback: name match   │
                                               │ Stub for missed ones   │
                                               │                        │
                                               │ AR PDF recovery        │
                                               │ (Tavily for failed)    │
                                               │                        │
                                               │ Global leaderboard     │
                                               └────────────────────────┘
```

---

## The Two-Pass Scoring Design

### Why not let the LLM do everything?

LLMs are bad at:
- Arithmetic ("Is 1248 > 1147? By what percent?" → often wrong)
- Date math ("Is March 31, 2026 within the last 180 days?" → unreliable)
- Consistent scoring ("Score this 75" then "Score that 72" → drift)

LLMs are good at:
- Interpreting signals ("AI-powered fleet tracking" → tech adoption)
- Reading between lines ("CEO cessation + new CEO designate" → leadership change)
- Narrative synthesis ("Revenue growing + CXO change + digital investment → strong buy signal")

So we split the work:

```
 Pass 1: PYTHON (deterministic)        Pass 2: LLM (qualitative)
 ───────────────────────────────        ─────────────────────────────
 revenue_growth_pct_4q = 8.87%         "Growing at 8.87% with new CEO
 has_recent_cxo_change = true            and AI fleet tracking signals
 has_concall_last_90d = false            strong tech modernization.
 in_revenue_target_band = false          Risk: above target revenue band."
 employee_band = "1k-5k"
                                        score: 72
                                        top_signals: ["AI adoption", ...]
```

### Pass 1: Computed Features

```python
def _compute_features(company: dict) -> dict:
    financials = company.get("financials") or {}
    documents  = company.get("documents") or {}
    profile    = company.get("profile") or {}

    # 1. Revenue growth (first quarter vs last quarter)
    quarters = financials.get("revenue_quarters") or []
    growth_pct = None
    if len(quarters) >= 2:
        latest   = float(quarters[0]["revenue_crore"])   # most recent
        earliest = float(quarters[-1]["revenue_crore"])   # oldest
        if earliest > 0:
            growth_pct = ((latest - earliest) / earliest) * 100

    # 2. CXO change in last 180 days
    cutoff = datetime.now() - timedelta(days=180)
    has_cxo = False
    for ann in documents.get("announcements") or []:
        text = f"{ann.get('title','')} {ann.get('context','')}"
        if any(kw in text for kw in CXO_KEYWORDS):
            dt = _parse_screener_date(ann.get("date"))
            if dt and dt >= cutoff:
                has_cxo = True
                break

    # 3. Recent concall (last 90 days)
    # 4. Revenue in target band (500-2000 Cr)
    # 5. Employee band (<1k / 1k-5k / 5k-10k / 10k-50k / >50k)

    return {
        "revenue_growth_pct_4q": growth_pct,
        "has_recent_cxo_change": has_cxo,
        "has_concall_last_90d":  has_concall,
        "in_revenue_target_band": in_band,
        "employee_band":          band,
    }
```

These features are attached to the company dict **before** the LLM sees it.
The LLM prompt says: *"computed_features are ground truth. Do NOT recompute."*

### Pass 2: LLM Scoring

The LLM receives a **slim payload** — not the full enriched record:

```python
def _build_prompt_payload(companies: list) -> str:
    """Only sends what the LLM needs. Does NOT mutate the original dicts."""
    slim = []
    for i, c in enumerate(companies):
        slim.append({
            "idx": i,                                    # for merge-back
            "name": c.get("name"),
            "ticker": c["identifiers"]["yahoo_ticker"],
            "industry": c["profile"]["industry"],
            "computed_features": c["computed_features"],  # ground truth
            "revenue_quarters": fin["revenue_quarters"][:4],
            "tech_signals": seed["tech_signals"][:4],     # LLM seed
            "recent_news": seed["recent_news"][:3],
            "recent_announcements": docs["announcements"][:5],
            "hiring_news": news["hiring"][:2],
            "description": profile["description"][:160],
        })
    return json.dumps(slim)
```

The LLM scoring prompt:

```
Score these 3 "travel logistics" companies on IT/software buy-likelihood.

computed_features are pre-calculated Python ground truth — do NOT recompute,
use them directly when reasoning about growth, CXO changes, concalls, revenue band.

DATA: [{"idx":0,"name":"TCI Ltd","computed_features":{...},...}, ...]

Return JSON: {"sector":"...","sector_summary":"...",
  "scored_companies":[{"idx":0,"name":"...","score":85,"growth_trend":"growing",
  "top_signals":[...],"reasoning":"...","recommended_approach":"...",
  "outreach_angle":"...","risk_factors":[...]}]}

IMPORTANT: include the 'idx' field from the input so scores can be matched back.
```

### Scoring Rubric

The LLM uses this rubric (embedded in the system prompt):

```
 Category           Weight   What the LLM evaluates
 ─────────────────  ──────   ────────────────────────────────────────────
 Revenue Growth      35 pts   computed_features.revenue_growth_pct_4q
                              + growth_trend interpretation
 Tech Signals        30 pts   LLM seed tech_signals + hiring_news
                              + announcement context
 Strategic Moves     20 pts   CXO changes + concall recency
                              + announcements (expansion, M&A, etc.)
 Recency             15 pts   How recent are the signals?
                              Fresh signals → higher score
```

---

## Pre-Filtering: Scoreable vs Unscoreable

Not all companies from the researcher are real. Some are:
- **Unlisted** (e.g., Nippon Express India — Japanese subsidiary, not on BSE/NSE)
- **Delisted** (e.g., Gati Limited — merged into Allcargo, no longer traded)

The researcher correctly flags these with `provenance.failed_stages = ["name_resolution"]`.

```python
def _is_scoreable(company: dict) -> bool:
    failed = company.get("provenance", {}).get("failed_stages", [])
    return "name_resolution" not in failed
```

Unscoreable companies get `score = 0` and a clear reason:

```
analyst: {
  score: 0,
  reasoning: "Company not found on BSE/NSE — likely unlisted or delisted.
              Failed stages: name_resolution, yfinance, screener",
  risk_factors: ["not_listed"],
}
```

**Why this matters:** Without pre-filtering, the LLM would try to score companies
with zero data and produce confident-sounding nonsense. Pre-filtering saves
tokens AND prevents garbage scores.

---

## Score Merging: The idx Strategy

### The Problem

The LLM might return a different name than what we sent:

```
Sent:     "Transport Corporation of India Limited"
Returned: "Transport Corporation of India"        ← missing "Limited"
```

Exact name matching fails. Fuzzy matching is expensive and error-prone.

### The Solution

We send an `idx` field with each company and ask the LLM to echo it back:

```
Sent:     {"idx": 0, "name": "Transport Corporation of India Limited", ...}
Returned: {"idx": 0, "name": "Transport Corp of India", "score": 72, ...}
                ↑
                Match by idx (foolproof), ignore the name
```

```python
# Merge: try idx first, fall back to name
for sc in scored.get("scored_companies") or []:
    idx = sc.get("idx")
    target = None
    if isinstance(idx, int) and 0 <= idx < len(scoreable):
        target = scoreable[idx]           # primary: idx match
    if target is None:
        key = sc.get("name", "").strip().lower()
        target = by_name.get(key)         # fallback: name match
    if target is None:
        continue
    target["analyst"] = { score, top_signals, reasoning, ... }
```

---

## Annual Report Recovery

Some companies resolve on yfinance but don't have annual reports on screener.in.
The researcher marks them with `failed_stages = ["annual_report_pdf"]`.

The analyst does a **recovery pass** after scoring:

```
 For each SCOREABLE company with "annual_report_pdf" in failed_stages:
   │
   ├─ Tavily search: "{name} annual report 2024 2025 filetype:pdf"
   │   └─ First .pdf URL from results
   │
   ├─ CompanyDataFetcher.extract_annual_report_pdf(pdf_url)
   │   └─ Returns text excerpt or None
   │
   └─ On success:
       ├─ company["documents"]["annual_report_excerpt"] = excerpt
       └─ Remove "annual_report_pdf" from failed_stages
```

This is the **only Tavily call the analyst makes**. It runs only for the ~20% of
companies where screener didn't have the PDF — not all companies.

---

## Output Schema

The analyst adds an `analyst` block and `computed_features` block to each company.
The rest of the unified schema passes through untouched.

```
company (after analyst):
  ├── name, exchange, sector_query          ← unchanged
  ├── identifiers                           ← unchanged
  ├── profile                               ← unchanged
  ├── financials                            ← unchanged
  ├── documents                             ← may gain annual_report_excerpt via recovery
  ├── news                                  ← unchanged
  ├── llm_seed                              ← unchanged
  ├── provenance                            ← failed_stages may shrink via recovery
  │
  ├── computed_features  ← NEW (Python-computed)
  │     revenue_growth_pct_4q: 8.87
  │     has_recent_cxo_change: true
  │     has_concall_last_90d: false
  │     in_revenue_target_band: false
  │     employee_band: "1k-5k"
  │
  └── analyst  ← NEW (LLM-scored)
        score: 72
        global_rank: 3
        growth_trend: "growing"
        top_signals: ["AI fleet tracking", "new CEO designate"]
        reasoning: "Revenue growing at 8.87% with..."
        recommended_approach: "Reference their AI fleet tracking..."
        outreach_angle: "Logistics tech modernization"
        risk_factors: ["above target revenue band"]
        scored_at: "2026-04-12T10:50:00"
```

### The Global Leaderboard

After all sectors are scored, companies are sorted globally:

```python
all_companies.sort(
    key=lambda x: x.get("analyst", {}).get("score", 0),
    reverse=True
)
for i, c in enumerate(all_companies, 1):
    c["analyst"]["global_rank"] = i
```

The leaderboard is the reporter's input — top N companies get email drafts.

---

## Error Handling

```
 Failure                                    Behavior
 ────────────────────────────────────────   ──────────────────────────────────
 name_resolution in failed_stages           score = 0, skip LLM call entirely
 LLM JSON parse failure                     scored = {}, all companies get stubs
 LLM omits a company from response          stub analyst block (score=0)
 LLM returns wrong idx                      fallback to name matching
 Tavily AR recovery fails                   continue, company keeps failed stage
 Bedrock throttling / timeout               uncaught → propagates up
 Screener date unparseable                  _parse_screener_date returns None
```

---

## Token Efficiency

```
 Without pre-filter:    5 companies × 1 prompt = 1 LLM call with 5 records
 With pre-filter:       3 scoreable  × 1 prompt = 1 LLM call with 3 records
                        2 skipped                = 0 LLM calls
                                                   ──────────
                                                   40% token savings
```

For a full run with 8 sectors × 15 companies, if 30% are unresolvable:
- Without filter: 120 companies → 8 LLM calls (15 records each)
- With filter: 84 scoreable → 8 LLM calls (10.5 records each) → ~30% cheaper

---

## How to Extend

### Changing the scoring rubric

Edit the system prompt in `create_analyst_agent()`:
```python
system_prompt = (
    "Scoring (100 pts): Revenue Growth 35, Tech Signals 30, "
    "Strategic Moves 20, Recency 15.\n"
    # Change the weights or add new dimensions here
)
```

### Adding a new computed feature

1. Add the computation to `_compute_features()`:
```python
# Example: check if company has recent credit rating upgrade
has_rating_upgrade = False
for r in documents.get("credit_ratings") or []:
    if "upgrade" in (r.get("title") or "").lower():
        has_rating_upgrade = True
        break
```

2. Include it in the return dict
3. It automatically flows to the LLM payload via `computed_features`

### Changing the scoreability threshold

Currently: company is scoreable if `name_resolution` succeeded.
You could make it stricter:

```python
def _is_scoreable(company: dict) -> bool:
    failed = company.get("provenance", {}).get("failed_stages", [])
    # Require both ticker resolution AND yfinance data
    return "name_resolution" not in failed and "yfinance" not in failed
```
