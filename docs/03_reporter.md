# Agent 3: The Reporter

## What It Does

The Reporter is the **presentation and outreach** engine. It does two things:
1. **Email generation** — One batched LLM call writes personalized B2B outreach
   emails for the top N companies
2. **Dashboard rendering** — Injects the full enriched data into an HTML template
   that becomes a self-contained, filterable, exportable sales dashboard

The key design: **the LLM writes only emails.** The HTML is pre-built — no LLM
tokens wasted on markup, styling, or table generation.

---

## Architecture

```
              ┌──────────────────────────────────────────────┐
              │              REPORTER AGENT                  │
              │           (agents/reporter.py)               │
              └───────────────────┬──────────────────────────┘
                                  │
          analyst_output          │
          (global_leaderboard)    │
                                  │
              ┌───────────────────▼───────────────────────────┐
              │  Step 1: Extract top N companies              │
              │  (config.TOP_COMPANIES_FOR_REPORT = 15)       │
              └───────────────────┬───────────────────────────┘
                                  │
              ┌───────────────────▼───────────────────────────┐
              │  Step 2: Batch email generation (1 LLM call)  │
              │                                               │
              │  Input per company:                           │
              │    name + sector + top_signals[:2] +          │
              │    outreach_angle + hiring_news[0][:150]      │
              │                                               │
              │  Output: JSON array of {index, subject, body} │
              │                                               │
              │  Merge into company["outreach"] = {           │
              │    email_subject, email_body, drafted_at      │
              │  }                                            │
              └───────────────────┬───────────────────────────┘
                                  │
              ┌───────────────────▼───────────────────────────┐
              │  Step 3: Template injection                   │
              │                                               │
              │  Read templates/report_template.html          │
              │  JSON.stringify(companies)                    │
              │  Replace /*__DATA__*/[]/*__END__*/ placeholder│
              │  Write to output/report.html                  │
              │                                               │
              │  The HTML is self-contained:                  │
              │    - No external API calls                    │
              │    - No server needed                         │
              │    - Opens in any browser                     │
              └───────────────────────────────────────────────┘
```

---

## Step 1: Extracting the Top N

The analyst produces a `global_leaderboard` — all companies across all sectors,
sorted by `analyst.score` descending. The reporter takes the top slice:

```python
leaderboard = analyst_output.get("global_leaderboard", [])
top = leaderboard[:config.TOP_COMPANIES_FOR_REPORT]  # 15
```

Why only top N? Email drafts cost LLM tokens. We don't want to draft emails for
companies scored at 20/100 — they're not worth reaching out to.

---

## Step 2: Batch Email Generation

### Theory

Each email must:
- Be under 120 words
- Reference ONE specific signal (not generic "your company is growing")
- Have a concrete CTA (not "let's chat sometime")
- Feel written by a human who read the company's news

### The Batch Strategy

Instead of 15 separate LLM calls (one per email), we send **one call with all 15
specs**. This saves 14 round-trips of latency and reduces overhead tokens.

```python
def _batch_email_drafts(companies: list, agent: Agent) -> list:
    specs = []
    for i, c in enumerate(companies):
        analyst = c.get("analyst") or {}
        signals = (analyst.get("top_signals") or [])[:2]
        angle   = analyst.get("outreach_angle") or "N/A"

        # Enrich with hiring news for extra context
        hiring_blurb = ""
        hiring_hits = ((c.get("news") or {}).get("hiring") or [])
        if hiring_hits:
            content = (hiring_hits[0].get("content") or "")[:150]
            if content:
                hiring_blurb = f" | hiring: {content}"

        specs.append(
            f"{i+1}. {c.get('name')} | {sector} | "
            f"signals: {', '.join(signals)} | "
            f"angle: {angle}{hiring_blurb}"
        )

    prompt = (
        "Write outreach emails for these prospects.\n\n"
        + "\n".join(specs) + "\n\n"
        'Return JSON: [{"index":1,"subject":"…","body":"…"},…]'
    )
```

### Prompt Example (what the LLM sees)

```
Write outreach emails for these prospects.

1. TCI Ltd | travel logistics | signals: AI fleet tracking, new CEO designate
   | angle: Logistics tech modernization
   | hiring: Annual Report 2024-25 Transport Corporation of India...
2. TRENT Ltd | travel logistics | signals: AI inventory prediction, warehouse expansion
   | angle: Retail supply chain digitization
3. Surya Roshni Ltd | travel logistics | signals: IoT fleet management
   | angle: Manufacturing logistics optimization

Return JSON: [{"index":1,"subject":"…","body":"…"},…]
```

### Merge Logic

The LLM returns indices matching the spec list. We merge by index:

```python
for em in emails:
    idx = em.get("index", 0) - 1      # 1-based → 0-based
    if 0 <= idx < len(companies):
        companies[idx]["outreach"] = {
            "email_subject": em.get("subject", ""),
            "email_body":    em.get("body", ""),
            "drafted_at":    datetime.now().isoformat(),
        }
```

Companies without a matching email get a fallback stub:

```python
for c in companies:
    if "outreach" not in c:
        c["outreach"] = {
            "email_subject": f"Quick question for {c.get('name')}",
            "email_body": "[Draft pending]",
            "_fallback": True,        # template can render this differently
        }
```

---

## Step 3: Template Injection

### How It Works

The HTML template is a self-contained single file. No React, no build step, no
server. Just vanilla HTML + CSS + JS.

The data injection point:

```html
<script>
const RAW_DATA = /*__DATA__*/[]/*__END__*/;
</script>
```

The reporter replaces the placeholder with the actual JSON:

```python
def _inject_data_into_template(companies: list) -> str:
    with open(template_file) as f:
        html = f.read()

    # Deep copy + serialize (default=str handles datetimes)
    safe = []
    for c in companies:
        copy = json.loads(json.dumps(c, default=str))
        # Safety cap: annual report excerpts can be huge
        docs = copy.get("documents")
        if isinstance(docs, dict):
            excerpt = docs.get("annual_report_excerpt")
            if isinstance(excerpt, str) and len(excerpt) > 500:
                docs["annual_report_excerpt"] = excerpt[:500]
        safe.append(copy)

    data_json = json.dumps(safe, ensure_ascii=False, default=str)
    html = html.replace("/*__DATA__*/[]/*__END__*/",
                         f"/*__DATA__*/{data_json}/*__END__*/")
    return html
```

### The Accessor Layer

The template's JS uses a `mapCompany()` function that flattens the nested
unified schema into rendering-friendly flat names:

```javascript
function mapCompany(c) {
  const an = c.analyst || {};
  const pr = c.profile || {};
  const fi = c.financials || {};
  const id = c.identifiers || {};
  const fe = c.computed_features || {};
  return {
    // Table columns
    name:         c.name,
    ticker:       id.yahoo_ticker || '',
    sector:       c.sector_query || pr.industry || '',
    score:        an.score || 0,
    global_rank:  an.global_rank || 0,
    revenue_growth_pct: fe.revenue_growth_pct_4q || 0,

    // Detail panel - Financials
    revenue_ttm:  fi.revenue_ttm_crore,
    market_cap:   fi.market_cap_crore,
    ebitda_margin: fi.ebitda_margin,

    // Detail panel - Signals
    top_signals:  an.top_signals || [],
    reasoning:    an.reasoning || '',

    // Detail panel - Outreach
    email_subject: (c.outreach || {}).email_subject || '',
    email_body:    (c.outreach || {}).email_body || '',

    // Data quality
    failed:       (c.provenance || {}).failed_stages || [],
    confidence:   id.name_match_confidence || 0,
  };
}

let companies = RAW_DATA.map(mapCompany);
```

This pattern means:
- The **schema can evolve** without rewriting every rendering function
- All field mappings are in **one place**
- The rendering code uses clean names like `c.score` not `c.analyst.score`

---

## The Dashboard

### Layout

```
┌──────────┬───────────────────────────────────────────────────────────┐
│          │  Header: Prospect Intelligence            [CSV] [Print]  │
│ SIDEBAR  ├──────────────────────────────────────────────────────────┤
│          │                                                          │
│ Search   │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐│
│          │  │Prospects│ │Scoreable│ │High Pri│ │Avg Scor│ │Top Sect││
│ Sectors  │  │  15     │ │  12    │ │  4     │ │  68    │ │Fintech ││
│ [x] IT   │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘│
│ [x] Mfg  │                                                          │
│          │  Showing 12 of 15 companies                              │
│ Min Score│  ┌───────────────────────────────────────────────────────┐│
│ [===•==] │  │ # │ Company          │Sector│Score  │Trend│Data│Signal││
│          │  ├───┼──────────────────┼──────┼───────┼─────┼────┼──────┤│
│ Trend    │  │ 1 │ KPIT Tech        │ ISV  │ 89 ██ │↑+34%│ ●  │AI   ││
│ [All]    │  │ 2 │ Persistent Sys   │ ISV  │ 82 ██ │↑+21%│ ●  │Cloud││
│ [Growing]│  │ 3 │ TCI Ltd          │ Logi │ 72 ██ │↑+9% │ ●  │Fleet││
│          │  │...│ ...              │ ...  │ ...   │ ... │ ...│ ... ││
│ Data Qual│  └───────────────────────────────────────────────────────┘│
│ [All]    │                                                          │
│ [Full]   │  Expanded detail (when row clicked):                     │
│          │  ┌───────────────────────────────────────────────────────┐│
│ Last     │  │ [Overview] [Financials] [Signals] [Outreach]         ││
│ refreshed│  │                                                       ││
│ 12 Apr   │  │  Revenue: ₹4,790 Cr    Market Cap: ₹7,790 Cr        ││
│          │  │  EBITDA: 10.4%          Profit: 9.3%                  ││
│          │  │  Quarterly Revenue:  ▉▉▉▊  (1248/1139/1174/1147)     ││
│          │  └───────────────────────────────────────────────────────┘│
└──────────┴──────────────────────────────────────────────────────────┘
```

### Tabbed Detail Panel

Each company row expands into a panel with 4 tabs:

**Tab 1: Overview**
- Company description + website link + screener.in link
- Badges: industry, employee band, revenue band, CXO change, concall
- Name match confidence indicator
- Provenance warning if failed_stages exist

**Tab 2: Financials**
- 8-card grid: Revenue TTM, Market Cap, EBITDA, Profit Margin,
  Growth YoY, Employees, Target Price, Current Price
- Quarterly revenue bar chart (pure CSS, no library)
- Credit ratings list

**Tab 3: Signals & Intelligence**
- Buy signals list (green dots)
- Analyst reasoning (narrative)
- Risk factors (red warning icons)
- Recent announcements (from screener, with date + CXO context)
- Hiring news (from Tavily, with source links)

**Tab 4: Outreach**
- Outreach angle + recommended approach
- Email draft card with subject + body
- Copy-to-clipboard button

### Data Quality Column

A colored dot in the table indicates enrichment completeness:

```
 ●  Green   = 0 failed stages (full data)
 ●  Yellow  = 1-2 failed stages (partial — e.g., no PDF)
 ●  Red     = name_resolution failed (barely any data)
```

This lets the sales team quickly see which prospects have solid data
vs which ones need manual verification.

### Filtering

```javascript
// All filters run client-side (no server needed)
function getFiltered() {
  return companies.filter(c => {
    // Text search: company name or ticker
    if (q && !c.name.toLowerCase().includes(q) &&
        !c.ticker.toLowerCase().includes(q)) return false;
    // Minimum score slider
    if (c.score < minScore) return false;
    // Sector checkboxes
    if (!checkedSectors.includes(c.sector)) return false;
    // Revenue trend: growing / flat
    if (activeTrend === 'growing' && c.revenue_growth_pct <= 0) return false;
    // Data quality: full data only
    if (activeDQ === 'good' && c.failed.length > 0) return false;
    return true;
  });
}
```

### CSV Export

```javascript
function exportCSV() {
  const headers = ['Rank','Company','Ticker','Sector','Industry','Score',
                   'Growth%','Revenue Cr','Market Cap Cr','Employees',
                   'Top Signal','Reasoning'];
  // ... maps company data to CSV rows
  // Triggers browser download of prospects_report.csv
}
```

---

## The S3 Pipeline: How Everything Connects

```
 main.py: run_pipeline()
 │
 ├── run_id = generate_run_id()       "run_2026-04-12T10-46-07"
 │
 ├── researcher_output = run_researcher()
 │   └── upload_artifact(run_id, "research.json", researcher_output)
 │       → s3://prospect-ai-data/runs/run_2026-04-12T10-46-07/research.json
 │
 ├── analyst_output = run_analyst(researcher_output)
 │   └── upload_artifact(run_id, "analysis.json", analyst_output)
 │       → s3://prospect-ai-data/runs/run_2026-04-12T10-46-07/analysis.json
 │
 ├── html = run_reporter(analyst_output)
 │   └── upload_artifact(run_id, "report.html", html)
 │       → s3://prospect-ai-data/runs/run_2026-04-12T10-46-07/report.html
 │
 └── metrics = _build_metrics(run_id, start, end, ...)
     └── upload_artifact(run_id, "metrics.json", metrics)
         → s3://prospect-ai-data/runs/run_2026-04-12T10-46-07/metrics.json
```

### Querying by Run ID

```python
from utils.s3_storage import get_run_artifacts, list_runs

# List recent runs
runs = list_runs(limit=10)
# → ["run_2026-04-12T10-46-07", "run_2026-04-11T14-30-22", ...]

# Get all 4 artifacts for a specific run
artifacts = get_run_artifacts("run_2026-04-12T10-46-07")
# → {
#     "run_id": "run_2026-04-12T10-46-07",
#     "research": { sectors: [...] },          # parsed JSON
#     "analysis": { global_leaderboard: [...] }, # parsed JSON
#     "report":   "<html>...</html>",           # raw HTML string
#     "metrics":  { duration_seconds: 120, ... }, # parsed JSON
#   }
```

### Metrics JSON

```json
{
  "run_id": "run_2026-04-12T10-46-07",
  "started_at": "2026-04-12T10:46:07",
  "completed_at": "2026-04-12T10:48:07",
  "duration_seconds": 120.3,
  "agents": {
    "Researcher":      { "calls": 1, "in": 12000, "out": 5000, "latency": 23.4 },
    "Analyst":         { "calls": 1, "in": 8000,  "out": 3000, "latency": 15.2 },
    "Reporter-emails": { "calls": 1, "in": 2000,  "out": 1500, "latency": 8.1  }
  },
  "totals": {
    "input_tokens": 22000,
    "output_tokens": 9500,
    "total_tokens": 31500,
    "estimated_cost_usd": 0.21
  },
  "pipeline": {
    "sectors_processed": 1,
    "companies_discovered": 5,
    "companies_resolved": 3,
    "companies_scored": 3,
    "companies_skipped": 2
  }
}
```

---

## Token Efficiency

The reporter is the cheapest agent because the LLM only writes emails:

```
 Agent             LLM Calls    Tokens (est.)    Cost
 ────────────────  ──────────   ──────────────   ─────
 Researcher         1/sector    ~17,000           $0.08
 Analyst            1/sector    ~11,000           $0.05
 Reporter           1 total     ~3,500            $0.02
 ──────────────────────────────────────────────────────
 Full run (1 sector, 5 companies)                $0.15
 Full run (8 sectors, 120 companies)             ~$1.20
```

The original design had the LLM generate HTML — that alone cost 50,000+ tokens.
The template-based approach uses ~3,500 tokens total (just the email batch).

---

## How to Extend

### Adding a new tab to the detail panel

1. Add the tab button in `renderDetail()`:
```javascript
'<button class="tab-btn" onclick="switchTab(this,uid,\'mytab\')">My Tab</button>'
```

2. Add the content div:
```javascript
'<div id="'+uid+'_mytab" class="tab-content">Your content here</div>'
```

3. Map any new fields in `mapCompany()`.

### Changing the email style

Edit the system prompt in `create_email_agent()`:
```python
system_prompt = (
    "You write short B2B sales outreach emails..."
    "Each email: subject + body under 120 words."
    "Reference ONE specific signal."
    # Add: "Use a casual tone" or "Include a specific ROI stat"
)
```

### Changing how many companies get emails

```python
TOP_COMPANIES_FOR_REPORT = 15   # config.py — change this
```

### Adding the report to a notification system

After `run_reporter()` returns the HTML:
```python
html = run_reporter(analyst_output)
# Upload to S3 (already done)
upload_artifact(run_id, "report.html", html)
# Send via email / Slack / webhook
send_notification(run_id, html_url=f"https://your-bucket.s3.amazonaws.com/runs/{run_id}/report.html")
```

---

## Full System Summary

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                        PROSPECT AI PIPELINE                             │
 │                                                                         │
 │  config.py                                                              │
 │  ┌─────────────────────────────────────────────────────────────────┐    │
 │  │ SECTORS, REVENUE_MIN/MAX, BUY_SIGNALS, BEDROCK_MODEL_ID,       │    │
 │  │ S3_BUCKET, S3_RUNS_PREFIX, TOP_COMPANIES_FOR_REPORT            │    │
 │  └─────────────────────────────────────────────────────────────────┘    │
 │                                                                         │
 │  main.py: run_pipeline()                                                │
 │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
 │  │  RESEARCHER   │───▶│   ANALYST    │───▶│   REPORTER   │              │
 │  │              │    │              │    │              │              │
 │  │ Discovery    │    │ Score 0-100  │    │ Email drafts │              │
 │  │ + Enrichment │    │ Python feats │    │ + HTML report│              │
 │  │              │    │ + LLM quals  │    │              │              │
 │  │ 5 sources:   │    │              │    │ Template-    │              │
 │  │  Tavily      │    │ Pre-filter   │    │ based (no    │              │
 │  │  yfinance    │    │ unresolvable │    │ LLM HTML)    │              │
 │  │  screener.in │    │              │    │              │              │
 │  │  LLM (names) │    │ AR recovery  │    │ 1 LLM call   │              │
 │  │  PDF extract │    │ via Tavily   │    │ for all      │              │
 │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
 │         │                   │                   │                      │
 │  ┌──────▼───────────────────▼───────────────────▼───────────────┐      │
 │  │                   S3 STORAGE                                 │      │
 │  │  s3://prospect-ai-data/runs/{run_id}/                        │      │
 │  │    ├── research.json   (enriched companies)                  │      │
 │  │    ├── analysis.json   (scored leaderboard)                  │      │
 │  │    ├── report.html     (self-contained dashboard)            │      │
 │  │    └── metrics.json    (tokens, timing, cost)                │      │
 │  └──────────────────────────────────────────────────────────────┘      │
 │                                                                         │
 │  utils/                                                                 │
 │  ├── token_tracker.py   (per-call metrics, cost estimation)             │
 │  ├── s3_storage.py      (run-id-based artifact storage)                 │
 │  └── logger.py          (file + stdout logging)                         │
 │                                                                         │
 │  tools/                                                                 │
 │  └── company_scraper.py (yfinance + screener.in + PDF extraction)       │
 │                                                                         │
 └─────────────────────────────────────────────────────────────────────────┘
```

### Unified Company Schema (all 3 agents agree on this)

```
{
  name, exchange, sector_query,
  identifiers:       { yahoo_ticker, screener_symbol, name_match_confidence }
  profile:           { description, long_business_summary, website, employees, industry }
  financials:        { revenue_ttm_crore, market_cap_crore, margins, growth, quarters }
  documents:         { screener_url, announcements, annual_reports, concalls, credit_ratings }
  news:              { hiring: [{title, url, content}] }
  llm_seed:          { tech_signals, recent_news, key_quotes }
  provenance:        { researched_at, failed_stages, errors }
  computed_features: { revenue_growth_pct_4q, has_cxo, has_concall, in_band, emp_band }
  analyst:           { score, global_rank, top_signals, reasoning, outreach_angle }
  outreach:          { email_subject, email_body, drafted_at }
}
```

**Each agent owns its namespace. No agent overwrites another's data.**
- Researcher writes: identifiers, profile, financials, documents, news, provenance
- Analyst writes: computed_features, analyst
- Reporter writes: outreach
