# agents/researcher.py — Token-optimized with usage tracking + scraper enrichment
#
# Output shape: each company is normalized into the unified schema with
# top-level namespaces (identifiers / profile / financials / documents /
# news / provenance). The analyst and reporter consume this same shape.

import os, json, time, difflib, requests
from datetime import datetime
from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel
import yfinance as yf
import config
from utils.token_tracker import tracker
from utils.logger import log, warn, error
from utils.s3_storage import upload_research_output
from tools.company_scraper import CompanyDataFetcher

load_dotenv()


def tavily_search(query: str, max_results: int = 5) -> list:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set")
    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": max_results,
              "search_depth": "advanced", "include_answer": True},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def _compact_search(results: list, char_limit: int) -> str:
    """Trim search results to stay under char_limit — saves input tokens."""
    parts = []
    total = 0
    for r in results:
        chunk = f"{r.get('url','')}\n{r.get('content','')[:300]}\n"
        if total + len(chunk) > char_limit:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


def create_researcher_agent():
    model = BedrockModel(model_id=config.BEDROCK_MODEL_ID, region_name=config.AWS_REGION)

    # Leaner system prompt — ~40% fewer tokens than original
    system_prompt = (
        "You are a market research analyst for Indian listed companies (BSE/NSE). "
        f"Find companies with revenue ₹{config.REVENUE_MIN_CRORE}-{config.REVENUE_MAX_CRORE} Cr. "
        "Return ONLY valid JSON. No markdown fences, no extra text.\n"
        'Format: {"sector":"…","companies":[{"name":"…","ticker":"…","exchange":"BSE/NSE",'
        '"revenue_quarters":[{"quarter":"Q1 FY24","revenue_crore":650}],'
        '"recent_news":["…"],"tech_signals":["…"],"key_quotes":["…"],'
        '"website":"…","description":"…"}]}'
    )
    return Agent(model=model, system_prompt=system_prompt)


def _call_agent(agent, prompt: str, label: str):
    """Call agent, measure latency, record tokens."""
    t0 = time.time()
    response = agent(prompt)
    latency = time.time() - t0
    tracker.record("Researcher", response, prompt, latency)
    return response


def research_sector(sector: str) -> dict:
    log(f"🔍 Researching sector: {sector}")
    agent = create_researcher_agent()
    limit = config.MAX_SEARCH_CHARS

    # Single combined search instead of 3 separate ones — saves 1 Tavily call.
    # Both calls are now wrapped: a single Tavily 429/5xx no longer kills the
    # whole sector loop. Empty results just make the LLM work harder.
    q1 = f"Indian listed BSE NSE {sector} revenue 500-2000 crore 2024 2025"
    q2 = f"Indian {sector} digital transformation AI technology 2024 2025"

    try:
        r1 = tavily_search(q1, 6)
    except Exception as e:
        warn(f"[research] sector tavily q1 FAIL for {sector!r}: {e}")
        r1 = []
    try:
        r2 = tavily_search(q2, 4)
    except Exception as e:
        warn(f"[research] sector tavily q2 FAIL for {sector!r}: {e}")
        r2 = []

    log(f"Tavily returned {len(r1)} + {len(r2)} results for sector '{sector}'")

    search_text = _compact_search(r1, limit)
    tech_text = _compact_search(r2, limit // 2)

    # Push the model to actually return MAX_COMPANIES_PER_SECTOR — earlier
    # "up to 5" wording made it lazy and return 1.
    prompt = (
        f'List EXACTLY {config.MAX_COMPANIES_PER_SECTOR} Indian listed companies (BSE/NSE) in "{sector}" '
        f"with annual revenue ₹{config.REVENUE_MIN_CRORE}-{config.REVENUE_MAX_CRORE} Cr. "
        f"You MUST return {config.MAX_COMPANIES_PER_SECTOR} entries — if fewer perfect matches exist, "
        "include adjacent ones. Use the OFFICIAL stock symbols (e.g. SONATSOFTW for Sonata Software, "
        "not SONATA). Use the company's full legal name in the 'name' field.\n\n"
        f"=== SEARCH ===\n{search_text}\n\n=== TECH NEWS ===\n{tech_text}\n\n"
        f"Return structured JSON with a top-level 'companies' array of length {config.MAX_COMPANIES_PER_SECTOR}. "
        "Estimate quarterly revenue from annual if needed."
    )

    response = _call_agent(agent, prompt, sector)
    response_text = str(response)
    log(f"LLM returned {len(response_text)} chars for sector '{sector}'")

    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        result = json.loads(response_text[start:end]) if start != -1 and end > start else {
            "sector": sector, "companies": [], "error": "No JSON found"}
    except json.JSONDecodeError as e:
        error(f"JSON parse failed for sector '{sector}': {e}")
        result = {"sector": sector, "companies": [], "error": str(e)}

    result["sector"] = sector  # ensure set even on parse failure
    result["researched_at"] = datetime.now().isoformat()

    companies = result.get("companies", [])
    company_names = [c.get("name", "?") for c in companies]
    log(f"LLM identified {len(companies)} companies in '{sector}': {company_names}")

    # ── Per-company enrichment via tools.company_scraper ──────────────
    ok = 0
    for company in companies:
        try:
            _enrich_company(company, sector_query=sector)
            failed = (company.get("provenance") or {}).get("failed_stages", [])
            if "yfinance" not in failed and "name_resolution" not in failed:
                ok += 1
        except Exception as e:
            error(f"enrich failed for {company.get('name', '?')}: {e}")
            company.setdefault("provenance", {"researched_at": datetime.now().isoformat()})
            company["provenance"].setdefault("failed_stages", []).append("enrichment")
            company["provenance"].setdefault("errors", []).append(str(e))
    log(f"Enriched {ok}/{len(companies)} companies (with yfinance data) in sector '{sector}'")

    return result


# ── Topics for per-company news enrichment ─────────────────────────
# Reduced to ONE topic — leadership/financials/announcements/annual_report
# are all already covered by the screener.in scrape that runs seconds earlier.
# Hiring is the only signal Tavily contributes that screener can't.
NEWS_TOPICS = {
    "hiring": '"{name}" hiring layoffs headcount India 2024 2025',
}


# Annual report selection — prefer the most recent fiscal year. Screener
# titles use varied formats: "Financial Year 2025", "FY25", "2024-25", etc.
# Match against all common variants (case-insensitive). If none match, the
# caller falls back to annual_reports[0].
TARGET_YEARS = (
    "2024-25", "2023-24",           # hyphenated FY
    "FY25", "FY24", "FY 25", "FY 24",  # abbreviated
    "Financial Year 2025", "Financial Year 2024",  # screener's actual format
    "Year 2025", "Year 2024",       # partial
)


def _get_target_annual_report_url(annual_reports: list) -> str | None:
    """Filter screener annual_reports for FY24/FY25 specifically.
    Returns the first .pdf URL whose title or date contains any TARGET_YEARS
    token (case-insensitive). None if no recent-year report is found —
    caller decides whether to fall back to annual_reports[0]."""
    for entry in annual_reports or []:
        url = (entry.get("url") or "")
        if not url.lower().endswith(".pdf"):
            continue
        haystack = f"{entry.get('title','')} {entry.get('date','')}".lower()
        for yr in TARGET_YEARS:
            if yr.lower() in haystack:
                return url
    return None


def _build_search_queries(name: str) -> list[str]:
    """Progressively shorter search queries from a company name.
    Yahoo's search index chokes on the trailing 'Ltd' / 'Limited' that
    Indian LLMs love to include (e.g. 'Shipping Corporation of India Ltd'
    → 0 results, but 'Shipping Corporation of India' → SCI.BO instantly).

    Returns [full_name, without_suffixes, first_3_words, first_2_words]
    with duplicates removed."""
    queries = [name]
    # Strip common suffixes
    stripped = name
    for token in ("Limited", "Ltd.", "Ltd", "Pvt.", "Pvt"):
        stripped = stripped.replace(token, "")
    stripped = stripped.strip(" ,.")
    if stripped and stripped != name:
        queries.append(stripped)
    # First N words — catches abbreviated names
    words = stripped.split()
    if len(words) > 3:
        queries.append(" ".join(words[:3]))
    if len(words) > 2:
        queries.append(" ".join(words[:2]))
    # De-dup while preserving order
    seen = set()
    out = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _resolve_yahoo_ticker(name: str, exchange_hint: str = "") -> tuple:
    """
    Use yfinance.Search to map a company NAME to a real Yahoo symbol.
    LLMs hallucinate Indian tickers (e.g. 'SONATA' instead of 'SONATSOFTW') —
    name-based search is the reliable path. Prefers .NS over .BO.

    Tries progressively shorter name variants because Yahoo's search index
    often fails on formal names with 'Ltd' / 'Limited' suffixes (e.g.
    'Shipping Corporation of India Ltd' → 0 results, but without 'Ltd' →
    SCI.BO immediately).

    When longname IS available, difflib.SequenceMatcher rejects matches
    below 0.6. When longname is NOT available (rare, but happens), the
    top-ranked Indian result is accepted with a 0.5 "blind" confidence.

    Returns (symbol, confidence, longname). All zero/None on failure.
    """
    # ── Multi-query search: try progressively shorter name variants ──
    queries = _build_search_queries(name)
    indian: list = []
    matched_query = name
    for query in queries:
        try:
            quotes = yf.Search(query, max_results=8).quotes or []
        except Exception as e:
            warn(f"[enrich] yf.Search failed for {query!r}: {e}")
            continue
        indian = [q for q in quotes if str(q.get("symbol", "")).endswith((".NS", ".BO"))]
        if indian:
            matched_query = query
            if query != name:
                log(f"[enrich] yf.Search retry hit on {query!r}: {len(indian)} Indian results")
            break

    if not indian:
        return (None, 0.0, None)

    # ── Exchange-preferred ranking ───────────────────────────────────
    if exchange_hint.upper() == "BSE":
        ranked = sorted(indian, key=lambda q: 0 if q["symbol"].endswith(".BO") else 1)
    else:
        ranked = sorted(indian, key=lambda q: 0 if q["symbol"].endswith(".NS") else 1)

    # ── Name similarity check ────────────────────────────────────────
    name_lower = name.lower()
    best = None
    best_conf = 0.0
    for q in ranked:
        longname = (q.get("longname") or q.get("shortname") or "").strip()
        if not longname:
            continue
        conf = difflib.SequenceMatcher(None, name_lower, longname.lower()).ratio()
        if conf > best_conf:
            best = q
            best_conf = conf

    # Fallback: no quote had a longname/shortname → accept top-ranked Indian
    # result with "blind" confidence. yf.Search found it from our name query,
    # so it's likely correct — we just can't compute similarity.
    if best is None:
        best = ranked[0]
        best_conf = 0.5
        longname = best.get("symbol", "")
        log(f"[enrich] blind match {name!r} → {best['symbol']} (no longname in yf response)")
    else:
        longname = (best.get("longname") or best.get("shortname") or "")
        if best_conf < 0.6:
            log(
                f"[enrich] name match too weak for {name!r} → {longname!r} "
                f"(confidence={best_conf:.2f}), rejecting"
            )
            return (None, best_conf, longname)

    return (best["symbol"], best_conf, longname)


def _search_company_news(name: str) -> dict:
    """
    Run one targeted Tavily query per topic (currently: hiring only).
    Returns raw snippets shaped as {topic: [{title, url, content}, ...]}.
    Errors per-topic are caught so one bad query doesn't kill the company.
    """
    out: dict = {}
    for topic, template in NEWS_TOPICS.items():
        query = template.format(name=name)
        try:
            results = tavily_search(query, max_results=4)
        except Exception as e:
            warn(f"[enrich] Tavily FAIL {name} topic={topic}: {e}")
            out[topic] = []
            continue
        hits = []
        for r in results or []:
            hits.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content") or "")[:500],
            })
        out[topic] = hits
        log(f"[enrich] news OK {name} topic={topic}: {len(hits)} hits")
        # Tiny pause so we don't hammer Tavily
        time.sleep(0.4)
    return out


def _try_extract_annual_report_pdf(name: str, screener: dict) -> str | None:
    """
    Two-stage screener-only selection (NO Tavily here — that's the analyst's
    recovery pass for companies that fall through both stages):

      Stage 1: target FY24/FY25 specifically via _get_target_annual_report_url
               — most recent fiscal year, the one we actually want.
      Stage 2: fall back to annual_reports[0] (whatever .pdf is newest in
               screener's list, regardless of year).

    Returns the extracted MD&A excerpt or None.
    """
    annual_reports = (screener or {}).get("annual_reports") or []

    # Stage 1: targeted FY24/FY25
    pdf_url = _get_target_annual_report_url(annual_reports)
    if pdf_url:
        log(f"[enrich] PDF stage1 (FY24/25 hit) {name}: {pdf_url}")
    else:
        # Stage 2: fall back to first .pdf URL in the list (newest-first)
        for entry in annual_reports:
            url = entry.get("url", "")
            if url.lower().endswith(".pdf"):
                pdf_url = url
                log(f"[enrich] PDF stage2 (fallback annual_reports[0]) {name}: {pdf_url}")
                break

    if not pdf_url:
        log(f"[enrich] PDF skipped {name} — no .pdf URL in screener.annual_reports")
        return None

    excerpt = CompanyDataFetcher(name).extract_annual_report_pdf(pdf_url)
    if excerpt:
        log(f"[enrich] PDF OK {name}: {len(excerpt)} chars from {pdf_url}")
    else:
        warn(f"[enrich] PDF FAIL {name} ({pdf_url})")
    return excerpt


def _to_crore(val) -> float | None:
    """Convert raw INR to crore for the schema. None-safe."""
    try:
        return round(float(val) / 1e7, 2) if val is not None else None
    except (TypeError, ValueError):
        return None


def _enrich_company(company: dict, sector_query: str = "") -> None:
    """
    Transform an LLM-discovered company dict (in place) into the unified schema:

      {
        name, exchange, sector_query,
        identifiers: { yahoo_ticker, screener_symbol, name_match_confidence },
        profile:     { description, long_business_summary, website, employees, industry },
        financials:  { revenue_ttm_crore, market_cap_crore, ..., revenue_quarters },
        documents:   { screener_url, announcements, annual_reports, concalls,
                       credit_ratings, annual_report_excerpt },
        news:        { hiring: [...] },
        llm_seed:    { tech_signals, recent_news, key_quotes, ticker_guess },
        provenance:  { researched_at, failed_stages, errors }
      }
    """
    name = company.get("name", "?")
    exchange_hint = (company.get("exchange") or "").strip().upper()

    log(f"[enrich] START {name}")

    # ── Preserve LLM seed before we restructure the dict ─────────────
    llm_seed = {
        "tech_signals": company.get("tech_signals", []) or [],
        "recent_news": company.get("recent_news", []) or [],
        "key_quotes": company.get("key_quotes", []) or [],
        "ticker_guess": company.get("ticker"),
    }
    llm_description = company.get("description", "") or ""
    llm_website = company.get("website", "") or ""
    llm_quarters = company.get("revenue_quarters", []) or []

    failed_stages: list = []
    errors: list = []

    # ── 1) Resolve a real Yahoo ticker by COMPANY NAME (not LLM guess) ──
    yahoo_ticker, name_confidence, longname = _resolve_yahoo_ticker(name, exchange_hint)
    if yahoo_ticker:
        log(
            f"[enrich] resolved {name} → {yahoo_ticker} "
            f"(confidence={name_confidence:.2f}, longname={longname!r})"
        )
    else:
        warn(f"[enrich] {name}: name resolution failed (confidence={name_confidence:.2f})")
        failed_stages.append("name_resolution")

    screener_symbol = None
    if yahoo_ticker and yahoo_ticker.endswith((".NS", ".BO")):
        screener_symbol = yahoo_ticker.rsplit(".", 1)[0]

    identifiers = {
        "yahoo_ticker": yahoo_ticker,
        "screener_symbol": screener_symbol,
        "name_match_confidence": round(name_confidence, 3),
    }

    # ── 2) yfinance financials + quarterly revenue ───────────────────
    yf_data: dict = {}
    real_quarters: list = []
    if yahoo_ticker:
        try:
            fetcher = CompanyDataFetcher(name, yahoo_ticker=yahoo_ticker)
            yf_data = fetcher.get_financials_from_yahoo() or {}
            present = sorted([k for k, v in yf_data.items() if v is not None])
            log(f"[enrich] yfinance OK {name} ({yahoo_ticker}): fields={present}")
            try:
                real_quarters = fetcher.get_quarterly_revenue()
                if real_quarters:
                    log(
                        f"[enrich] quarterly revenue OK {name}: "
                        f"{len(real_quarters)} quarters from yfinance"
                    )
            except Exception as qe:
                warn(f"[enrich] quarterly revenue FAIL {name}: {qe}")
        except Exception as e:
            warn(f"[enrich] yfinance FAIL {name} ({yahoo_ticker}): {e}")
            failed_stages.append("yfinance")
            errors.append(f"yfinance: {e}")
    else:
        failed_stages.append("yfinance")

    financials = {
        "revenue_ttm_crore": _to_crore(yf_data.get("revenue")),
        "market_cap_crore": _to_crore(yf_data.get("market_cap")),
        "ebitda_margin": yf_data.get("ebitda_margins"),
        "profit_margin": yf_data.get("profit_margins"),
        "revenue_growth_yoy": yf_data.get("revenue_growth"),
        "earnings_growth_yoy": yf_data.get("earnings_growth"),
        "free_cashflow_crore": _to_crore(yf_data.get("free_cashflow")),
        "total_debt_crore": _to_crore(yf_data.get("total_debt")),
        "current_price": yf_data.get("current_price"),
        "target_mean_price": yf_data.get("target_mean_price"),
        "analyst_recommendation": yf_data.get("recommendation_key"),
        # Real yfinance quarters preferred; LLM-guessed quarters as fallback only
        "revenue_quarters": real_quarters if real_quarters else llm_quarters,
    }

    profile = {
        "description": llm_description,
        "long_business_summary": yf_data.get("long_business_summary") or "",
        "website": yf_data.get("website") or llm_website or "",
        "employees": yf_data.get("employees"),
        "industry": yf_data.get("industry") or yf_data.get("sector") or "",
    }

    # ── 3) Screener.in documents ─────────────────────────────────────
    screener_docs: dict = {}
    if screener_symbol:
        try:
            screener_docs = (
                CompanyDataFetcher(name).get_screener_documents(screener_symbol) or {}
            )
            log(
                f"[enrich] screener OK {name} ({screener_symbol}): "
                f"announcements={len(screener_docs.get('announcements', []))} "
                f"annual_reports={len(screener_docs.get('annual_reports', []))} "
                f"concalls={len(screener_docs.get('concalls', []))} "
                f"credit_ratings={len(screener_docs.get('credit_ratings', []))}"
            )
            if screener_docs.get("error"):
                warn(f"[enrich] screener {name}: {screener_docs['error']}")
                failed_stages.append("screener")
                errors.append(f"screener: {screener_docs['error']}")
        except Exception as e:
            warn(f"[enrich] screener FAIL {name} ({screener_symbol}): {e}")
            failed_stages.append("screener")
            errors.append(f"screener: {e}")
    else:
        log(f"[enrich] screener skipped {name} — no resolved ticker to derive symbol")
        failed_stages.append("screener")

    # ── 4) Per-company hiring news (only Tavily topic remaining) ────
    news = _search_company_news(name)
    if not news.get("hiring"):
        # Empty hiring is not necessarily a failure (could be no real news),
        # so we don't mark it failed unless Tavily errored — that's already
        # logged inside _search_company_news.
        pass

    # ── 5) Annual report PDF excerpt ─────────────────────────────────
    annual_excerpt = _try_extract_annual_report_pdf(name, screener_docs)
    if annual_excerpt is None and screener_docs.get("annual_reports"):
        failed_stages.append("annual_report_pdf")

    documents = {
        "screener_url": screener_docs.get("url"),
        "announcements": screener_docs.get("announcements", []),
        "annual_reports": screener_docs.get("annual_reports", []),
        "concalls": screener_docs.get("concalls", []),
        "credit_ratings": screener_docs.get("credit_ratings", []),
        "annual_report_excerpt": annual_excerpt,
    }

    # ── 6) Replace the dict contents with the unified schema ────────
    company.clear()
    company["name"] = name
    company["exchange"] = exchange_hint or ""
    company["sector_query"] = sector_query
    company["identifiers"] = identifiers
    company["profile"] = profile
    company["financials"] = financials
    company["documents"] = documents
    company["news"] = news
    company["llm_seed"] = llm_seed
    company["provenance"] = {
        "researched_at": datetime.now().isoformat(),
        "failed_stages": failed_stages,
        "errors": errors,
    }

    log(f"[enrich] DONE {name}")


def run_researcher() -> dict:
    log("=" * 60)
    log("AGENT 1: RESEARCHER STARTING")
    log("=" * 60)

    all_data = {"run_date": datetime.now().isoformat(), "sectors": []}

    for sector in config.SECTORS:
        sector_data = research_sector(sector)
        all_data["sectors"].append(sector_data)
        log(f"   sector '{sector}' done")

    total = sum(len(s.get("companies", [])) for s in all_data["sectors"])
    log(f"✅ RESEARCHER DONE — {total} companies across {len(all_data['sectors'])} sectors")

    # Upload combined run to S3 (single file per run, timestamped key).
    # No local file fallback — fail loud so the user notices misconfiguration.
    try:
        uri = upload_research_output(all_data)
        log(f"✅ Research output archived → {uri}")
    except Exception as e:
        error(f"S3 upload FAILED, run output not persisted: {e}")
        raise

    return all_data


if __name__ == "__main__":
    tracker.reset()
    run_researcher()
    tracker.print_summary()
