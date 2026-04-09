# agents/researcher.py — Token-optimized with usage tracking + scraper enrichment

import os, json, time, requests
from datetime import datetime
from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel
import yfinance as yf
import config
from utils.token_tracker import tracker
from utils.logger import log, warn, error
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

    # Single combined search instead of 3 separate ones — saves 1 Tavily call
    q1 = f"Indian listed BSE NSE {sector} revenue 500-2000 crore 2024 2025"
    q2 = f"Indian {sector} digital transformation AI technology 2024 2025"
    r1 = tavily_search(q1, 6)
    r2 = tavily_search(q2, 4)
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

    result["researched_at"] = datetime.now().isoformat()

    companies = result.get("companies", [])
    company_names = [c.get("name", "?") for c in companies]
    log(f"LLM identified {len(companies)} companies in '{sector}': {company_names}")

    # ── Per-company enrichment via tools.company_scraper ──────────────
    ok = 0
    for company in companies:
        try:
            _enrich_company(company)
            yf_block = company.get("enriched", {}).get("yfinance") or {}
            if isinstance(yf_block, dict) and "error" not in yf_block:
                ok += 1
        except Exception as e:
            error(f"enrich failed for {company.get('name', '?')}: {e}")
            company["enriched"] = {"error": str(e)}
    log(f"Enriched {ok}/{len(companies)} companies (with yfinance data) in sector '{sector}'")

    return result


# ── Topics for per-company news enrichment ─────────────────────────
# Each topic produces one Tavily query. Snippets are saved raw under
# enriched["news"][topic] — no extra LLM call (user wants raw data).
NEWS_TOPICS = {
    "hiring":        '"{name}" hiring layoffs employees India 2024 2025',
    "leadership":    '"{name}" CEO CXO leadership awards interview India',
    "financials":    '"{name}" quarterly results revenue profit guidance 2024 2025',
    "announcements": '"{name}" press release announcement India 2024 2025',
    "annual_report": '"{name}" annual report 2024 PDF',
}


def _resolve_yahoo_ticker(name: str, exchange_hint: str = "") -> str | None:
    """
    Use yfinance.Search to map a company NAME to a real Yahoo symbol.
    LLMs hallucinate Indian tickers (e.g. 'SONATA' instead of 'SONATSOFTW') —
    name-based search is the reliable path. Prefers .NS over .BO.
    """
    try:
        quotes = yf.Search(name, max_results=8).quotes or []
    except Exception as e:
        warn(f"[enrich] yf.Search failed for {name!r}: {e}")
        return None

    indian = [q for q in quotes if str(q.get("symbol", "")).endswith((".NS", ".BO"))]
    if not indian:
        return None
    # Prefer NSE if exchange hint says NSE or is empty; honor BSE hint if given
    if exchange_hint.upper() == "BSE":
        ns_first = sorted(indian, key=lambda q: 0 if q["symbol"].endswith(".BO") else 1)
    else:
        ns_first = sorted(indian, key=lambda q: 0 if q["symbol"].endswith(".NS") else 1)
    return ns_first[0]["symbol"]


def _search_company_news(name: str) -> dict:
    """
    Run one targeted Tavily query per topic. Returns raw snippets shaped as
    {topic: [{title, url, content (truncated)}, ...]} — no LLM summarization.
    Errors per-topic are caught so one bad query doesn't kill the whole company.
    """
    out: dict = {}
    for topic, template in NEWS_TOPICS.items():
        query = template.format(name=name)
        try:
            results = tavily_search(query, max_results=3)
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
        # tiny pause so we don't hammer Tavily across 25 calls in a few seconds
        time.sleep(0.5)
    return out


def _try_extract_annual_report_pdf(name: str, screener: dict, news: dict) -> str | None:
    """
    Pick the most recent annual report PDF URL and extract first ~3000 chars
    via CompanyDataFetcher. Prefers screener.in's curated annual_reports list
    (clean, dated, ordered newest-first); falls back to whatever .pdf URL the
    Tavily 'annual_report' search turned up. Returns excerpt or None.
    """
    pdf_url = None

    # 1) Preferred source: screener.annual_reports (already newest-first)
    for entry in (screener or {}).get("annual_reports", []):
        url = entry.get("url", "")
        if url.lower().endswith(".pdf"):
            pdf_url = url
            break

    # 2) Fallback: any .pdf URL from the Tavily annual_report search
    if not pdf_url:
        for hit in (news or {}).get("annual_report", []):
            url = hit.get("url", "")
            if url.lower().endswith(".pdf"):
                pdf_url = url
                break

    if not pdf_url:
        log(f"[enrich] PDF skipped {name} — no .pdf URL found (screener or news)")
        return None
    try:
        excerpt = CompanyDataFetcher(name).extract_annual_report_pdf(pdf_url)
        log(f"[enrich] PDF OK {name}: {len(excerpt)} chars from {pdf_url}")
        return excerpt
    except Exception as e:
        warn(f"[enrich] PDF FAIL {name} ({pdf_url}): {e}")
        return None


def _enrich_company(company: dict) -> None:
    """Attach scraper-fetched data to a company dict in place under company['enriched']."""
    name = company.get("name", "?")
    exchange_hint = (company.get("exchange") or "").strip().upper()

    log(f"[enrich] START {name}")

    enriched = {
        "resolved_ticker": None,
        "screener_symbol": None,
        "yfinance": None,
        "screener": {},
        "news": {},
        "annual_report_pdf_excerpt": None,
        "fetched_at": datetime.now().isoformat(),
    }

    # 1) Resolve a real Yahoo ticker by COMPANY NAME (not the LLM's guess).
    yahoo_ticker = _resolve_yahoo_ticker(name, exchange_hint)
    enriched["resolved_ticker"] = yahoo_ticker
    if yahoo_ticker:
        log(f"[enrich] resolved {name} → {yahoo_ticker}")
    else:
        warn(f"[enrich] {name}: yfinance.Search returned no Indian listing")

    # 2) yfinance financials
    if yahoo_ticker:
        try:
            fetcher = CompanyDataFetcher(name, yahoo_ticker=yahoo_ticker)
            fin = fetcher.get_financials_from_yahoo()
            enriched["yfinance"] = fin
            present = sorted([k for k, v in (fin or {}).items() if v is not None])
            log(f"[enrich] yfinance OK {name} ({yahoo_ticker}): fields={present}")
        except Exception as e:
            warn(f"[enrich] yfinance FAIL {name} ({yahoo_ticker}): {e}")
            enriched["yfinance"] = {"error": str(e)}
    else:
        enriched["yfinance"] = {"error": "no resolved ticker"}

    # 3) Screener.in documents — replaces the old BSE-API call. One HTTP fetch
    # gives us announcements (with CXO-change context!), annual reports,
    # concalls and credit ratings.
    screener_symbol = None
    if yahoo_ticker and yahoo_ticker.endswith((".NS", ".BO")):
        screener_symbol = yahoo_ticker.rsplit(".", 1)[0]
    enriched["screener_symbol"] = screener_symbol
    if screener_symbol:
        try:
            screener_docs = CompanyDataFetcher(name).get_screener_documents(screener_symbol)
            enriched["screener"] = screener_docs
            log(
                f"[enrich] screener OK {name} ({screener_symbol}): "
                f"announcements={len(screener_docs.get('announcements', []))} "
                f"annual_reports={len(screener_docs.get('annual_reports', []))} "
                f"concalls={len(screener_docs.get('concalls', []))} "
                f"credit_ratings={len(screener_docs.get('credit_ratings', []))}"
            )
            if screener_docs.get("error"):
                warn(f"[enrich] screener {name}: {screener_docs['error']}")
        except Exception as e:
            warn(f"[enrich] screener FAIL {name} ({screener_symbol}): {e}")
            enriched["screener"] = {"error": str(e)}
    else:
        log(f"[enrich] screener skipped {name} — no resolved ticker to derive symbol")

    # 4) Per-company Tavily news search across the 5 topics — the meat of the
    # enrichment for the user's stated needs (hiring/CXO/financials/etc.)
    enriched["news"] = _search_company_news(name)

    # 5) Annual report PDF extraction — prefers screener's curated list,
    # falls back to Tavily annual_report search hits.
    enriched["annual_report_pdf_excerpt"] = _try_extract_annual_report_pdf(
        name, enriched.get("screener", {}), enriched["news"]
    )

    company["enriched"] = enriched
    log(f"[enrich] DONE {name}")


def run_researcher() -> dict:
    log("=" * 60)
    log("AGENT 1: RESEARCHER STARTING")
    log("=" * 60)

    all_data = {"run_date": datetime.now().isoformat(), "sectors": []}

    for sector in config.SECTORS:
        sector_data = research_sector(sector)
        all_data["sectors"].append(sector_data)

        safe = sector.replace(" ", "_").replace("/", "_")
        path = f"data/raw/{safe}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(sector_data, f, indent=2)
        log(f"💾 Saved {path}")

    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/all_sectors.json", "w") as f:
        json.dump(all_data, f, indent=2)

    total = sum(len(s.get("companies", [])) for s in all_data["sectors"])
    log(f"✅ RESEARCHER DONE — {total} companies across {len(all_data['sectors'])} sectors")
    return all_data


if __name__ == "__main__":
    tracker.reset()
    run_researcher()
    tracker.print_summary()
