"""Researcher agent – discovers companies via Tavily and enriches with financial data."""

import difflib
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import yfinance as yf
from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel

import config
from utils.token_tracker import tracker
from utils.logger import log, warn, error
from tools.company_scraper import CompanyDataFetcher

load_dotenv()

# ----------------------------------------------------------------------
# Tavily search wrapper
# ----------------------------------------------------------------------
def tavily_search(query: str, max_results: int = 5) -> List[Dict]:
    """Perform a Tavily search and return the list of results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set")
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": True,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def compact_search_results(results: List[Dict], char_limit: int) -> str:
    """Trim search results to stay under char_limit to save input tokens."""
    parts = []
    total_chars = 0
    for item in results:
        chunk = f"{item.get('url', '')}\n{item.get('content', '')[:300]}\n"
        if total_chars + len(chunk) > char_limit:
            break
        parts.append(chunk)
        total_chars += len(chunk)
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Agent creation and invocation
# ----------------------------------------------------------------------
def create_researcher_agent() -> Agent:
    """Build the Bedrock agent for company discovery."""
    model = BedrockModel(model_id=config.BEDROCK_MODEL_ID, region_name=config.AWS_REGION)
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


def call_agent(agent: Agent, prompt: str, label: str) -> str:
    """Invoke the agent, measure latency, and record token usage."""
    start = time.time()
    response = agent(prompt)
    tracker.record("Researcher", response, prompt, time.time() - start)
    return str(response)


# ----------------------------------------------------------------------
# Name resolution (LLM ticker → real Yahoo symbol)
# ----------------------------------------------------------------------
def build_name_search_queries(name: str) -> List[str]:
    """Generate progressively shorter variants to handle Yahoo's index quirks."""
    queries = [name]
    stripped = name
    for suffix in ("Limited", "Ltd.", "Ltd", "Pvt.", "Pvt"):
        stripped = stripped.replace(suffix, "")
    stripped = stripped.strip(" ,.")
    if stripped and stripped != name:
        queries.append(stripped)

    words = stripped.split()
    if len(words) > 3:
        queries.append(" ".join(words[:3]))
    if len(words) > 2:
        queries.append(" ".join(words[:2]))

    # Remove duplicates while preserving order
    seen = set()
    return [q for q in queries if not (q in seen or seen.add(q))]


def resolve_yahoo_ticker(name: str, exchange_hint: str = "") -> Tuple[Optional[str], float, Optional[str]]:
    """
    Use yfinance.Search to map a company name to a real Yahoo symbol.

    Returns (symbol, confidence, longname). Symbol is None on failure.
    """
    queries = build_name_search_queries(name)
    indian_quotes: List[Dict] = []
    matched_query = name

    for query in queries:
        try:
            results = yf.Search(query, max_results=8).quotes or []
        except Exception as e:
            warn(f"[enrich] yf.Search failed for {query!r}: {e}")
            continue
        indian_quotes = [q for q in results if str(q.get("symbol", "")).endswith((".NS", ".BO"))]
        if indian_quotes:
            matched_query = query
            if query != name:
                log(f"[enrich] yf.Search retry hit on {query!r}: {len(indian_quotes)} Indian results")
            break

    if not indian_quotes:
        return None, 0.0, None

    # Prefer .NS unless BSE is explicitly hinted
    if exchange_hint.upper() == "BSE":
        ranked = sorted(indian_quotes, key=lambda q: 0 if q["symbol"].endswith(".BO") else 1)
    else:
        ranked = sorted(indian_quotes, key=lambda q: 0 if q["symbol"].endswith(".NS") else 1)

    best_quote = None
    best_conf = 0.0
    name_lower = name.lower()

    for quote in ranked:
        longname = (quote.get("longname") or quote.get("shortname") or "").strip()
        if not longname:
            continue
        conf = difflib.SequenceMatcher(None, name_lower, longname.lower()).ratio()
        if conf > best_conf:
            best_quote = quote
            best_conf = conf

    if best_quote is None:
        # Fallback: no longname available – accept top result with blind confidence
        best_quote = ranked[0]
        best_conf = config.BLIND_MATCH_CONFIDENCE
        log(f"[enrich] blind match {name!r} → {best_quote['symbol']} (no longname in yf response)")

    elif best_conf < config.NAME_MATCH_CONFIDENCE_THRESHOLD:
        log(
            f"[enrich] name match too weak for {name!r} → {best_quote.get('longname')!r} "
            f"(confidence={best_conf:.2f}), rejecting"
        )
        return None, best_conf, best_quote.get("longname")

    return best_quote["symbol"], best_conf, best_quote.get("longname") or best_quote.get("shortname")


# ----------------------------------------------------------------------
# News and document enrichment
# ----------------------------------------------------------------------
NEWS_TOPICS = {
    "hiring": '"{name}" hiring layoffs headcount India 2024 2025',
}


def fetch_company_news(name: str) -> Dict[str, List[Dict]]:
    """Fetch hiring‑related news via Tavily."""
    out = {}
    for topic, template in NEWS_TOPICS.items():
        query = template.format(name=name)
        try:
            results = tavily_search(query, max_results=4)
        except Exception as e:
            warn(f"[enrich] Tavily FAIL {name} topic={topic}: {e}")
            out[topic] = []
            continue
        out[topic] = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content") or "")[:500],
            }
            for r in results or []
        ]
        log(f"[enrich] news OK {name} topic={topic}: {len(out[topic])} hits")
        time.sleep(0.4)
    return out


TARGET_ANNUAL_REPORT_YEARS = (
    "2024-25", "2023-24",
    "FY25", "FY24", "FY 25", "FY 24",
    "Financial Year 2025", "Financial Year 2024",
    "Year 2025", "Year 2024",
)


def find_recent_annual_report_url(annual_reports: List[Dict]) -> Optional[str]:
    """Select the most relevant annual report PDF URL (FY24/FY25 preferred)."""
    for entry in annual_reports or []:
        url = entry.get("url", "")
        if not url.lower().endswith(".pdf"):
            continue
        haystack = f"{entry.get('title', '')} {entry.get('date', '')}".lower()
        if any(yr.lower() in haystack for yr in TARGET_ANNUAL_REPORT_YEARS):
            return url
    return None


def extract_annual_report_excerpt(name: str, screener_docs: Dict) -> Optional[str]:
    """Attempt to download and extract MD&A from the best available annual report."""
    annual_reports = screener_docs.get("annual_reports") or []
    pdf_url = find_recent_annual_report_url(annual_reports)

    if not pdf_url:
        # Fallback to first .pdf in the list
        for entry in annual_reports:
            url = entry.get("url", "")
            if url.lower().endswith(".pdf"):
                pdf_url = url
                log(f"[enrich] PDF fallback annual_reports[0] {name}: {pdf_url}")
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


# ----------------------------------------------------------------------
# Core enrichment: LLM seed → unified schema
# ----------------------------------------------------------------------
def to_crore(value: Optional[float]) -> Optional[float]:
    """Convert INR to crore."""
    try:
        return round(float(value) / 1e7, 2) if value is not None else None
    except (TypeError, ValueError):
        return None


def enrich_company(company: Dict, sector_query: str = "") -> None:
    """
    Transform an LLM‑discovered company dict into the unified schema (in‑place).
    """
    name = company.get("name", "?")
    exchange_hint = (company.get("exchange") or "").strip().upper()
    log(f"[enrich] START {name}")

    # Preserve LLM seed before overwriting
    llm_seed = {
        "tech_signals": company.get("tech_signals", []) or [],
        "recent_news": company.get("recent_news", []) or [],
        "key_quotes": company.get("key_quotes", []) or [],
        "ticker_guess": company.get("ticker"),
    }
    llm_description = company.get("description", "") or ""
    llm_website = company.get("website", "") or ""
    llm_quarters = company.get("revenue_quarters", []) or []

    failed_stages = []
    errors = []

    # 1. Resolve Yahoo ticker
    yahoo_ticker, name_confidence, longname = resolve_yahoo_ticker(name, exchange_hint)
    if yahoo_ticker:
        log(f"[enrich] resolved {name} → {yahoo_ticker} (confidence={name_confidence:.2f})")
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

    # 2. yfinance financials
    yf_data = {}
    real_quarters = []
    if yahoo_ticker:
        try:
            fetcher = CompanyDataFetcher(name, yahoo_ticker=yahoo_ticker)
            yf_data = fetcher.get_financials_from_yahoo() or {}
            log(f"[enrich] yfinance OK {name}: fields={list(yf_data.keys())}")
            try:
                real_quarters = fetcher.get_quarterly_revenue()
                if real_quarters:
                    log(f"[enrich] quarterly revenue OK {name}: {len(real_quarters)} quarters")
            except Exception as qe:
                warn(f"[enrich] quarterly revenue FAIL {name}: {qe}")
        except Exception as e:
            warn(f"[enrich] yfinance FAIL {name}: {e}")
            failed_stages.append("yfinance")
            errors.append(f"yfinance: {e}")
    else:
        failed_stages.append("yfinance")

    financials = {
        "revenue_ttm_crore": to_crore(yf_data.get("revenue")),
        "market_cap_crore": to_crore(yf_data.get("market_cap")),
        "ebitda_margin": yf_data.get("ebitda_margins"),
        "profit_margin": yf_data.get("profit_margins"),
        "revenue_growth_yoy": yf_data.get("revenue_growth"),
        "earnings_growth_yoy": yf_data.get("earnings_growth"),
        "free_cashflow_crore": to_crore(yf_data.get("free_cashflow")),
        "total_debt_crore": to_crore(yf_data.get("total_debt")),
        "current_price": yf_data.get("current_price"),
        "target_mean_price": yf_data.get("target_mean_price"),
        "analyst_recommendation": yf_data.get("recommendation_key"),
        "revenue_quarters": real_quarters if real_quarters else llm_quarters,
    }

    profile = {
        "description": llm_description,
        "long_business_summary": yf_data.get("long_business_summary") or "",
        "website": yf_data.get("website") or llm_website or "",
        "employees": yf_data.get("employees"),
        "industry": yf_data.get("industry") or yf_data.get("sector") or "",
    }

    # 3. Screener.in documents
    screener_docs = {}
    if screener_symbol:
        try:
            fetcher = CompanyDataFetcher(name)
            screener_docs = fetcher.get_screener_documents(screener_symbol) or {}
            log(
                f"[enrich] screener OK {name}: "
                f"announcements={len(screener_docs.get('announcements', []))} "
                f"annual_reports={len(screener_docs.get('annual_reports', []))}"
            )
            if screener_docs.get("error"):
                warn(f"[enrich] screener {name}: {screener_docs['error']}")
                failed_stages.append("screener")
                errors.append(f"screener: {screener_docs['error']}")
        except Exception as e:
            warn(f"[enrich] screener FAIL {name}: {e}")
            failed_stages.append("screener")
            errors.append(f"screener: {e}")
    else:
        log(f"[enrich] screener skipped {name} — no resolved ticker")
        failed_stages.append("screener")

    # 4. Hiring news
    news = fetch_company_news(name)

    # 5. Annual report PDF excerpt
    annual_excerpt = extract_annual_report_excerpt(name, screener_docs)
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

    # Replace entire dict with unified schema
    company.clear()
    company.update({
        "name": name,
        "exchange": exchange_hint or "",
        "sector_query": sector_query,
        "identifiers": identifiers,
        "profile": profile,
        "financials": financials,
        "documents": documents,
        "news": news,
        "llm_seed": llm_seed,
        "provenance": {
            "researched_at": datetime.now().isoformat(),
            "failed_stages": failed_stages,
            "errors": errors,
        },
    })

    log(f"[enrich] DONE {name}")


# ----------------------------------------------------------------------
# Sector‑level research
# ----------------------------------------------------------------------
def research_sector(sector: str) -> Dict:
    """Discover and enrich companies for a single sector."""
    log(f"🔍 Researching sector: {sector}")
    agent = create_researcher_agent()
    char_limit = config.MAX_SEARCH_CHARS

    # Tavily queries (two instead of three to save calls)
    financial_query = f"Indian listed BSE NSE {sector} revenue 500-2000 crore 2024 2025"
    tech_query = f"Indian {sector} digital transformation AI technology 2024 2025"

    try:
        financial_results = tavily_search(financial_query, 6)
    except Exception as e:
        warn(f"[research] Tavily financial query FAIL for {sector!r}: {e}")
        financial_results = []
    try:
        tech_results = tavily_search(tech_query, 4)
    except Exception as e:
        warn(f"[research] Tavily tech query FAIL for {sector!r}: {e}")
        tech_results = []

    log(f"Tavily returned {len(financial_results)} + {len(tech_results)} results for '{sector}'")

    search_text = compact_search_results(financial_results, char_limit)
    tech_text = compact_search_results(tech_results, char_limit // 2)

    prompt = (
        f'List EXACTLY {config.MAX_COMPANIES_PER_SECTOR} Indian listed companies (BSE/NSE) in "{sector}" '
        f"with annual revenue ₹{config.REVENUE_MIN_CRORE}-{config.REVENUE_MAX_CRORE} Cr. "
        f"You MUST return {config.MAX_COMPANIES_PER_SECTOR} entries — if fewer perfect matches exist, "
        "include adjacent ones. Use the OFFICIAL stock symbols.\n\n"
        f"=== SEARCH ===\n{search_text}\n\n=== TECH NEWS ===\n{tech_text}\n\n"
        f"Return structured JSON with a top-level 'companies' array of length {config.MAX_COMPANIES_PER_SECTOR}. "
        "Estimate quarterly revenue from annual if needed."
    )

    response_text = call_agent(agent, prompt, sector)
    log(f"LLM returned {len(response_text)} chars for sector '{sector}'")

    # Parse JSON
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        result = json.loads(response_text[start:end]) if start != -1 and end > start else {}
    except json.JSONDecodeError as e:
        error(f"JSON parse failed for '{sector}': {e}")
        result = {}

    result["sector"] = sector
    result["researched_at"] = datetime.now().isoformat()
    companies = result.get("companies", [])
    log(f"LLM identified {len(companies)} companies in '{sector}': {[c.get('name', '?') for c in companies]}")

    # Enrich each company
    enriched_count = 0
    for company in companies:
        try:
            enrich_company(company, sector_query=sector)
            failed = company.get("provenance", {}).get("failed_stages", [])
            if "yfinance" not in failed and "name_resolution" not in failed:
                enriched_count += 1
        except Exception as e:
            error(f"Enrichment failed for {company.get('name', '?')}: {e}")
            company.setdefault("provenance", {})["researched_at"] = datetime.now().isoformat()
            company["provenance"].setdefault("failed_stages", []).append("enrichment")
            company["provenance"].setdefault("errors", []).append(str(e))

    log(f"Enriched {enriched_count}/{len(companies)} companies in sector '{sector}'")
    return result


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def run_researcher() -> Dict:
    """Discover and enrich companies across all configured sectors."""
    log("=" * 60)
    log("AGENT 1: RESEARCHER STARTING")
    log("=" * 60)

    output = {
        "run_date": datetime.now().isoformat(),
        "sectors": [research_sector(sector) for sector in config.SECTORS],
    }

    total_companies = sum(len(s.get("companies", [])) for s in output["sectors"])
    log(f"✅ RESEARCHER DONE — {total_companies} companies across {len(output['sectors'])} sectors")
    return output


if __name__ == "__main__":
    tracker.reset()
    run_researcher()
    tracker.print_summary()