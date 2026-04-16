"""Analyst agent: two‑pass scoring – deterministic Python features + LLM qualitative signals."""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from strands import Agent
from strands.models import BedrockModel

import config
from utils.token_tracker import tracker
from agents.researcher import tavily_search
from tools.company_scraper import CompanyDataFetcher

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
CXO_KEYWORDS = (
    "CEO",
    "CTO",
    "CXO",
    "CDO",
    "CFO",
    "Chief Executive",
    "Chief Technology",
    "Chief Digital",
    "Chief Financial",
    "Managing Director",
)


# ----------------------------------------------------------------------
# Date parsing helper
# ----------------------------------------------------------------------
def parse_screener_date(date_str: str) -> Optional[datetime]:
    """Parse a Screener date string into a datetime object.

    Formats supported: '12 Mar', '12 Mar 2024', '12 March 2024'.
    Missing year defaults to the current year.
    """
    if not date_str:
        return None

    s = date_str.strip()
    for fmt in ("%d %b %Y", "%d %B %Y", "%d %b", "%d %B"):
        try:
            dt = datetime.strptime(s, fmt)
            if "%Y" not in fmt:
                dt = dt.replace(year=datetime.now().year)
            return dt
        except ValueError:
            continue
    return None


# ----------------------------------------------------------------------
# Deterministic feature computation (Python ground truth)
# ----------------------------------------------------------------------
def employee_count_band(count: Optional[int]) -> str:
    """Categorise employee count into a human‑readable band."""
    if count is None:
        return "unknown"
    try:
        n = int(count)
    except (TypeError, ValueError):
        return "unknown"

    if n < 1000:
        return "<1k"
    if n < 5000:
        return "1k-5k"
    if n < 10000:
        return "5k-10k"
    if n < 50000:
        return "10k-50k"
    return ">50k"


def compute_revenue_growth_4q(revenue_quarters: List[Dict]) -> Optional[float]:
    """Calculate revenue growth percentage over the last 4 available quarters."""
    if len(revenue_quarters) < 2:
        return None
    try:
        latest = float(revenue_quarters[0].get("revenue_crore", 0))
        earliest = float(revenue_quarters[-1].get("revenue_crore", 0))
        if earliest <= 0:
            return None
        return round(((latest - earliest) / earliest) * 100, 2)
    except (TypeError, ValueError):
        return None


def has_recent_cxo_change(announcements: List[Dict]) -> bool:
    """Check if any CXO‑related announcement occurred within the configured window."""
    cutoff = datetime.now() - timedelta(days=config.CXO_CHANGE_WINDOW_DAYS)
    for ann in announcements:
        text = f"{ann.get('title', '')} {ann.get('context', '')}"
        if not any(keyword in text for keyword in CXO_KEYWORDS):
            continue
        dt = parse_screener_date(ann.get("date", ""))
        if dt and dt >= cutoff:
            return True
    return False


def has_recent_concall(concalls: List[Dict]) -> bool:
    """Check if any concall occurred within the configured recency window."""
    cutoff = datetime.now() - timedelta(days=config.CONCALL_RECENCY_DAYS)
    for call in concalls:
        dt = parse_screener_date(call.get("date", ""))
        if dt and dt >= cutoff:
            return True
    return False


def is_in_revenue_band(revenue_ttm_crore: Optional[float]) -> bool:
    """Verify if TTM revenue falls inside the configured target band."""
    if not isinstance(revenue_ttm_crore, (int, float)):
        return False
    return config.REVENUE_MIN_CRORE <= revenue_ttm_crore <= config.REVENUE_MAX_CRORE


def compute_deterministic_features(company: Dict) -> Dict:
    """Calculate all pre‑LLM, ground‑truth features from the company record."""
    financials = company.get("financials") or {}
    documents = company.get("documents") or {}
    profile = company.get("profile") or {}

    revenue_quarters = financials.get("revenue_quarters") or []
    announcements = documents.get("announcements") or []
    concalls = documents.get("concalls") or []

    return {
        "revenue_growth_pct_4q": compute_revenue_growth_4q(revenue_quarters),
        "has_recent_cxo_change": has_recent_cxo_change(announcements),
        "has_concall_last_90d": has_recent_concall(concalls),
        "in_revenue_target_band": is_in_revenue_band(
            financials.get("revenue_ttm_crore")
        ),
        "employee_band": employee_count_band(profile.get("employees")),
    }


# ----------------------------------------------------------------------
# Agent factory
# ----------------------------------------------------------------------
def create_analyst_agent() -> Agent:
    """Create a Bedrock‑backed agent with the analyst scoring system prompt."""
    signals_csv = ", ".join(config.BUY_SIGNALS[:10])
    system_prompt = (
        "You are a B2B sales intelligence analyst scoring Indian companies on likelihood "
        "to buy IT/software/AI services in 6-12 months.\n"
        "Scoring (100 pts): Revenue Growth 35, Tech Signals 30, Strategic Moves 20, Recency 15.\n"
        f"Signal keywords: {signals_csv}\n"
        "computed_features are pre-calculated Python values, treat them as ground truth. "
        "Score qualitative signals only.\n"
        "Return ONLY valid JSON, no extra text."
    )
    return Agent(
        model=BedrockModel(
            model_id=config.BEDROCK_MODEL_ID, region_name=config.AWS_REGION
        ),
        system_prompt=system_prompt,
    )


# ----------------------------------------------------------------------
# Scoring eligibility
# ----------------------------------------------------------------------
def is_company_scoreable(company: Dict) -> bool:
    """Determine if the company is listed and can be scored."""
    failed = (company.get("provenance") or {}).get("failed_stages") or []
    return "name_resolution" not in failed


def mark_company_unscoreable(company: Dict) -> None:
    """Assign a zero score and explain why the company cannot be scored."""
    failed_stages = (company.get("provenance") or {}).get("failed_stages") or []
    company["analyst"] = {
        "score": 0,
        "growth_trend": "unknown",
        "top_signals": [],
        "reasoning": (
            "Company not found on BSE/NSE — likely unlisted or delisted. "
            f"Failed stages: {', '.join(failed_stages)}"
        ),
        "recommended_approach": "Verify listing status before outreach",
        "outreach_angle": "",
        "risk_factors": ["not_listed"],
        "scored_at": datetime.now().isoformat(),
    }


# ----------------------------------------------------------------------
# LLM prompt construction
# ----------------------------------------------------------------------
def build_scoring_payload(scoreable_companies: List[Dict]) -> str:
    """Convert a list of company dicts into a compact JSON payload for the LLM."""
    slim = []
    for idx, company in enumerate(scoreable_companies):
        financials = company.get("financials") or {}
        documents = company.get("documents") or {}
        llm_seed = company.get("llm_seed") or {}
        profile = company.get("profile") or {}

        slim.append(
            {
                "idx": idx,
                "name": company.get("name"),
                "ticker": (company.get("identifiers") or {}).get("yahoo_ticker"),
                "industry": profile.get("industry"),
                "computed_features": company.get("computed_features"),
                "revenue_quarters": (financials.get("revenue_quarters") or [])[:4],
                "tech_signals": (llm_seed.get("tech_signals") or [])[:4],
                "recent_news": (llm_seed.get("recent_news") or [])[:3],
                "recent_announcements": [
                    {
                        "title": a.get("title"),
                        "date": a.get("date"),
                        "context": a.get("context"),
                    }
                    for a in (documents.get("announcements") or [])[:5]
                ],
                "hiring_news": [
                    {"title": h.get("title"), "snippet": (h.get("content") or "")[:200]}
                    for h in ((company.get("news") or {}).get("hiring") or [])[:2]
                ],
                "description": (profile.get("description") or "")[:160],
            }
        )
    return json.dumps(slim, separators=(",", ":"))


def call_llm_for_scoring(scoreable: List[Dict], sector_name: str, agent: Agent) -> Dict:
    """Invoke the LLM with the scoring prompt and parse the JSON response."""
    payload = build_scoring_payload(scoreable)
    prompt = (
        f'Score these {len(scoreable)} "{sector_name}" companies on IT/software buy-likelihood.\n\n'
        "computed_features are pre-calculated Python ground truth — do NOT recompute, "
        "use them directly when reasoning about growth, CXO changes, concalls, revenue band.\n\n"
        f"DATA:\n{payload}\n\n"
        'Return JSON: {"sector":"…","sector_summary":"…",'
        '"scored_companies":[{"idx":0,"name":"…","score":85,"growth_trend":"growing",'
        '"top_signals":["…"],"reasoning":"…","recommended_approach":"…",'
        '"outreach_angle":"…","risk_factors":["…"]}]}\n'
        "IMPORTANT: include the 'idx' field from the input so scores can be matched back."
    )

    start_time = time.time()
    response = agent(prompt)
    tracker.record("Analyst", response, prompt, time.time() - start_time)

    response_text = str(response)
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response_text[json_start:json_end])
    except json.JSONDecodeError as ex:
        print(f"   Analyst JSON parse failed: {ex}")
    return {}


# ----------------------------------------------------------------------
# Merge LLM scores back into company records
# ----------------------------------------------------------------------
def merge_llm_scores(scored_response: Dict, scoreable_companies: List[Dict]) -> int:
    """Write LLM scores into the original company dicts, matching by index or name."""
    by_name = {(c.get("name") or "").strip().lower(): c for c in scoreable_companies}
    matched = 0

    for sc in scored_response.get("scored_companies") or []:
        idx = sc.get("idx")
        target = None
        if isinstance(idx, int) and 0 <= idx < len(scoreable_companies):
            target = scoreable_companies[idx]
        if target is None:
            target = by_name.get((sc.get("name") or "").strip().lower())
        if target is None:
            continue

        target["analyst"] = {
            "score": sc.get("score", 0),
            "growth_trend": sc.get("growth_trend", ""),
            "top_signals": sc.get("top_signals", []) or [],
            "reasoning": sc.get("reasoning", ""),
            "recommended_approach": sc.get("recommended_approach", ""),
            "outreach_angle": sc.get("outreach_angle", ""),
            "risk_factors": sc.get("risk_factors", []) or [],
            "scored_at": datetime.now().isoformat(),
        }
        matched += 1

    # Stub any company the LLM omitted
    for company in scoreable_companies:
        if "analyst" not in company:
            company["analyst"] = {
                "score": 0,
                "growth_trend": "unknown",
                "top_signals": [],
                "reasoning": "LLM did not return a score for this company",
                "recommended_approach": "",
                "outreach_angle": "",
                "risk_factors": [],
                "scored_at": datetime.now().isoformat(),
            }
    return matched


# ----------------------------------------------------------------------
# Annual report recovery via Tavily
# ----------------------------------------------------------------------
def recover_missing_annual_reports(scoreable: List[Dict]) -> int:
    """Try to find annual report PDFs for companies where Screener failed."""
    recovered = 0
    for company in scoreable:
        provenance = company.get("provenance") or {}
        failed_stages = provenance.get("failed_stages") or []
        if "annual_report_pdf" not in failed_stages:
            continue

        name = company.get("name", "?")
        query = f"{name} annual report 2024 2025 filetype:pdf"
        try:
            results = tavily_search(query, max_results=4)
        except Exception as e:
            print(f"   AR recovery Tavily FAIL {name}: {e}")
            continue

        pdf_url = next(
            (
                r.get("url")
                for r in (results or [])
                if (r.get("url") or "").lower().endswith(".pdf")
            ),
            None,
        )
        if not pdf_url:
            continue

        excerpt = CompanyDataFetcher(name).extract_annual_report_pdf(pdf_url)
        if not excerpt:
            continue

        company.setdefault("documents", {})["annual_report_excerpt"] = excerpt
        company["provenance"]["failed_stages"] = [
            stage for stage in failed_stages if stage != "annual_report_pdf"
        ]
        recovered += 1
        print(f"   AR recovery OK {name}: {len(excerpt)} chars from {pdf_url}")
        time.sleep(0.4)

    return recovered


# ----------------------------------------------------------------------
# Sector‑level scoring orchestrator
# ----------------------------------------------------------------------
def partition_and_score(sector_data: Dict, agent: Agent) -> Tuple[Dict, int, int]:
    """Split companies into scoreable/unscoreable, compute features, and run LLM scoring.

    Returns:
        scored_response (dict) – parsed LLM output
        scoreable_count (int)
        skipped_count (int)
    """
    sector_name = sector_data.get("sector", "Unknown")
    companies = sector_data.get("companies", [])

    scoreable = []
    skipped = 0
    for company in companies:
        company["computed_features"] = compute_deterministic_features(company)
        if is_company_scoreable(company):
            scoreable.append(company)
        else:
            mark_company_unscoreable(company)
            name = company.get("name", "?")
            print(f"   Skipped {name} (unresolvable — score=0)")
            skipped += 1

    if not scoreable:
        return {}, 0, skipped

    scored_response = call_llm_for_scoring(scoreable, sector_name, agent)
    matched = merge_llm_scores(scored_response, scoreable)
    print(f"   Scored {matched}/{len(scoreable)} scoreable companies")

    recovered = recover_missing_annual_reports(scoreable)
    if recovered:
        print(f"   AR recovery: {recovered} companies repaired via Tavily")

    return scored_response, len(scoreable), skipped


def score_single_sector(sector_data: Dict, agent: Agent) -> Dict:
    """Process one sector: score all companies and return an enriched sector dict."""
    sector_name = sector_data.get("sector", "Unknown")
    companies = sector_data.get("companies", [])
    print(f"\nAnalysing: {sector_name} ({len(companies)} companies)")

    if not companies:
        return {
            "sector": sector_name,
            "sector_summary": "No companies found",
            "companies": [],
        }

    scored_response, _, _ = partition_and_score(sector_data, agent)

    return {
        "sector": sector_name,
        "sector_summary": scored_response.get("sector_summary", ""),
        "companies": companies,  # all companies, both scored and skipped
    }


def add_global_ranking(analyst_output: Dict) -> None:
    """Add a 'global_rank' field to every scored company, sorted by analyst score."""
    all_companies = []
    for sector in analyst_output["sectors"]:
        all_companies.extend(sector.get("companies", []))

    all_companies.sort(
        key=lambda c: (c.get("analyst") or {}).get("score", 0),
        reverse=True,
    )
    for rank, company in enumerate(all_companies, start=1):
        if "analyst" in company:
            company["analyst"]["global_rank"] = rank

    analyst_output["global_leaderboard"] = all_companies
    analyst_output["total_companies_analysed"] = len(all_companies)


def run_analyst(researcher_output: Dict) -> Dict:
    """Entry point: score every sector from the researcher output."""
    print("\n" + "=" * 60)
    print("AGENT 2: ANALYST STARTING")
    print("=" * 60)

    agent = create_analyst_agent()
    analyst_output = {
        "analysed_at": datetime.now().isoformat(),
        "sectors": [],
    }

    for sector_data in researcher_output.get("sectors", []):
        analyst_output["sectors"].append(score_single_sector(sector_data, agent))

    add_global_ranking(analyst_output)

    total = analyst_output["total_companies_analysed"]
    print(f"\n✅ ANALYST DONE — {total} companies scored")
    if total:
        top_company = analyst_output["global_leaderboard"][0]
        print(
            f"   Top: {top_company.get('name')} "
            f"(Score: {(top_company.get('analyst') or {}).get('score')})"
        )
    return analyst_output


# ----------------------------------------------------------------------
# CLI test entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from utils.s3_storage import get_run_artifacts, list_runs

    tracker.reset()
    runs = list_runs(limit=1)
    if not runs:
        raise SystemExit("No S3 runs found — run the full pipeline first")

    run_analyst(get_run_artifacts(runs[0])["research"])
    tracker.print_summary()
