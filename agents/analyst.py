# agents/analyst.py — Token-optimized with usage tracking
#
# Two-pass scoring:
#   1) Compute deterministic features in Python (revenue growth, CXO recency,
#      concall recency, revenue band, employee band). LLM treats these as
#      ground truth and never does arithmetic.
#   2) LLM scores qualitative signals only and returns a slim JSON block.
# The LLM result is merged INTO the original company dicts under .analyst —
# the full enriched record is passed forward to the reporter, never stripped.

import json, os, time
from datetime import datetime, timedelta
from strands import Agent
from strands.models import BedrockModel
import config
from utils.token_tracker import tracker
from agents.researcher import tavily_search
from tools.company_scraper import CompanyDataFetcher


CXO_KEYWORDS = (
    "CEO", "CTO", "CXO", "CDO", "CFO",
    "Chief Executive", "Chief Technology",
    "Chief Digital", "Chief Financial",
    "Managing Director",
)


def _parse_screener_date(date_str: str):
    """Screener dates look like '12 Mar', '12 Mar 2024', '12 March 2024'.
    If year is missing, assume current year (screener lists are newest-first
    so this is acceptable for recency-window checks)."""
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


def _employee_band(n) -> str:
    if n is None:
        return "unknown"
    try:
        n = int(n)
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


def _compute_features(company: dict) -> dict:
    """Deterministic Python features computed BEFORE the LLM call.
    The LLM is told to treat these as ground truth and score qualitative
    signals only — never asked to do arithmetic on raw numbers."""
    financials = company.get("financials") or {}
    documents = company.get("documents") or {}
    profile = company.get("profile") or {}

    # Revenue growth across stored quarters (most-recent-first ordering)
    quarters = financials.get("revenue_quarters") or []
    growth_pct = None
    if len(quarters) >= 2:
        try:
            latest = float(quarters[0].get("revenue_crore") or 0)
            earliest = float(quarters[-1].get("revenue_crore") or 0)
            if earliest > 0:
                growth_pct = round(((latest - earliest) / earliest) * 100, 2)
        except (TypeError, ValueError):
            pass

    # CXO change in last 180 days (announcements parsed from screener)
    cutoff_180 = datetime.now() - timedelta(days=180)
    has_cxo = False
    for ann in documents.get("announcements") or []:
        text = f"{ann.get('title','')} {ann.get('context','')}"
        if not any(kw in text for kw in CXO_KEYWORDS):
            continue
        dt = _parse_screener_date(ann.get("date", ""))
        if dt and dt >= cutoff_180:
            has_cxo = True
            break

    # Concall in last 90 days
    cutoff_90 = datetime.now() - timedelta(days=90)
    has_concall = False
    for c in documents.get("concalls") or []:
        dt = _parse_screener_date(c.get("date", ""))
        if dt and dt >= cutoff_90:
            has_concall = True
            break

    # Revenue band (uses real yfinance number, not LLM guess)
    rev_crore = financials.get("revenue_ttm_crore")
    in_band = False
    if isinstance(rev_crore, (int, float)):
        in_band = config.REVENUE_MIN_CRORE <= rev_crore <= config.REVENUE_MAX_CRORE

    return {
        "revenue_growth_pct_4q": growth_pct,
        "has_recent_cxo_change": has_cxo,
        "has_concall_last_90d": has_concall,
        "in_revenue_target_band": in_band,
        "employee_band": _employee_band(profile.get("employees")),
    }


def create_analyst_agent():
    model = BedrockModel(model_id=config.BEDROCK_MODEL_ID, region_name=config.AWS_REGION)
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
    return Agent(model=model, system_prompt=system_prompt)


def _is_scoreable(company: dict) -> bool:
    """A company is scoreable if name resolution succeeded and we have at
    least some yfinance data. Companies where the researcher couldn't even
    find a ticker (Nippon Express = unlisted, Gati = delisted) get score=0
    immediately — no point burning LLM tokens on them."""
    failed = (company.get("provenance") or {}).get("failed_stages") or []
    return "name_resolution" not in failed


def _build_prompt_payload(companies: list) -> str:
    """Build slim prompt-only payload from the unified schema. Each entry
    carries an 'idx' matching the company's position in the input list so
    the LLM's response can be merged back by index (names can drift).
    Does NOT mutate the company dicts."""
    slim = []
    for i, c in enumerate(companies):
        financials = c.get("financials") or {}
        documents = c.get("documents") or {}
        llm_seed = c.get("llm_seed") or {}
        slim.append({
            "idx": i,
            "name": c.get("name"),
            "ticker": (c.get("identifiers") or {}).get("yahoo_ticker"),
            "industry": (c.get("profile") or {}).get("industry"),
            "computed_features": c.get("computed_features"),
            "revenue_quarters": (financials.get("revenue_quarters") or [])[:4],
            "tech_signals": (llm_seed.get("tech_signals") or [])[:4],
            "recent_news": (llm_seed.get("recent_news") or [])[:3],
            "recent_announcements": [
                {"title": a.get("title"), "date": a.get("date"), "context": a.get("context")}
                for a in (documents.get("announcements") or [])[:5]
            ],
            "hiring_news": [
                {"title": h.get("title"), "snippet": (h.get("content") or "")[:200]}
                for h in ((c.get("news") or {}).get("hiring") or [])[:2]
            ],
            "description": ((c.get("profile") or {}).get("description") or "")[:160],
        })
    return json.dumps(slim, separators=(",", ":"))


def score_sector(sector_data: dict, agent: Agent) -> dict:
    sector_name = sector_data.get("sector", "Unknown")
    companies = sector_data.get("companies", [])
    print(f"\n📊 Analysing: {sector_name} ({len(companies)} companies)")

    if not companies:
        return {
            "sector": sector_name,
            "sector_summary": "No companies found",
            "companies": [],
        }

    # ── 0) Pre-filter: separate scoreable vs unscoreable ─────────────
    # Companies where the researcher couldn't resolve a ticker (unlisted,
    # delisted, hallucinated by LLM) get score=0 immediately — no LLM call.
    scoreable = []
    for c in companies:
        c["computed_features"] = _compute_features(c)
        if _is_scoreable(c):
            scoreable.append(c)
        else:
            name = c.get("name", "?")
            failed = (c.get("provenance") or {}).get("failed_stages") or []
            reason = (
                "Company not found on BSE/NSE — likely unlisted or delisted. "
                f"Failed stages: {', '.join(failed)}"
            )
            c["analyst"] = {
                "score": 0,
                "growth_trend": "unknown",
                "top_signals": [],
                "reasoning": reason,
                "recommended_approach": "Verify listing status before outreach",
                "outreach_angle": "",
                "risk_factors": ["not_listed"],
                "scored_at": datetime.now().isoformat(),
            }
            print(f"   ⏭️  Skipped {name} (unresolvable — score=0)")

    skipped = len(companies) - len(scoreable)
    if skipped:
        print(f"   ({skipped} companies auto-scored 0 — not listed / not found)")

    if not scoreable:
        return {
            "sector": sector_name,
            "sector_summary": "No scoreable companies found",
            "companies": companies,
        }

    # ── 1) Build slim payload for the LLM ────────────────────────────
    # Each entry carries an 'idx' (position in scoreable[]) so we can
    # merge by index — name matching alone breaks when the LLM returns
    # "Transport Corporation of India" but the dict has "...Limited".
    payload = _build_prompt_payload(scoreable)

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

    t0 = time.time()
    response = agent(prompt)
    tracker.record("Analyst", response, prompt, time.time() - t0)

    response_text = str(response)
    try:
        s, e = response_text.find('{'), response_text.rfind('}') + 1
        scored = json.loads(response_text[s:e]) if s != -1 and e > s else {}
    except json.JSONDecodeError as ex:
        print(f"   ⚠️  Analyst JSON parse failed: {ex}")
        scored = {}

    # ── 2) Merge LLM scores into the original company dicts ──────────
    # Primary: match by idx (foolproof). Fallback: match by name (for
    # LLMs that ignore the idx instruction).
    by_name = {(c.get("name") or "").strip().lower(): c for c in scoreable}
    matched = 0
    for sc in scored.get("scored_companies") or []:
        # Try idx first
        idx = sc.get("idx")
        target = None
        if isinstance(idx, int) and 0 <= idx < len(scoreable):
            target = scoreable[idx]
        # Fallback: name match
        if target is None:
            key = (sc.get("name") or "").strip().lower()
            target = by_name.get(key)
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

    # Stub block for any scoreable company the LLM missed
    for c in scoreable:
        if "analyst" not in c:
            c["analyst"] = {
                "score": 0,
                "growth_trend": "unknown",
                "top_signals": [],
                "reasoning": "LLM did not return a score for this company",
                "recommended_approach": "",
                "outreach_angle": "",
                "risk_factors": [],
                "scored_at": datetime.now().isoformat(),
            }

    print(f"   ✅ Scored {matched}/{len(scoreable)} scoreable companies")

    # ── 3) Annual-report PDF recovery via Tavily ─────────────────────
    # For scoreable companies whose researcher pass couldn't pull a
    # screener AR (provenance.failed_stages has "annual_report_pdf"),
    # try ONE Tavily search. Skips unresolvable companies entirely.
    recovered = 0
    for c in scoreable:
        provenance = c.get("provenance") or {}
        failed = provenance.get("failed_stages") or []
        if "annual_report_pdf" not in failed:
            continue

        name = c.get("name", "?")
        query = f"{name} annual report 2024 2025 filetype:pdf"
        try:
            results = tavily_search(query, max_results=4)
        except Exception as e:
            print(f"   ⚠️  AR recovery Tavily FAIL {name}: {e}")
            continue

        pdf_url = next(
            (r.get("url") for r in (results or [])
             if (r.get("url") or "").lower().endswith(".pdf")),
            None,
        )
        if not pdf_url:
            continue

        excerpt = CompanyDataFetcher(name).extract_annual_report_pdf(pdf_url)
        if not excerpt:
            continue

        c.setdefault("documents", {})["annual_report_excerpt"] = excerpt
        c["provenance"]["failed_stages"] = [s for s in failed if s != "annual_report_pdf"]
        recovered += 1
        print(f"   📄 AR recovery OK {name}: {len(excerpt)} chars from {pdf_url}")
        time.sleep(0.4)

    if recovered:
        print(f"   ↻ AR recovery: {recovered} companies repaired via Tavily")

    return {
        "sector": sector_name,
        "sector_summary": scored.get("sector_summary", ""),
        "companies": companies,  # ALL companies (scored + skipped)
    }


def run_analyst(researcher_output: dict) -> dict:
    print("\n" + "=" * 60)
    print("AGENT 2: ANALYST STARTING")
    print("=" * 60)

    agent = create_analyst_agent()
    all_scored = {"analysed_at": datetime.now().isoformat(), "sectors": []}

    for sector_data in researcher_output.get("sectors", []):
        all_scored["sectors"].append(score_sector(sector_data, agent))

    # Global leaderboard — sort full records by .analyst.score, never strip
    all_companies = []
    for s in all_scored["sectors"]:
        all_companies.extend(s.get("companies", []))
    all_companies.sort(
        key=lambda x: (x.get("analyst") or {}).get("score", 0),
        reverse=True,
    )
    for i, c in enumerate(all_companies, 1):
        if "analyst" in c:
            c["analyst"]["global_rank"] = i

    all_scored["global_leaderboard"] = all_companies
    all_scored["total_companies_analysed"] = len(all_companies)

    os.makedirs("data", exist_ok=True)
    with open("data/analyst_output.json", "w") as f:
        json.dump(all_scored, f, indent=2, default=str)

    print(f"\n✅ ANALYST DONE — {len(all_companies)} companies scored")
    if all_companies:
        top = all_companies[0]
        print(
            f"   Top: {top.get('name')} "
            f"(Score: {(top.get('analyst') or {}).get('score')})"
        )
    return all_scored


if __name__ == "__main__":
    tracker.reset()
    with open("data/raw/all_sectors.json") as f:
        data = json.load(f)
    run_analyst(data)
    tracker.print_summary()
