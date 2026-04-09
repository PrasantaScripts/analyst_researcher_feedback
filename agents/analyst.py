# agents/analyst.py — Token-optimized with usage tracking

import json, time
from datetime import datetime
from strands import Agent
from strands.models import BedrockModel
import config
from utils.token_tracker import tracker


def create_analyst_agent():
    model = BedrockModel(model_id=config.BEDROCK_MODEL_ID, region_name=config.AWS_REGION)

    # Compact system prompt — same rubric, ~45% fewer tokens
    signals_csv = ", ".join(config.BUY_SIGNALS[:10])
    system_prompt = (
        "You are a B2B sales intelligence analyst scoring Indian companies on likelihood "
        "to buy IT/software/AI services in 6-12 months.\n"
        "Scoring (100 pts): Revenue Growth 35, Tech Signals 30, Strategic Moves 20, Recency 15.\n"
        f"Signal keywords: {signals_csv}\n"
        "Return ONLY valid JSON, no extra text."
    )
    return Agent(model=model, system_prompt=system_prompt)


def _compact_companies(companies: list) -> str:
    """Strip verbose fields before sending to analyst — saves input tokens."""
    slim = []
    for c in companies:
        slim.append({
            "name": c.get("name"),
            "ticker": c.get("ticker"),
            "revenue_quarters": c.get("revenue_quarters", [])[-4:],  # last 4 only
            "recent_news": c.get("recent_news", [])[:3],
            "tech_signals": c.get("tech_signals", [])[:4],
            "description": c.get("description", "")[:120],
        })
    return json.dumps(slim, separators=(",", ":"))  # no whitespace


def score_sector(sector_data: dict, agent: Agent) -> dict:
    sector_name = sector_data.get("sector", "Unknown")
    companies = sector_data.get("companies", [])
    print(f"\n📊 Analysing: {sector_name} ({len(companies)} companies)")

    if not companies:
        return {"sector": sector_name, "scored_companies": [], "sector_insights": "No companies found"}

    compact = _compact_companies(companies)

    prompt = (
        f'Score these {len(companies)} "{sector_name}" companies on IT/software buy-likelihood.\n\n'
        f"DATA:\n{compact}\n\n"
        f'Return JSON: {{"sector":"{sector_name}","sector_summary":"…",'
        '"scored_companies":[{"name":"…","ticker":"…","sector":"…","score":85,"rank":1,'
        '"growth_trend":"growing","top_signals":["…"],"reasoning":"…",'
        '"recommended_approach":"…","outreach_angle":"…","risk_factors":["…"],'
        '"revenue_growth_pct":23.5,"description":"…","website":"…"}]}}'
    )

    t0 = time.time()
    response = agent(prompt)
    tracker.record("Analyst", response, prompt, time.time() - t0)

    response_text = str(response)
    try:
        s, e = response_text.find('{'), response_text.rfind('}') + 1
        result = json.loads(response_text[s:e]) if s != -1 and e > s else {
            "sector": sector_name, "scored_companies": [], "error": "No JSON"}
    except json.JSONDecodeError as ex:
        result = {"sector": sector_name, "scored_companies": [], "error": str(ex)}

    print(f"   ✅ Scored {len(result.get('scored_companies', []))} companies")
    return result


def run_analyst(researcher_output: dict) -> dict:
    print("\n" + "=" * 60)
    print("AGENT 2: ANALYST STARTING")
    print("=" * 60)

    agent = create_analyst_agent()
    all_scored = {"analysed_at": datetime.now().isoformat(), "sectors": []}

    for sector_data in researcher_output.get("sectors", []):
        all_scored["sectors"].append(score_sector(sector_data, agent))

    # Global leaderboard
    all_companies = []
    for s in all_scored["sectors"]:
        all_companies.extend(s.get("scored_companies", []))
    all_companies.sort(key=lambda x: x.get("score", 0), reverse=True)
    for i, c in enumerate(all_companies, 1):
        c["global_rank"] = i

    all_scored["global_leaderboard"] = all_companies
    all_scored["total_companies_analysed"] = len(all_companies)

    with open("data/analyst_output.json", "w") as f:
        json.dump(all_scored, f, indent=2)

    print(f"\n✅ ANALYST DONE — {len(all_companies)} companies scored")
    if all_companies:
        print(f"   Top: {all_companies[0].get('name')} (Score: {all_companies[0].get('score')})")
    return all_scored


if __name__ == "__main__":
    tracker.reset()
    with open("data/raw/all_sectors.json") as f:
        data = json.load(f)
    run_analyst(data)
    tracker.print_summary()
