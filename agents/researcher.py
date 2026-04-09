# agents/researcher.py — Token-optimized with usage tracking

import os, json, time, requests
from datetime import datetime
from dotenv import load_dotenv
from strands import Agent
from strands.models import BedrockModel
import config
from utils.token_tracker import tracker

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
    print(f"\n🔍 Researching: {sector}")
    agent = create_researcher_agent()
    limit = config.MAX_SEARCH_CHARS

    # Single combined search instead of 3 separate ones — saves 1 Tavily call
    q1 = f"Indian listed BSE NSE {sector} revenue 500-2000 crore 2024 2025"
    q2 = f"Indian {sector} digital transformation AI technology 2024 2025"
    r1 = tavily_search(q1, 6)
    r2 = tavily_search(q2, 4)

    search_text = _compact_search(r1, limit)
    tech_text = _compact_search(r2, limit // 2)

    # Shorter prompt — gives same instruction with fewer tokens
    prompt = (
        f'Find up to {config.MAX_COMPANIES_PER_SECTOR} Indian listed companies in "{sector}" '
        f"with revenue ₹{config.REVENUE_MIN_CRORE}-{config.REVENUE_MAX_CRORE} Cr.\n\n"
        f"=== SEARCH ===\n{search_text}\n\n=== TECH NEWS ===\n{tech_text}\n\n"
        "Return structured JSON. Estimate quarterly revenue from annual if needed. "
        "Only include companies you are confident about."
    )

    response = _call_agent(agent, prompt, sector)
    response_text = str(response)

    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        result = json.loads(response_text[start:end]) if start != -1 and end > start else {
            "sector": sector, "companies": [], "error": "No JSON found"}
    except json.JSONDecodeError as e:
        result = {"sector": sector, "companies": [], "error": str(e)}

    result["researched_at"] = datetime.now().isoformat()
    print(f"   ✅ Found {len(result.get('companies', []))} companies")
    return result


def run_researcher() -> dict:
    print("\n" + "=" * 60)
    print("AGENT 1: RESEARCHER STARTING")
    print("=" * 60)

    all_data = {"run_date": datetime.now().isoformat(), "sectors": []}

    for sector in config.SECTORS:
        sector_data = research_sector(sector)
        all_data["sectors"].append(sector_data)

        safe = sector.replace(" ", "_").replace("/", "_")
        path = f"data/raw/{safe}.json"
        with open(path, "w") as f:
            json.dump(sector_data, f, indent=2)
        print(f"   💾 Saved {path}")

    with open("data/raw/all_sectors.json", "w") as f:
        json.dump(all_data, f, indent=2)

    total = sum(len(s.get("companies", [])) for s in all_data["sectors"])
    print(f"\n✅ RESEARCHER DONE — {total} companies across {len(all_data['sectors'])} sectors")
    return all_data


if __name__ == "__main__":
    tracker.reset()
    run_researcher()
    tracker.print_summary()
