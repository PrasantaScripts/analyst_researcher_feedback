# agents/reporter.py — Template-based: LLM only writes email drafts, HTML is pre-built

import json, time, os
from datetime import datetime
from strands import Agent
from strands.models import BedrockModel
import config
from utils.token_tracker import tracker

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "templates", "report_template.html")


def create_email_agent():
    model = BedrockModel(model_id=config.BEDROCK_MODEL_ID, region_name=config.AWS_REGION)
    system_prompt = (
        "You write short B2B sales outreach emails for Indian IT/software prospects. "
        "Each email: subject + body under 120 words. Reference ONE specific signal. "
        "No generic openers. End with easy CTA. Return ONLY JSON array, no extra text."
    )
    return Agent(model=model, system_prompt=system_prompt)


def _batch_email_drafts(companies: list, agent: Agent) -> list:
    """Generate emails for all companies in ONE call."""
    if not companies:
        return companies

    specs = []
    for i, c in enumerate(companies):
        specs.append(
            f"{i+1}. {c.get('name')} | {c.get('sector')} | "
            f"signals: {', '.join(c.get('top_signals', [])[:2])} | "
            f"angle: {c.get('outreach_angle', 'N/A')}"
        )

    prompt = (
        "Write outreach emails for these prospects.\n\n"
        + "\n".join(specs) + "\n\n"
        'Return JSON: [{"index":1,"subject":"…","body":"…"},…]'
    )

    t0 = time.time()
    response = agent(prompt)
    tracker.record("Reporter-emails", response, prompt, time.time() - t0)

    resp_text = str(response)
    try:
        s = resp_text.find('[')
        e = resp_text.rfind(']') + 1
        emails = json.loads(resp_text[s:e]) if s != -1 and e > s else []
    except (json.JSONDecodeError, ValueError):
        emails = []

    for em in emails:
        idx = em.get("index", 0) - 1
        if 0 <= idx < len(companies):
            companies[idx]["email_draft"] = f"SUBJECT: {em.get('subject','')}\n\n{em.get('body','')}"

    for c in companies:
        if "email_draft" not in c:
            c["email_draft"] = f"SUBJECT: Quick question for {c.get('name','your team')}\n\n[Draft pending]"

    return companies


def _inject_data_into_template(companies: list) -> str:
    """Read the HTML template and inject the company JSON data."""
    template_file = TEMPLATE_PATH
    if not os.path.exists(template_file):
        template_file = os.path.join("templates", "report_template.html")

    with open(template_file, "r", encoding="utf-8") as f:
        html = f.read()

    clean = []
    for c in companies:
        clean.append({
            "name": c.get("name", ""),
            "ticker": c.get("ticker", ""),
            "sector": c.get("sector", ""),
            "score": c.get("score", 0),
            "global_rank": c.get("global_rank", 0),
            "revenue_growth_pct": c.get("revenue_growth_pct", 0),
            "growth_trend": c.get("growth_trend", ""),
            "top_signals": c.get("top_signals", []),
            "reasoning": c.get("reasoning", ""),
            "risk_factors": c.get("risk_factors", []),
            "email_draft": c.get("email_draft", ""),
            "outreach_angle": c.get("outreach_angle", ""),
            "recommended_approach": c.get("recommended_approach", ""),
            "description": c.get("description", ""),
            "website": c.get("website", ""),
        })

    data_json = json.dumps(clean, ensure_ascii=False)
    html = html.replace("/*__DATA__*/[]/*__END__*/", f"/*__DATA__*/{data_json}/*__END__*/")
    return html


def run_reporter(analyst_output: dict) -> str:
    print("\n" + "=" * 60)
    print("AGENT 3: REPORTER STARTING")
    print("=" * 60)

    leaderboard = analyst_output.get("global_leaderboard", [])
    top = leaderboard[:config.TOP_COMPANIES_FOR_REPORT]

    agent = create_email_agent()
    print(f"   ✉️  Generating email drafts for {len(top)} companies (1 batched call)...")
    top = _batch_email_drafts(top, agent)

    print("   📄 Injecting data into HTML template...")
    html = _inject_data_into_template(top)

    output_path = "output/report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ REPORTER DONE -> {output_path}")
    print(f"   (HTML template used - no LLM tokens spent on HTML generation)")
    return html


if __name__ == "__main__":
    tracker.reset()
    with open("data/analyst_output.json") as f:
        data = json.load(f)
    run_reporter(data)
    tracker.print_summary()
