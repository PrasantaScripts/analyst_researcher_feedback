# agents/reporter.py — Template-based: LLM only writes email drafts, HTML is pre-built
#
# Reads the unified schema (analyst block, news.hiring, profile, etc.).
# Passes the FULL company record into the HTML template — the template
# is responsible for hiding fields it doesn't render. Only safety
# truncation is annual_report_excerpt → 500 chars.

import json, os, time
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
    """Generate emails for all companies in ONE call.
    Reads top_signals + outreach_angle from .analyst (new schema) and
    enriches each spec with the first hiring news snippet for extra context."""
    if not companies:
        return companies

    specs = []
    for i, c in enumerate(companies):
        analyst = c.get("analyst") or {}
        signals = (analyst.get("top_signals") or [])[:2]
        angle = analyst.get("outreach_angle") or "N/A"
        sector = c.get("sector_query") or (c.get("profile") or {}).get("industry") or ""

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
            companies[idx]["outreach"] = {
                "email_subject": em.get("subject", ""),
                "email_body": em.get("body", ""),
                "drafted_at": datetime.now().isoformat(),
            }

    for c in companies:
        if "outreach" not in c:
            c["outreach"] = {
                "email_subject": f"Quick question for {c.get('name','your team')}",
                "email_body": "[Draft pending]",
                "drafted_at": datetime.now().isoformat(),
                "_fallback": True,
            }

    return companies


def _inject_data_into_template(companies: list) -> str:
    """Inject the FULL company records into the template — no field whitelist.
    Template hides what it doesn't render. Only safety truncation:
    documents.annual_report_excerpt is capped at 500 chars to keep the
    inline JSON blob from blowing up the HTML payload."""
    template_file = TEMPLATE_PATH
    if not os.path.exists(template_file):
        template_file = os.path.join("templates", "report_template.html")

    with open(template_file, "r", encoding="utf-8") as f:
        html = f.read()

    safe = []
    for c in companies:
        # Deep-ish copy via JSON round-trip; default=str handles datetimes
        copy = json.loads(json.dumps(c, default=str))
        docs = copy.get("documents")
        if isinstance(docs, dict):
            excerpt = docs.get("annual_report_excerpt")
            if isinstance(excerpt, str) and len(excerpt) > 500:
                docs["annual_report_excerpt"] = excerpt[:500]
        safe.append(copy)

    data_json = json.dumps(safe, ensure_ascii=False, default=str)
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

    os.makedirs("output", exist_ok=True)
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
