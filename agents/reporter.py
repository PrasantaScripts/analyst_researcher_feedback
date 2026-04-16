"""Reporter agent – generates email drafts and injects data into an HTML template."""

import json
import os
import time
from datetime import datetime
from typing import Dict, List

from strands import Agent
from strands.models import BedrockModel

import config
from utils.token_tracker import tracker

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "templates", "report_template.html")


def create_email_agent() -> Agent:
    """Create a Bedrock agent specialised in writing B2B outreach emails."""
    model = BedrockModel(model_id=config.BEDROCK_MODEL_ID, region_name=config.AWS_REGION)
    system_prompt = (
        "You write short B2B sales outreach emails for Indian IT/software prospects. "
        "Each email: subject + body under 120 words. Reference ONE specific signal. "
        "No generic openers. End with easy CTA. Return ONLY JSON array, no extra text."
    )
    return Agent(model=model, system_prompt=system_prompt)


def build_email_specs(companies: List[Dict]) -> List[str]:
    """Create a textual description for each company to be used in the LLM prompt."""
    specs = []
    for idx, company in enumerate(companies):
        analyst = company.get("analyst") or {}
        signals = (analyst.get("top_signals") or [])[:2]
        angle = analyst.get("outreach_angle") or "N/A"
        sector = company.get("sector_query") or (company.get("profile") or {}).get("industry") or ""

        hiring_blurb = ""
        hiring_hits = (company.get("news") or {}).get("hiring") or []
        if hiring_hits:
            content = (hiring_hits[0].get("content") or "")[:150]
            if content:
                hiring_blurb = f" | hiring: {content}"

        specs.append(
            f"{idx + 1}. {company.get('name')} | {sector} | "
            f"signals: {', '.join(signals)} | "
            f"angle: {angle}{hiring_blurb}"
        )
    return specs


def generate_email_drafts(companies: List[Dict], agent: Agent) -> List[Dict]:
    """Call the LLM once to generate email drafts for all companies."""
    if not companies:
        return companies

    specs = build_email_specs(companies)
    prompt = (
        "Write outreach emails for these prospects.\n\n"
        + "\n".join(specs)
        + "\n\n"
        'Return JSON: [{"index":1,"subject":"…","body":"…"},…]'
    )

    start = time.time()
    response = agent(prompt)
    tracker.record("Reporter-emails", response, prompt, time.time() - start)

    # Parse LLM response
    response_text = str(response)
    try:
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        emails = json.loads(response_text[json_start:json_end]) if json_start != -1 and json_end > json_start else []
    except (json.JSONDecodeError, ValueError):
        emails = []

    # Attach drafts to companies
    for email_obj in emails:
        idx = email_obj.get("index", 0) - 1
        if 0 <= idx < len(companies):
            companies[idx]["outreach"] = {
                "email_subject": email_obj.get("subject", ""),
                "email_body": email_obj.get("body", ""),
                "drafted_at": datetime.now().isoformat(),
            }

    # Fallback for any missing drafts
    for company in companies:
        if "outreach" not in company:
            company["outreach"] = {
                "email_subject": f"Quick question for {company.get('name', 'your team')}",
                "email_body": "[Draft pending]",
                "drafted_at": datetime.now().isoformat(),
                "_fallback": True,
            }

    return companies


def sanitize_company_data(company: Dict) -> Dict:
    """Create a safe copy of company data with truncated annual report excerpts."""
    # Deep copy via JSON round‑trip
    copy = json.loads(json.dumps(company, default=str))
    docs = copy.get("documents")
    if isinstance(docs, dict):
        excerpt = docs.get("annual_report_excerpt")
        if isinstance(excerpt, str) and len(excerpt) > 500:
            docs["annual_report_excerpt"] = excerpt[:500]
    return copy


def inject_into_html_template(companies: List[Dict]) -> str:
    """Load the HTML template and embed the company data as a JSON blob."""
    template_path = TEMPLATE_PATH
    if not os.path.exists(template_path):
        template_path = os.path.join("templates", "report_template.html")

    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    safe_companies = [sanitize_company_data(c) for c in companies]
    data_json = json.dumps(safe_companies, ensure_ascii=False, default=str)
    html = html.replace("/*__DATA__*/[]/*__END__*/", f"/*__DATA__*/{data_json}/*__END__*/")
    return html


def run_reporter(analyst_output: Dict) -> str:
    """Generate the final HTML report with email drafts for the top companies."""
    print("\n" + "=" * 60)
    print("AGENT 3: REPORTER STARTING")
    print("=" * 60)

    leaderboard = analyst_output.get("global_leaderboard", [])
    top_companies = leaderboard[: config.TOP_COMPANIES_FOR_REPORT]

    email_agent = create_email_agent()
    print(f"   ✉️  Generating email drafts for {len(top_companies)} companies (1 batched call)...")
    top_companies = generate_email_drafts(top_companies, email_agent)

    print("   📄 Injecting data into HTML template...")
    html_report = inject_into_html_template(top_companies)

    print("\nREPORTER DONE — report uploaded to S3 by pipeline")
    return html_report


if __name__ == "__main__":
    from utils.s3_storage import get_run_artifacts, list_runs

    tracker.reset()
    runs = list_runs(limit=1)
    if not runs:
        raise SystemExit("No S3 runs found — run the full pipeline first")
    run_reporter(get_run_artifacts(runs[0])["analysis"])
    tracker.print_summary()