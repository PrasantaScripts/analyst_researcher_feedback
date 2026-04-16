"""Main pipeline orchestrator with run‑id based S3 archival."""

import json
import os
from datetime import datetime
from typing import Dict
import config
from agents.researcher import run_researcher
from agents.analyst import run_analyst
from agents.reporter import run_reporter
from utils.token_tracker import tracker
from utils.logger import log
from utils.s3_storage import preflight_check, generate_run_id, upload_artifact


def build_pipeline_metrics(
    run_id: str,
    start_time: datetime,
    end_time: datetime,
    researcher_output: Dict,
    analyst_output: Dict,
) -> Dict:
    """Construct the metrics artifact from token tracker and pipeline stats."""
    total_input, total_output, total = tracker.totals()

    all_companies = []
    for sector in researcher_output.get("sectors", []):
        all_companies.extend(sector.get("companies", []))

    resolved_count = sum(
        1 for company in all_companies
        if "name_resolution" not in (company.get("provenance") or {}).get("failed_stages", [])
    )

    return {
        "run_id": run_id,
        "started_at": start_time.isoformat(),
        "completed_at": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 1),
        "agents": tracker.by_agent(),
        "totals": {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total,
            "estimated_cost_usd": round(tracker.estimate_cost(), 4),
        },
        "pipeline": {
            "sectors_processed": len(researcher_output.get("sectors", [])),
            "companies_discovered": len(all_companies),
            "companies_resolved": resolved_count,
            "companies_scored": analyst_output.get("total_companies_analysed", 0),
            "companies_skipped": len(all_companies) - resolved_count,
        },
        "calls": [
            {
                "agent": call.agent,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "latency": call.latency_sec,
                "time": call.timestamp,
            }
            for call in tracker.calls
        ],
    }


def save_token_usage_log(run_id: str) -> None:
    """Append token usage to logs/token_usage.json for local tracking."""
    total_input, total_output, total = tracker.totals()
    record = {
        "run_id": run_id,
        "run_date": datetime.now().isoformat(),
        "total_calls": len(tracker.calls),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total,
        "estimated_cost_usd": round(tracker.estimate_cost(), 4),
        "by_agent": tracker.by_agent(),
        "calls": [
            {
                "agent": call.agent,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "latency": call.latency_sec,
                "time": call.timestamp,
            }
            for call in tracker.calls
        ],
    }

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/token_usage.json"
    history = []
    if os.path.exists(log_path):
        try:
            with open(log_path, encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, ValueError):
            history = []
    history.append(record)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Token log saved to {log_path}")


def run_pipeline() -> str:
    """Execute the full three‑agent pipeline and upload artifacts to S3."""
    os.makedirs("logs", exist_ok=True)

    tracker.reset()
    start_time = datetime.now()
    run_id = generate_run_id()

    # Fail fast if S3 is misconfigured
    preflight_check()

    log("=" * 60)
    log(f"PROSPECTS AI PIPELINE STARTING — {run_id}")
    log("=" * 60)

    log("Agent 1: Researcher")
    researcher_output = run_researcher()
    researcher_output["run_id"] = run_id
    upload_artifact(run_id, "research.json", researcher_output)

    log("Agent 2: Analyst")
    analyst_output = run_analyst(researcher_output)
    analyst_output["run_id"] = run_id
    upload_artifact(run_id, "analysis.json", analyst_output)

    log("Agent 3: Reporter")
    report_html = run_reporter(analyst_output)
    upload_artifact(run_id, "report.html", report_html)

    end_time = datetime.now()
    duration_sec = (end_time - start_time).total_seconds()

    metrics = build_pipeline_metrics(run_id, start_time, end_time, researcher_output, analyst_output)
    upload_artifact(run_id, "metrics.json", metrics)

    log(
        f"PIPELINE COMPLETE in {duration_sec:.0f}s — {run_id}\n"
        f"   s3://{config.S3_BUCKET}/{config.S3_RUNS_PREFIX}/{run_id}/"
    )

    tracker.print_summary()
    save_token_usage_log(run_id)

    return run_id


def lambda_handler(event, context):
    """AWS Lambda entry point."""
    run_id = run_pipeline()
    return {
        "statusCode": 200,
        "body": json.dumps({"run_id": run_id, "message": "Pipeline complete"}),
    }


if __name__ == "__main__":
    run_pipeline()