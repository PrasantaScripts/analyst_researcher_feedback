# main.py — Runs the full 3-agent pipeline with run-id-based S3 archival
#
# Pipeline: Researcher → Analyst → Reporter
# Artifacts: research.json, analysis.json, report.html, metrics.json
# All stored under s3://{S3_BUCKET}/runs/{run_id}/

import json, os, sys
from datetime import datetime
import config
from agents.researcher import run_researcher
from agents.analyst import run_analyst
from agents.reporter import run_reporter
from utils.token_tracker import tracker
from utils.logger import log
from utils.s3_storage import preflight_check, generate_run_id, upload_artifact


def run_pipeline(skip_research=False, skip_analysis=False):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    tracker.reset()
    start = datetime.now()
    run_id = generate_run_id()

    # Fail fast if S3 is misconfigured
    preflight_check()

    log("=" * 60)
    log(f"PROSPECTS AI PIPELINE STARTING — {run_id}")
    log("=" * 60)

    # ── RESEARCHER ─────────────────────────────────────────────────
    if skip_research and os.path.exists("data/raw/all_sectors.json"):
        log("⏭️  Skipping research — loading cached data")
        with open("data/raw/all_sectors.json") as f:
            researcher_output = json.load(f)
    else:
        log("▶️  Agent 1: Researcher")
        researcher_output = run_researcher()

    researcher_output["run_id"] = run_id
    upload_artifact(run_id, "research.json", researcher_output)

    # ── ANALYST ────────────────────────────────────────────────────
    if skip_analysis and os.path.exists("data/analyst_output.json"):
        log("⏭️  Skipping analysis — loading cached scores")
        with open("data/analyst_output.json") as f:
            analyst_output = json.load(f)
    else:
        log("▶️  Agent 2: Analyst")
        analyst_output = run_analyst(researcher_output)

    analyst_output["run_id"] = run_id
    upload_artifact(run_id, "analysis.json", analyst_output)

    # ── REPORTER ───────────────────────────────────────────────────
    log("▶️  Agent 3: Reporter")
    html = run_reporter(analyst_output)
    upload_artifact(run_id, "report.html", html)

    # ── METRICS ────────────────────────────────────────────────────
    end = datetime.now()
    duration = (end - start).total_seconds()
    metrics = _build_metrics(run_id, start, end, researcher_output, analyst_output)
    upload_artifact(run_id, "metrics.json", metrics)

    log(
        f"🏁 PIPELINE COMPLETE in {duration:.0f}s — {run_id}\n"
        f"   s3://{config.S3_BUCKET}/{config.S3_RUNS_PREFIX}/{run_id}/"
    )

    # ══════════════════════════════════════════════════════════════
    # TOKEN USAGE SUMMARY
    # ══════════════════════════════════════════════════════════════
    tracker.print_summary()

    # Also save token log locally
    _save_token_log(run_id)

    return run_id


def _build_metrics(run_id: str, start: datetime, end: datetime,
                   researcher_output: dict, analyst_output: dict) -> dict:
    """Build the metrics artifact from the token tracker and pipeline stats."""
    inp, out, total = tracker.totals()

    # Count pipeline stats from researcher output
    all_companies = []
    for s in researcher_output.get("sectors", []):
        all_companies.extend(s.get("companies", []))

    resolved = sum(
        1 for c in all_companies
        if "name_resolution" not in (c.get("provenance") or {}).get("failed_stages", [])
    )
    scored = analyst_output.get("total_companies_analysed", 0)

    return {
        "run_id": run_id,
        "started_at": start.isoformat(),
        "completed_at": end.isoformat(),
        "duration_seconds": round((end - start).total_seconds(), 1),
        "agents": tracker.by_agent(),
        "totals": {
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": total,
            "estimated_cost_usd": round(tracker.estimate_cost(), 4),
        },
        "pipeline": {
            "sectors_processed": len(researcher_output.get("sectors", [])),
            "companies_discovered": len(all_companies),
            "companies_resolved": resolved,
            "companies_scored": scored,
            "companies_skipped": len(all_companies) - resolved,
        },
        "calls": [
            {
                "agent": c.agent,
                "in": c.input_tokens,
                "out": c.output_tokens,
                "latency": c.latency_sec,
                "time": c.timestamp,
            }
            for c in tracker.calls
        ],
    }


def _save_token_log(run_id: str = ""):
    """Persist token usage as JSON for local historical tracking."""
    inp, out, total = tracker.totals()
    record = {
        "run_id": run_id,
        "run_date": datetime.now().isoformat(),
        "total_calls": len(tracker.calls),
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": total,
        "estimated_cost_usd": round(tracker.estimate_cost(), 4),
        "by_agent": tracker.by_agent(),
        "calls": [
            {"agent": c.agent, "in": c.input_tokens, "out": c.output_tokens,
             "latency": c.latency_sec, "time": c.timestamp}
            for c in tracker.calls
        ],
    }
    path = "logs/token_usage.json"
    history = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                history = json.load(f)
        except (json.JSONDecodeError, ValueError):
            history = []
    history.append(record)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"📝 Token log saved to {path}")


def lambda_handler(event, context):
    run_id = run_pipeline()
    return {
        "statusCode": 200,
        "body": json.dumps({"run_id": run_id, "message": "Pipeline complete"}),
    }


if __name__ == "__main__":
    skip_research = "--skip-research" in sys.argv or "--report-only" in sys.argv
    skip_analysis = "--report-only" in sys.argv
    run_pipeline(skip_research=skip_research, skip_analysis=skip_analysis)
