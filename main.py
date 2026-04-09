# main.py — Runs the research pipeline with full token usage reporting
#
# NOTE: Analyst + Reporter are TEMPORARILY paused while we lock down the
# Researcher + company_scraper integration. Re-enable by uncommenting the
# imports and the two blocks marked "ANALYST" / "REPORTER" below.

import json, os, sys
from datetime import datetime
import config
from agents.researcher import run_researcher
# from agents.analyst import run_analyst   # paused
# from agents.reporter import run_reporter  # paused
from utils.token_tracker import tracker
from utils.logger import log
from utils.s3_storage import preflight_check


def run_pipeline(skip_research=False, skip_analysis=False):
    os.makedirs("logs", exist_ok=True)
    tracker.reset()
    start = datetime.now()

    # Fail fast if S3 is misconfigured — don't burn 12-18 min of LLM/Tavily calls
    # only to discover at the end that we can't archive the result.
    preflight_check()

    log("=" * 60)
    log("PROSPECTS AI PIPELINE STARTING")
    log("=" * 60)

    # ── RESEARCHER ─────────────────────────────────────────────────
    if skip_research and os.path.exists("data/raw/all_sectors.json"):
        log("⏭️  Skipping research — loading cached data")
        with open("data/raw/all_sectors.json") as f:
            researcher_output = json.load(f)
    else:
        log("▶️  Agent 1: Researcher")
        researcher_output = run_researcher()

    # ── ANALYST ────────────────────────────────────────────────────
    # PAUSED — re-enable when researcher output is solid.
    # if skip_analysis and os.path.exists("data/analyst_output.json"):
    #     log("⏭️  Skipping analysis — loading cached scores")
    #     with open("data/analyst_output.json") as f:
    #         analyst_output = json.load(f)
    # else:
    #     log("▶️  Agent 2: Analyst")
    #     analyst_output = run_analyst(researcher_output)

    # ── REPORTER ───────────────────────────────────────────────────
    # PAUSED — depends on analyst output.
    # log("▶️  Agent 3: Reporter")
    # run_reporter(analyst_output)

    duration = (datetime.now() - start).total_seconds()
    log(
        f"🏁 RESEARCH-ONLY RUN COMPLETE in {duration:.0f}s → "
        f"s3://{config.S3_BUCKET}/{config.S3_RESEARCH_PREFIX}/"
    )

    # ══════════════════════════════════════════════════════════════
    # TOKEN USAGE SUMMARY — printed after everything completes
    # ══════════════════════════════════════════════════════════════
    tracker.print_summary()

    # Also save token log to file
    _save_token_log()

    return "output/report.html"


def _save_token_log():
    """Persist token usage as JSON for historical tracking."""
    inp, out, total = tracker.totals()
    record = {
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
    # Append to history
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
    # TODO: reporter is currently disabled — return string is stale until
    # the analyst+reporter blocks above are re-enabled.
    report_path = run_pipeline()
    return {"statusCode": 200, "body": "Report generated"}


if __name__ == "__main__":
    skip_research = "--skip-research" in sys.argv or "--report-only" in sys.argv
    skip_analysis = "--report-only" in sys.argv
    run_pipeline(skip_research=skip_research, skip_analysis=skip_analysis)
