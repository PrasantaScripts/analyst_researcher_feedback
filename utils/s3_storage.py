# utils/s3_storage.py — S3 helpers for pipeline artifact storage.
#
# Two key patterns coexist:
#
#   LEGACY (kept for backward compat):
#     s3://{S3_BUCKET}/{S3_RESEARCH_PREFIX}/all_sectors_<timestamp>.json
#
#   NEW (run-id based, all 4 artifacts under one key):
#     s3://{S3_BUCKET}/{S3_RUNS_PREFIX}/{run_id}/research.json
#     s3://{S3_BUCKET}/{S3_RUNS_PREFIX}/{run_id}/analysis.json
#     s3://{S3_BUCKET}/{S3_RUNS_PREFIX}/{run_id}/report.html
#     s3://{S3_BUCKET}/{S3_RUNS_PREFIX}/{run_id}/metrics.json
#
# Query by run_id → get all 4 artifacts back.

import json
from datetime import datetime
from typing import Optional, List, Dict

import boto3
from botocore.exceptions import ClientError, BotoCoreError

import config
from utils.logger import log, warn, error


def _client():
    """Lazy boto3 S3 client. Region comes from config.S3_REGION."""
    return boto3.client("s3", region_name=config.S3_REGION)


# Run ID

def generate_run_id() -> str:
    """Timestamp-based run ID. Sortable, human-readable, unique per second."""
    return f"run_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"


# Preflight

def preflight_check() -> None:
    """
    Verify the bucket exists and our credentials can reach it.
    Raises a clear RuntimeError on failure so callers can fail fast
    BEFORE doing 12-18 minutes of LLM/Tavily work.
    """
    bucket = config.S3_BUCKET
    if not bucket:
        raise RuntimeError("S3_BUCKET is not configured (set it in .env or config.py)")
    try:
        _client().head_bucket(Bucket=bucket)
    except (ClientError, BotoCoreError) as e:
        msg = (
            f"S3 preflight failed for bucket={bucket!r} region={config.S3_REGION!r}: {e}\n"
            f"Fix: ensure the bucket exists and the IAM user has "
            f"s3:ListBucket / s3:GetObject / s3:PutObject on arn:aws:s3:::{bucket}/*"
        )
        error(msg)
        raise RuntimeError(msg) from e
    log(f"[s3] preflight OK — bucket={bucket} region={config.S3_REGION}")


# Run-based artifact storage

VALID_ARTIFACTS = ("research.json", "analysis.json", "report.html", "metrics.json")

# Content-type map for artifact upload
_CONTENT_TYPES = {
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
}


def upload_artifact(run_id: str, artifact: str, data, content_type: str | None = None) -> str:
    """
    Upload a pipeline artifact to s3://{S3_BUCKET}/{S3_RUNS_PREFIX}/{run_id}/{artifact}.

    Args:
        run_id:       e.g. 'run_2026-04-12T10-46-07'
        artifact:     one of: research.json, analysis.json, report.html, metrics.json
        data:         dict/list for JSON artifacts, str for HTML
        content_type: override auto-detected content-type

    Returns the s3:// URI.
    """
    key = f"{config.S3_RUNS_PREFIX}/{run_id}/{artifact}"

    # Auto-detect content type from extension
    if content_type is None:
        ext = "." + artifact.rsplit(".", 1)[-1] if "." in artifact else ""
        content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")

    # Serialize
    if isinstance(data, (dict, list)):
        body = json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    elif isinstance(data, str):
        body = data.encode("utf-8")
    elif isinstance(data, bytes):
        body = data
    else:
        body = json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf-8")

    try:
        _client().put_object(
            Bucket=config.S3_BUCKET,
            Key=key,
            Body=body,
            ContentType=content_type,
        )
    except (ClientError, BotoCoreError) as e:
        error(f"[s3] upload FAILED s3://{config.S3_BUCKET}/{key}: {e}")
        raise

    uri = f"s3://{config.S3_BUCKET}/{key}"
    log(f"📤 [s3] Uploaded {len(body):,} bytes → {uri}")
    return uri


def download_artifact(run_id: str, artifact: str) -> bytes:
    """Download a single artifact for a given run. Returns raw bytes."""
    key = f"{config.S3_RUNS_PREFIX}/{run_id}/{artifact}"
    try:
        resp = _client().get_object(Bucket=config.S3_BUCKET, Key=key)
        body = resp["Body"].read()
    except (ClientError, BotoCoreError) as e:
        error(f"[s3] download FAILED s3://{config.S3_BUCKET}/{key}: {e}")
        raise
    log(f"📥 [s3] Downloaded {len(body):,} bytes from s3://{config.S3_BUCKET}/{key}")
    return body


def list_runs(limit: int = 50) -> List[str]:
    """
    List run IDs under {S3_RUNS_PREFIX}/, newest-first.
    Returns up to `limit` run_id strings like ['run_2026-04-12T10-46-07', ...].
    """
    prefix = f"{config.S3_RUNS_PREFIX}/"
    run_ids: set = set()
    try:
        paginator = _client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=config.S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                # Key format: runs/{run_id}/{artifact}
                parts = obj["Key"].removeprefix(prefix).split("/")
                if parts and parts[0].startswith("run_"):
                    run_ids.add(parts[0])
    except (ClientError, BotoCoreError) as e:
        error(f"[s3] list FAILED s3://{config.S3_BUCKET}/{prefix}: {e}")
        raise

    sorted_ids = sorted(run_ids, reverse=True)
    log(f"[s3] Found {len(sorted_ids)} run(s) under s3://{config.S3_BUCKET}/{prefix}")
    return sorted_ids[:limit]


def get_run_artifacts(run_id: str) -> Dict:
    """
    Download all artifacts for a run, returned as:
    {
        'run_id': 'run_...',
        'research': <dict>,        # parsed JSON
        'analysis': <dict>,        # parsed JSON
        'report': <str>,           # raw HTML string
        'metrics': <dict>,         # parsed JSON
    }
    Missing artifacts are None (not an error — e.g. if pipeline was interrupted).
    """
    result: Dict = {"run_id": run_id}
    artifact_map = {
        "research": "research.json",
        "analysis": "analysis.json",
        "report":   "report.html",
        "metrics":  "metrics.json",
    }
    for field, filename in artifact_map.items():
        try:
            raw = download_artifact(run_id, filename)
            if filename.endswith(".json"):
                result[field] = json.loads(raw.decode("utf-8"))
            else:
                result[field] = raw.decode("utf-8")
        except Exception as e:
            warn(f"[s3] artifact {filename} not found for {run_id}: {e}")
            result[field] = None
    return result


# Legacy helpers (backward compat)

def upload_research_output(data: Dict, key: Optional[str] = None) -> str:
    """
    LEGACY: Upload to the old s3://{S3_BUCKET}/{S3_RESEARCH_PREFIX}/ path.
    Kept for backward compatibility with existing researcher code.
    """
    if key is None:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        key = f"{config.S3_RESEARCH_PREFIX}/all_sectors_{ts}.json"

    body = json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    try:
        _client().put_object(
            Bucket=config.S3_BUCKET,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
    except (ClientError, BotoCoreError) as e:
        error(f"[s3] upload FAILED s3://{config.S3_BUCKET}/{key}: {e}")
        raise

    uri = f"s3://{config.S3_BUCKET}/{key}"
    log(f"📤 [s3] Uploaded {len(body):,} bytes → {uri}")
    return uri


def list_research_runs(limit: int = 50) -> List[Dict]:
    """LEGACY: List research runs under the old prefix."""
    prefix = f"{config.S3_RESEARCH_PREFIX}/"
    items: List[Dict] = []
    try:
        paginator = _client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=config.S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                items.append(
                    {
                        "key": obj["Key"],
                        "last_modified": obj["LastModified"].isoformat(),
                        "size_bytes": obj["Size"],
                    }
                )
    except (ClientError, BotoCoreError) as e:
        error(f"[s3] list FAILED s3://{config.S3_BUCKET}/{prefix}: {e}")
        raise

    items.sort(key=lambda r: r["last_modified"], reverse=True)
    log(f"[s3] Found {len(items)} run(s) under s3://{config.S3_BUCKET}/{prefix}")
    return items[:limit]


def download_research_output(key: Optional[str] = None) -> Dict:
    """LEGACY: Download from the old prefix."""
    if key is None:
        runs = list_research_runs(limit=1)
        if not runs:
            raise FileNotFoundError(
                f"No research runs found in s3://{config.S3_BUCKET}/{config.S3_RESEARCH_PREFIX}/"
            )
        key = runs[0]["key"]
        log(f"[s3] auto-picked latest run: {key}")

    try:
        resp = _client().get_object(Bucket=config.S3_BUCKET, Key=key)
        body = resp["Body"].read()
    except (ClientError, BotoCoreError) as e:
        error(f"[s3] download FAILED s3://{config.S3_BUCKET}/{key}: {e}")
        raise

    log(f"📥 [s3] Downloaded {len(body):,} bytes from s3://{config.S3_BUCKET}/{key}")
    return json.loads(body.decode("utf-8"))
