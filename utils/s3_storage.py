# utils/s3_storage.py — S3 helpers for research artifact storage.
#
# Each researcher run uploads ONE combined JSON to
#   s3://{S3_BUCKET}/{S3_RESEARCH_PREFIX}/all_sectors_<timestamp>.json
# The (currently paused) Analyst will later download via download_research_output().
#
# All public functions log via utils.logger so events show up in run_log.txt.

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


def upload_research_output(data: Dict, key: Optional[str] = None) -> str:
    """
    Upload a JSON-serializable dict to s3://{S3_BUCKET}/{key}.
    If key is None, generates: {S3_RESEARCH_PREFIX}/all_sectors_<YYYY-MM-DDTHH-MM-SS>.json
    Returns the s3:// URI.
    """
    if key is None:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        key = f"{config.S3_RESEARCH_PREFIX}/all_sectors_{ts}.json"

    body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
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
    """
    List research run objects under {S3_RESEARCH_PREFIX}/, newest-first.
    Returns up to `limit` entries: [{key, last_modified, size_bytes}, ...]
    """
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
    """
    Download a research run JSON from s3://{S3_BUCKET}/{key} and return as dict.
    If key is None, picks the latest run from list_research_runs(limit=1).
    Raises FileNotFoundError if no runs exist when key is None.
    """
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
