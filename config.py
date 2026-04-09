# config.py — Central settings (token-optimized)

from dotenv import load_dotenv
import os

load_dotenv()

# TEMP: single sector for scraper-integration test. Restore the full list
# below when re-enabling analyst+reporter.
SECTORS = ["independent software vendors"]
# SECTORS = [
#     "travel logistics and supply chain",
#     "healthcare and pharma",
#     "manufacturing",
#     "independent software vendors",
#     "export-oriented businesses",
#     "retail Services",
#     "fintech services",
#     "financial services",
# ]

REVENUE_MIN_CRORE = 500
REVENUE_MAX_CRORE = 2000
MAX_COMPANIES_PER_SECTOR = 5
QUARTERS_TO_ANALYZE = 4

BUY_SIGNALS = [
    "digital transformation", "AI adoption", "automation",
    "ERP implementation", "cloud migration", "data platform",
    "technology upgrade", "capacity expansion", "new CTO",
    "IT modernization", "process improvement", "analytics",
    "machine learning", "SaaS", "enterprise software",
]

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "qwen.qwen3-coder-30b-a3b-v1:0")

API_CALL_DELAY = 3
MAX_RETRIES = 5
BASE_RETRY_DELAY = 5

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ── Token-saving knobs ──────────────────────────────────────────────────────
MAX_SEARCH_CHARS = 2000        # cap search text fed to agent (was 3000)
MAX_OUTPUT_TOKENS = 2048       # cap per-call output
MAX_REPORT_DATA_CHARS = 6000   # cap data blob sent to reporter (was 8000)
TOP_COMPANIES_FOR_REPORT = 15  # email drafts only for top N (was 20)
