# config.py — Central settings (token-optimized)

from dotenv import load_dotenv
import os

load_dotenv()

SECTORS = [
    "travel logistics and supply chain",
    "healthcare and pharma",
    "manufacturing",
    "independent software vendors",
    "export-oriented businesses",
    "retail Services",
    "fintech services",
    "financial services",
]

REVENUE_MIN_CRORE = 500
REVENUE_MAX_CRORE = 2000
MAX_COMPANIES_PER_SECTOR = 8
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

# Token-saving knobs
MAX_SEARCH_CHARS = 4000        # input budget for Tavily text fed to LLM (~1000 tokens)
MAX_OUTPUT_TOKENS = 2048       # hard cap per-call to prevent runaway responses
MAX_REPORT_DATA_CHARS = 6000   # cap data blob sent to reporter
TOP_COMPANIES_FOR_REPORT = 100 # number of companies that receive email drafts

# Name-resolution confidence thresholds (agents/researcher.py)
NAME_MATCH_CONFIDENCE_THRESHOLD = 0.6  # reject yfinance result below this
BLIND_MATCH_CONFIDENCE = 0.5           # assigned when no longname returned by yfinance

# Time windows for computed features (agents/analyst.py)
CXO_CHANGE_WINDOW_DAYS = 180  # lookback for CXO change signals
CONCALL_RECENCY_DAYS = 90     # lookback for concall signals

# PDF extraction page ranges — 0-indexed, targets the MD&A section (tools/company_scraper.py)
PDF_MDNA_START_PAGE = 8   # skip cover + table of contents
PDF_MDNA_END_PAGE = 20    # end of MD&A section (exclusive)
PDF_FALLBACK_PAGES = 5    # pages to extract when PDF is shorter than start page

# S3 storage
S3_REGION = os.getenv("S3_REGION", AWS_REGION)         # falls back to AWS_DEFAULT_REGION
S3_RESEARCH_PREFIX = os.getenv("S3_RESEARCH_PREFIX", "research")  # legacy, kept for backward compat
S3_RUNS_PREFIX = os.getenv("S3_RUNS_PREFIX", "runs")              # new: runs/{run_id}/{artifact}
S3_BUCKET = os.getenv("S3_BUCKET")