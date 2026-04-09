# utils/logger.py — Shared timestamped logger (stdout + logs/run_log.txt)
#
# Pure stdlib so any module can import without pulling in agent / Bedrock deps.
# Keep this dead simple — match main.py's existing format exactly so historical
# logs stay consistent. Future upgrade: swap the implementation to
# logging.getLogger("brd") with FileHandler+StreamHandler; callers won't change.

import os
from datetime import datetime

# Resolve LOG_PATH to an ABSOLUTE path under the project root (the dir that
# contains utils/), so the log goes to the same file regardless of cwd.
_LOGGER_DIR = os.path.dirname(os.path.abspath(__file__))   # .../BRD/utils
_PROJECT_ROOT = os.path.dirname(_LOGGER_DIR)               # .../BRD
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
LOG_PATH = os.path.join(LOG_DIR, "run_log.txt")


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    # Defensive: re-ensure parent dir exists on every write so anything that
    # cleans logs/ mid-run (manual rm, watcher, pre-commit hook, etc.) can't
    # crash the pipeline.
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def warn(msg: str) -> None:
    log(msg, "WARN")


def error(msg: str) -> None:
    log(msg, "ERROR")
