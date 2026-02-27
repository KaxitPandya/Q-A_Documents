"""
Document Q&A Assistant — Logging Setup
"""

import logging
import os
from datetime import datetime
from config import LOG_DIR, LOG_LEVEL

os.makedirs(LOG_DIR, exist_ok=True)

_log_file = os.path.join(
    LOG_DIR,
    f"app_{datetime.now().strftime('%Y%m%d')}.log",
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("docqa")
