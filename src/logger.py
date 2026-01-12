import logging
import sys
import datetime
from pathlib import Path

DIRECTORY = Path(__file__).parent.parent
LOG_DIRECTORY = DIRECTORY / "logs"
LOG_DIRECTORY.mkdir(exist_ok=True)
LOG_FILE_PATH = LOG_DIRECTORY / f"{datetime.datetime.now().strftime('%Y%m%d')}.log"
# 1. Configure logging

stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")  # Also save to a file

logging.basicConfig(
    level=logging.INFO,  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[stream_handler, file_handler],
)

# 2. Create a logger instance for your agent
log = logging.getLogger("project_agent")
