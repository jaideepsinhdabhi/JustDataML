import logging as _logging
import os
from datetime import datetime

# Library code logs through this logger. Nothing is configured at import time;
# the CLI calls setup_logging() to write a log file.
logging = _logging.getLogger("jdml")
logging.addHandler(_logging.NullHandler())


def setup_logging(base_dir=None):
    """Log to <base_dir>/logs/<date>-logs/<timestamp>.log (CLI use)."""
    now = datetime.now()
    logs_path = os.path.join(base_dir or os.getcwd(), "logs", f"{now.strftime('%d_%m_%Y')}-logs")
    os.makedirs(logs_path, exist_ok=True)
    handler = _logging.FileHandler(os.path.join(logs_path, f"{now.strftime('%d_%m_%Y_%H_%M_%S')}.log"))
    handler.setFormatter(_logging.Formatter(
        "[ %(asctime)s ] - %(filename)s - %(lineno)d %(name)s - %(levelname)s - %(message)s"))
    logging.addHandler(handler)
    logging.setLevel(_logging.INFO)
