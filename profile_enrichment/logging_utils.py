import logging
from pathlib import Path

from profile_enrichment.config import CONFIG


def setup_logger() -> logging.Logger:
    log_dir = Path(CONFIG["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("profile_enrichment")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_dir / "enrichment.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
