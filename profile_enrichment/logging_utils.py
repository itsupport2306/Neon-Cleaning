import logging
from pathlib import Path

from profile_enrichment.config import CONFIG


class ConsoleFormatter(logging.Formatter):
    """Console formatter that keeps output to a single line per event.

    Tracebacks and stack traces still go to the log file; the terminal only
    shows the message, so long runs stay readable.
    """

    def formatException(self, ei) -> str:  # noqa: N802 - logging API
        return ""

    def formatStack(self, stack_info) -> str:  # noqa: N802 - logging API
        return ""

    def format(self, record: logging.LogRecord) -> str:
        saved_exc_info, saved_exc_text, saved_stack = record.exc_info, record.exc_text, record.stack_info
        record.exc_info = record.exc_text = record.stack_info = None
        try:
            return super().format(record)
        finally:
            record.exc_info, record.exc_text, record.stack_info = saved_exc_info, saved_exc_text, saved_stack


# Events that are useful in the log file but only add noise to the terminal.
CONSOLE_SUPPRESSED_EVENTS = ("NO_USEFUL_INFORMATION",)


class ConsoleNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = str(record.msg)
        return not any(event in message for event in CONSOLE_SUPPRESSED_EVENTS)


def _quiet_third_party_loggers() -> None:
    """Stop pypdf/pdfminer/httpx from writing parser warnings straight to stderr."""
    for name in ("pypdf", "pypdf._reader", "pdfminer", "PIL", "httpx", "openai"):
        third_party = logging.getLogger(name)
        third_party.setLevel(logging.ERROR)
        third_party.propagate = False
        third_party.addHandler(logging.NullHandler())


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
    stream_handler.setFormatter(
        ConsoleFormatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    stream_handler.addFilter(ConsoleNoiseFilter())

    _quiet_third_party_loggers()

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
