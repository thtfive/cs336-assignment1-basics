# log_setup.py
from __future__ import annotations
from loguru import logger
import logging
import sys
from pathlib import Path
from typing import Optional, Union

# Flag to avoid adding sinks multiple times
_LOGGER_CONFIGURED = False


class _InterceptHandler(logging.Handler):
    """Redirect standard library logging messages to Loguru."""
    def emit(self, record: logging.LogRecord):
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        # depth=6 helps point to the original caller in the stack trace
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def init_logger(
    log_dir: Union[str, Path] = "logs",
    level: str = "INFO",
    json_file: bool = False,
    rotation: str = "50 MB",
    retention: str = "14 days",
    compression: Optional[str] = "zip",
    diagnose: bool = False,   # True for debugging, False for production
    backtrace: bool = True,   # Better traceback formatting
    enqueue: bool = True,     # Thread/process-safe logging
    bridge_std_logging: bool = True,  # Redirect Python logging to Loguru
):
    """
    Initialize the logger once at the application entry point.

    After initialization, use get_logger(__name__) in any module.
    """
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return logger

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default sink to prevent duplicate console output
    logger.remove()
    level = level.upper()
    # Console sink
    logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        backtrace=backtrace,
        diagnose=diagnose,
        enqueue=enqueue,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[module]:<25}</cyan> | "
            "{name}:{function}:{line} - <level>{message}</level>"
        ),
    )

    # File sink (optionally JSON-serialized)
    file_format = "{message}" if json_file else (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[module]:<25} | "
        "{name}:{function}:{line} - {message}"
    )
    logger.add(
        log_dir / "train.log",
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=enqueue,
        serialize=json_file,
        format=file_format,
    )

    # Bridge Python's built-in logging
    if bridge_std_logging:
        logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    _LOGGER_CONFIGURED = True
    return logger


def get_logger(module_name: str = ""):
    """Bind the logger with a module name for better filtering and context."""
    return logger.bind(module=module_name or "__main__")
