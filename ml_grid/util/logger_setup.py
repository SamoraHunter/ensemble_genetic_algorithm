import logging
import os
from datetime import datetime
from typing import Optional
import sys

def setup_logger(log_folder_path: str = ".") -> logging.Logger:
    """Sets up a root logger to write to a file.

    This function configures the root logger to send messages to a timestamped
    log file within the specified directory. It ensures that handlers are not
    duplicated if called multiple times.

    Args:
        log_folder_path: The relative path to the directory where logs
            should be stored. Defaults to the current directory.

    Returns:
        The configured logger instance.
    """
    os.makedirs(log_folder_path, exist_ok=True)

    # Get the current date and time
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_folder_path, f"{current_date_time}_ensemble_GA.log")

    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(message)s")  # Simple format for console
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Get a specific logger for the application
    logger = logging.getLogger("ensemble_ga")
    logger.setLevel(logging.INFO)

    # Prevent propagation to the root logger during normal runs to avoid duplicate outputs,

    # but allow it during tests so that caplog can capture messages.
    logger.propagate = "pytest" in sys.modules

    # Avoid adding duplicate handlers
    # Clear existing handlers to prevent duplicates during re-runs (e.g., in tests)
    if logger.hasHandlers():
        logger.handlers.clear()

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)

    return logger


def restore_stdout():
    """Restores the original `sys.stdout` object."""
    if hasattr(sys, "_original_stdout"):
        sys.stdout = sys._original_stdout
