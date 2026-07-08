import logging
import os
import sys
from datetime import datetime


def setup_logger(log_folder_path: str = ".", verbose: int = 0) -> logging.Logger:
    """Sets up a root logger to write to a file.

    This function configures the root logger to send messages to a timestamped
    log file within the specified directory. It ensures that handlers are not
    duplicated if called multiple times.

    Args:
        log_folder_path: The relative path to the directory where logs
            should be stored. Defaults to the current directory.
        verbose: Verbosity level (0-3+). Levels 0-10 show INFO, levels 11+ also show DEBUG on console.

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

    # Determine console log level based on verbose parameter
    console_level = logging.DEBUG if verbose >= 11 else logging.INFO

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(message)s")  # Simple format for console
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level)

    # Get a specific logger for the application
    logger = logging.getLogger("ensemble_ga")
    logger.setLevel(logging.INFO)

    # Prevent propagation to the root logger during normal runs,
    # but allow it during tests so that caplog can capture messages.
    logger.propagate = False
    logger.propagate = "pytest" in sys.modules

    # Clear existing handlers to ensure only one active FileHandler at a time
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add the file handler (only one file handler ever)
    logger.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)

    return logger


def restore_stdout():
    """Restores the original `sys.stdout` object."""
    if hasattr(sys, "_original_stdout"):
        sys.stdout = sys._original_stdout
