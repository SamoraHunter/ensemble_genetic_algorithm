import logging
import os
from datetime import datetime
import sys
from IPython.core.getipython import get_ipython


def setup_logger(log_folder_path="."):
    # Get the directory path of the current module
    module_dir = os.path.dirname(os.path.realpath(__file__))

    # Get the root directory of the notebook
    notebook_dir = os.path.dirname(
        get_ipython().config["IPKernelApp"]["connection_file"]
    )

    # Navigate up from the notebook directory to get the logs directory
    folder_name = log_folder_path
    current_dir = os.getcwd()
    # Combine the current directory and folder name to get the target directory
    logs_dir = os.path.abspath(os.path.join(current_dir, folder_name))
    print("logs_dir", logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    # Get the current date and time
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Store original stdout before any modifications
    if not hasattr(sys, "_original_stdout"):
        sys._original_stdout = sys.stdout

    # Create a logger and clear any existing handlers
    logger = logging.getLogger(__name__)
    logger.handlers.clear()  # Remove any existing handlers
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent propagation to avoid duplicates

    # Set up file handler
    log_file = os.path.join(logs_dir, f"{current_date_time}_ensemble_GA.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Define a trace function for logging
    def tracefunc(frame, event, arg):
        # Only log events from files within the notebook directory
        if notebook_dir in frame.f_code.co_filename:
            if event == "line":
                logger.debug(
                    f"{event}: {frame.f_code.co_filename} - Line {frame.f_lineno}"
                )
        return tracefunc

    # Register the trace function globally for all events
    sys.settrace(tracefunc)

    # Create a custom stdout writer that writes to both original stdout and log file
    class DualWriter:
        def __init__(self, logger, original_stdout):
            self.logger = logger
            self.original_stdout = original_stdout

        def write(self, message):
            if message.strip():
                # Write to original stdout (console) - this shows immediately
                self.original_stdout.write(message)
                self.original_stdout.flush()

                # Also write to log file through logger
                # Create log record manually to avoid formatter overhead for simple prints
                log_record = logging.LogRecord(
                    name=self.logger.name,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=message.strip(),
                    args=(),
                    exc_info=None,
                )
                # Only send to file handler, not console handler (to avoid duplication)
                file_handler.emit(log_record)

        def flush(self):
            self.original_stdout.flush()

        def isatty(self):
            return self.original_stdout.isatty()

    # Replace stdout with our dual writer
    sys.stdout = DualWriter(logger, sys._original_stdout)

    # Function to log messages that appear both in console and file with formatting
    def log_info(message):
        # This will appear in console with timestamp formatting and in log file
        formatted_msg = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - {message}"
        )
        sys._original_stdout.write(formatted_msg + "\n")
        sys._original_stdout.flush()
        logger.info(message)

    def log_debug(message):
        # Debug only goes to log file
        logger.debug(message)

    def log_error(message):
        formatted_msg = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {message}"
        )
        sys._original_stdout.write(formatted_msg + "\n")
        sys._original_stdout.flush()
        logger.error(message)

    # Add these methods to the logger for easy access
    logger.log_info = log_info
    logger.log_debug = log_debug
    logger.log_error = log_error

    return logger


def restore_stdout():
    """Helper function to restore original stdout if needed"""
    if hasattr(sys, "_original_stdout"):
        sys.stdout = sys._original_stdout
