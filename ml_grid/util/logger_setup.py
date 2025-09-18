import logging
import os
from datetime import datetime
import sys
from IPython.core.getipython import get_ipython


def setup_logger(log_folder_path: str = ".") -> logging.Logger:
    """Sets up a logger that writes to both console and a file.

    This function configures a logger that captures all stdout and redirects it
    to both the original console stdout and a timestamped log file. It is
    specifically designed to work within an IPython/Jupyter environment.

    It achieves this by replacing `sys.stdout` with a custom `DualWriter`
    class. It also sets up a global trace function to log line-by-line
    execution, which can be very verbose and is intended for deep debugging.

    Args:
        log_folder_path: The relative path to the directory where logs
            should be stored. Defaults to the current directory.

    Returns:
        A configured `logging.Logger` object with custom methods
        (`log_info`, `log_debug`, `log_error`) attached for convenience.
    """
    # Get the directory path of the current module
    module_dir = os.path.dirname(os.path.realpath(__file__))

    # This part is specific to running in a Jupyter/IPython environment
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
        """A trace function to log line execution for debugging."""
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
        """A custom file-like object to write to two destinations."""

        def __init__(self, logger: logging.Logger, original_stdout: object):
            """Initializes the DualWriter.

            Args:
                logger: The logger instance to write to.
                original_stdout: The original sys.stdout object.
            """
            self.logger = logger
            self.original_stdout = original_stdout

        def write(self, message: str):
            """Writes a message to both the console and the log file."""
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
            """Flushes the original stdout buffer."""
            self.original_stdout.flush()

        def isatty(self):
            """Checks if the original stdout is a TTY."""
            return self.original_stdout.isatty()

    # Replace stdout with our dual writer
    sys.stdout = DualWriter(logger, sys._original_stdout)

    # Function to log messages that appear both in console and file with formatting
    def log_info(message):
        """Logs an INFO message to both console and file."""
        formatted_msg = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - {message}"
        )
        sys._original_stdout.write(formatted_msg + "\n")
        sys._original_stdout.flush()
        logger.info(message)

    def log_debug(message):
        """Logs a DEBUG message to the file only."""
        logger.debug(message)

    def log_error(message):
        """Logs an ERROR message to both console and file."""
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
    """Restores the original `sys.stdout` object."""
    if hasattr(sys, "_original_stdout"):
        sys.stdout = sys._original_stdout
