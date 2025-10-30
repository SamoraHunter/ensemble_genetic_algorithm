import os
import logging
from ml_grid.util.logger_setup import setup_logger

def test_logger_setup(tmp_path, caplog):
    """
    Tests that the logger is set up correctly, writing to both a file and the logging capture.
    """
    log_folder = tmp_path / "logs"

    # 1. Setup the logger
    # Clear any existing handlers to ensure a clean test
    logger = logging.getLogger("ensemble_ga")
    logger.handlers.clear()
    
    logger = setup_logger(log_folder_path=str(log_folder))

    # 2. Generate a log message
    log_message = "This is a direct log message."
    logger.info(log_message)

    # 3. Verify the output was written to the log file
    log_files = os.listdir(log_folder)
    assert len(log_files) == 1, "Expected exactly one log file to be created."

    log_file_path = log_folder / log_files[0]
    with open(log_file_path, "r") as f:
        log_content = f.read()

    assert log_message in log_content, "The direct logger message was not found in the log file."

    # 4. Verify the output was captured by caplog
    assert log_message in caplog.text
    assert "INFO" in caplog.text # Check that the level is also present
    
    # 5. Verify that print statements are not captured in the log file
    print_message = "This should not be in the log."
    print(print_message)
    
    with open(log_file_path, "r") as f:
        log_content = f.read()
    assert print_message not in log_content
