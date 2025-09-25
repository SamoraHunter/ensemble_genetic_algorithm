import os
import sys
import logging
from ml_grid.util.logger_setup import setup_logger, restore_stdout


def test_logger_writes_stdout_to_file(tmp_path):
    """
    Tests that the logger is set up correctly and that the DualWriter
    successfully captures stdout and writes it to the log file.
    """
    log_folder = tmp_path / "logs"
    original_stdout = sys.stdout  # Save the original stdout at the start of the test

    try:
        # 1. Setup the logger, which redirects sys.stdout
        logger = setup_logger(log_folder_path=str(log_folder))

        # 2. Generate output
        print_message = "This is a test print statement."
        log_message = "This is a direct log message."

        print(print_message)
        logger.info(log_message)

    finally:
        # 3. Restore stdout to its original state to avoid side effects
        restore_stdout()
        # A final check to ensure we didn't leave stdout in a weird state
        assert sys.stdout == original_stdout

    # 4. Verify the output was written to the log file
    log_files = os.listdir(log_folder)
    assert len(log_files) == 1, "Expected exactly one log file to be created."

    log_file_path = log_folder / log_files[0]
    with open(log_file_path, "r") as f:
        log_content = f.read()

    assert print_message in log_content, "The message from print() was not found in the log file."
    assert log_message in log_content, "The direct logger message was not found in the log file."