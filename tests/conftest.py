"""Pytest configuration for automatic cleanup of test artifacts."""

import os
import re
import shutil


def pytest_runtest_teardown(item):
    """Clean up experiment directories created during test execution.
    
    This fixture removes timestamped experiment directories (e.g., 2026-06-28_21-35-03)
    that are created during test runs. It specifically targets:
    - Directories matching the pattern YYYY-MM-DD_HH-MM-SS in the experiments folder
    """
    cwd = os.getcwd()
    
    # Look for experiment directories with timestamp pattern in workspaces folder
    experiments_dir = os.path.join(cwd, "experiments")
    if os.path.exists(experiments_dir):
        for item_name in os.listdir(experiments_dir):
            item_path = os.path.join(experiments_dir, item_name)
            if os.path.isdir(item_path):
                # Check if it matches timestamp pattern YYYY-MM-DD_HH-MM-SS
                if re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", item_name):
                    try:
                        shutil.rmtree(item_path, ignore_errors=True)
                    except Exception:
                        pass
