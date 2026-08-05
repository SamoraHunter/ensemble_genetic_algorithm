"""Pytest configuration for automatic cleanup of test artifacts."""

import os
import re
import shutil


def _remove_directory_if_matches_pattern(path, pattern, max_depth=1):
    """Remove all directories under path that match the given pattern.

    Args:
        path: The directory to search in
        pattern: Regex pattern to match directory names
        max_depth: Maximum depth to traverse (default 1 for immediate children)
    """
    if not os.path.isdir(path):
        return

    try:
        items = os.listdir(path)
    except Exception:
        return

    for item_name in items:
        item_path = os.path.join(path, item_name)
        if os.path.isdir(item_path):
            if re.match(pattern, item_name):
                try:
                    shutil.rmtree(item_path, ignore_errors=True)
                except Exception:
                    pass
            elif max_depth > 1:
                _remove_directory_if_matches_pattern(item_path, pattern, max_depth - 1)


def pytest_sessionstart(session):
    """Clean up experiment directories created during test execution.

    This function removes timestamped experiment directories (e.g., 2026-06-28_21-35-03)
    that are created during test runs. It specifically targets:
    - Directories matching the pattern YYYY-MM-DD_HH-MM-SS in the experiments folder
    - Model store JSON files and empty directories left from tests

    Also cleans up DEAP creator classes to ensure test isolation.
    """
    cwd = os.getcwd()

    # Look for experiment directories with timestamp pattern in workspaces folder
    experiments_dir = os.path.join(cwd, "experiments")
    if os.path.exists(experiments_dir):
        _remove_directory_if_matches_pattern(
            experiments_dir, r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", max_depth=1
        )

    # Clean up model_store.json files created by tests
    shared_model_store_paths = [
        "/tmp/model_store.json",
        os.path.join(cwd, "model_store.json"),
    ]

    for ms_path in shared_model_store_paths:
        try:
            if os.path.exists(ms_path):
                os.remove(ms_path)
        except Exception:
            pass

    # Clean up mock-named files and directories (created when tests don't set model_store_path properly)
    import glob

    magic_files = glob.glob(os.path.join(cwd, "<*"))
    for mf in magic_files:
        try:
            if os.path.isfile(mf):
                os.remove(mf)
            elif os.path.isdir(mf):
                shutil.rmtree(mf, ignore_errors=True)
        except Exception:
            pass

    # Also check for lock files with similar patterns
    magic_locks = glob.glob(os.path.join(cwd, "<*.lock"))
    for ml in magic_locks:
        try:
            os.remove(ml)
        except Exception:
            pass

    # Clean up DEAP creator classes to ensure test isolation
    try:
        from deap import creator

        if hasattr(creator, "FitnessMax"):
            delattr(creator, "FitnessMax")
        if hasattr(creator, "Individual"):
            delattr(creator, "Individual")
    except Exception:
        pass

    # Clean up ml_grid module cache to ensure fresh imports between test runs
    try:
        import sys

        for mod_name in list(sys.modules.keys()):
            if "ml_grid" in mod_name:
                del sys.modules[mod_name]
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):
    """Clean up after all tests have completed."""
    pass
