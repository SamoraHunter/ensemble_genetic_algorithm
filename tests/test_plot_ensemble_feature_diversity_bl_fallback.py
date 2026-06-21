"""Test for plot_ensemble_feature_diversity with BL_ column containing non-list data."""

import json

import pandas as pd


def test_plot_ensemble_feature_diversity_bl_columns_with_non_list_entries():
    """Test plot_ensemble_feature_diversity when feature_names is missing and BL_ columns contain non-list entries.

    This test specifically covers lines 1160-1196 in the fallback path where:
    - 'feature_names' column doesn't exist (triggers else branch at line 1160)
    - BL_ columns exist but may contain non-list data (like integers, None, scalars)
    - decode_bl_features needs to handle TypeError when len() is called on non-list types

    The key edge case:
        - Line 1176-1177: len(bl_entry) comparison can throw TypeError if bl_entry is not a list
        - This triggers the except at line 1194-1195 which returns an empty set

    Covers:
        - Lines 1160-1166: Fallback path when feature_names is missing, requiring BL_ columns
        - Line 1172-1175: First branch - list of strings (already tested in other tests)
        - Lines 1176-1196: Second branch with len() comparison that can throw TypeError
        - Lines 1223-1227: Final validation check for empty temp_df after dropna
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    # DataFrame without 'feature_names' column - must use BL_ fallback path
    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78],
            # BL_ columns with mixed content including non-list entries
            # This triggers the TypeError path at line 1176 when len() is called
            "BL_0": [
                ["feature_a", "feature_b"],  # Valid list of strings
                42,  # Non-list: integer - triggers TypeError in decode_bl_features
            ],
            "BL_1": [
                "not_a_list",  # String - triggers TypeError in decode_bl_features
                ["feature_c"],
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Delete 'feature_names' to force fallback to BL_ columns path
    del explorer.df["feature_names"]

    result = explorer.plot_ensemble_feature_diversity(outcome_variable="auc")

    # Should return None without crashing, handling TypeError gracefully
    assert result is None


def test_plot_ensemble_feature_diversity_bl_columns_list_of_integers():
    """Test plot_ensemble_feature_diversity when BL_ columns contain list of integers matching feature length.

    This covers the specific path at lines 1176-1196 where:
    - bl_entry is a list with len() equal to original_feature_names
    - All elements are integers (not strings '0' or '1')
    - The isinstance(x, (int, np.integer)) check at line 1180 succeeds

    This triggers the branch that converts integer binary masks to feature names.

    Covers:
        - Line 1176-1177: Second branch where len(bl_entry) == len(original_feature_names)
        - Line 1180-1185: Integer list handling that extracts features based on bit position
        - Lines 1230-1256: Actual plotting code (using fallback to bl_cols path)
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78],
            # Both BL_ columns are lists of ints with len == 3 (matching original_feature_names)
            # This triggers the int-handling branch at lines 1180-1185
            "BL_0": [[1, 0, 1], [0, 1, 0]],
            "BL_1": [[0, 1, 1], [1, 1, 0]],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Delete 'feature_names' to force fallback to BL_ columns path
    del explorer.df["feature_names"]

    result = explorer.plot_ensemble_feature_diversity(outcome_variable="auc")

    assert result is None


def test_plot_ensemble_feature_diversity_bl_columns_string_binary_mask():
    """Test plot_ensemble_feature_diversity when BL_ columns contain string binary masks ("0" and "1").

    This covers the specific path at lines 1176-1193 (the second branch) where:
    - bl_entry is a list containing strings "0" or "1"
    - First branch check at line 1172-1175 fails because strings aren't in original_feature_names
    - Second branch check at line 1176-1187 succeeds
    - The isinstance(x, str) and x in {"0", "1"} check at line 1186-1187 succeeds

    This path handles a less common format where binary feature masks are stored
    as strings rather than integers. To reach this branch, the strings must NOT be
    valid feature names (so first branch returns empty set).

    Covers:
        - Line 1172-1175: First branch check fails (strings not in original_feature_names)
        - Line 1176-1187: Second branch with len() and string "0"/"1" checking
        - Line 1189-1193: Conversion of string masks to integer indices for feature extraction
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78],
            # BL_ columns with string binary masks ("0" and "1")
            # The strings must NOT be in original_feature_names to reach line 1176 branch
            "BL_0": [["0", "1", "0"], ["1", "0", "1"]],
            "BL_1": [["1", "1", "0"], ["0", "0", "1"]],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Delete 'feature_names' to force fallback to BL_ columns path
    del explorer.df["feature_names"]

    result = explorer.plot_ensemble_feature_diversity(outcome_variable="auc")

    assert result is None


def test_plot_ensemble_feature_diversity_only_one_bl_column():
    """Test plot_ensemble_feature_diversity when only one BL_ column exists.

    This covers lines 1162-1166 where the method checks for at least 2 BL columns.
    When only 1 BL column exists (or none), it logs an error and returns early.

    Covers:
        - Line 1161: bl_cols detection
        - Line 1162: if len(bl_cols) < 2 check
        - Lines 1163-1166: Error logging and early return
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
            ],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "auc": [0.85],
            # Only one BL_ column - this triggers the error path
            "BL_0": [[1, 0]],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    # Delete 'feature_names' to force fallback to BL_ columns path
    del explorer.df["feature_names"]

    result = explorer.plot_ensemble_feature_diversity(outcome_variable="auc")

    # Should return None due to insufficient BL columns
    assert result is None


def test_plot_parameter_distributions_config_path():
    """Test plot_parameter_distributions with param_type='config'.

    This covers lines 997-1056 where config parameters are plotted.

    Covers:
        - Line 996: if param_type in ["config", "run_details"] check
        - Lines 997-1006: param_map setup
        - Lines 1014-1055: Subplot grid and plotting loop for config params
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, True)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92],
            # These params should be in config_params for plotting
            "pop_val": [10, 20, 30],
            "g": [100, 200, 300],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_parameter_distributions(param_type="config")

    # Should return None after plotting
    assert result is None


def test_plot_parameter_distributions_run_details_path():
    """Test plot_parameter_distributions with param_type='run_details'.

    This covers lines 997-1056 where run details are plotted.

    Covers:
        - Line 996: if param_type in ["config", "run_details"] check
        - Lines 1002-1005: run_details param_map entry
        - Lines 1014-1055: Subplot grid and plotting loop for run details
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, False)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, True)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92],
            # These should be in run_details - need multiple params to trigger plot loop
            "sex": ["M", "F", "M"],
            "n_unique_out": ["A", "B", "C"],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_parameter_distributions(param_type="run_details")

    assert result is None


def test_plot_ensemble_feature_diversity_empty_after_dropna():
    """Test plot_ensemble_feature_diversity when temp_df becomes empty after dropna.

    This covers lines 1223-1227 where if temp_df.empty, the method returns early
    with a warning. This happens when all calls to get_avg_jaccard return None,
    which occurs when:
    - All feature_sets have < 2 elements (line 1204-1205)
    - Or the overall DataFrame becomes empty after dropping NaN rows

    Test scenario: All BL_ entries are either non-lists (returning empty set) or
    lists with insufficient base learners, resulting in all None values.
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
            ],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "auc": [0.85],
            # All non-list entries - will all return empty sets from decode_bl_features
            "BL_0": [42],  # Integer, not a list
            "BL_1": ["string"],  # String, not a list
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_ensemble_feature_diversity(outcome_variable="auc")

    # Should return None without plotting due to empty temp_df
    assert result is None
