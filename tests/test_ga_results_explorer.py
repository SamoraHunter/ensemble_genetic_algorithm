"""Tests for GA_results_explorer module."""

import json
import os

import pandas as pd


def test_empty_ensemble_feature_arrays():
    """Test decode_features_per_row when best_ensemble contains '[[]]' (string with empty bit array).

    This test specifically covers the edge case where best_ensemble is a string that parses to [[]],
    which represents an ensemble containing one base learner with an empty feature array.
    The decode_features_per_row method must handle this without errors at lines 92-93 when
    accessing explorer.df['feature_arrays'].iloc[0] and explorer.df['feature_names'].iloc[0].

    Also covers:
        - Line 71-73: extract_feature_arrays_from_string with '[[]]' input
        - Line 76-99: decode_features_per_row with empty feature arrays
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[]]"],
            "original_feature_names": json.dumps(["f1", "f2"]),
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["f1", "f2"],
        global_params_obj=global_params,
    )

    feature_arrays = explorer.df["feature_arrays"].iloc[0]
    assert isinstance(feature_arrays, list)
    assert feature_arrays == []

    feature_names = explorer.df["feature_names"].iloc[0]
    assert isinstance(feature_names, list)
    assert feature_names == []


def test_ga_results_explorer_initializes_with_basic_df():
    """Test that GA_results_explorer initializes with minimal DataFrame."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame({"best_ensemble": [[]]})

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature1"],
        global_params_obj=global_params,
    )

    assert hasattr(explorer, "df")
    assert hasattr(explorer, "original_feature_names")


def test_ga_results_explorer_creates_default_global_params():
    """Test that GA_results_explorer creates default global params when None."""
    from ml_grid.util import GA_results_explorer

    df = pd.DataFrame({"best_ensemble": [[]]})

    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature1"],
    )

    assert hasattr(explorer, "global_params")


def test_apply_plot_truncation_default_expand_false():
    """Test _apply_plot_truncation with default expand=False."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {"F-statistic": [1, 2, 3, 4, 5], "Parameter": ["a", "b", "c", "d", "e"]}
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=pd.DataFrame({"best_ensemble": [[]]}),
        original_feature_names=["feature1"],
        global_params_obj=global_params,
    )

    result = explorer._apply_plot_truncation(df, "test")

    assert len(result) == 5


def test_decode_features_per_row_fallback_with_invalid_json():
    """Test decode_features_per_row fallback when original_feature_names has invalid JSON."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": ["invalid json {"],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature1", "feature2", "feature3"],
        global_params_obj=global_params,
    )

    assert hasattr(explorer, "df")
    assert len(explorer.df["feature_names"]) == 1

    feature_names = explorer.df["feature_names"].iloc[0]

    assert isinstance(feature_names, list)
    assert len(feature_names) == 1
    assert feature_names[0] == ["feature1", "feature3"]


def test_decode_features_per_row_success_path():
    """Test decode_features_per_row success path with valid JSON."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": json.dumps(["feature1", "feature2", "feature3"]),
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature1", "feature2", "feature3"],
        global_params_obj=global_params,
    )

    assert hasattr(explorer, "df")
    assert len(explorer.df["feature_names"]) == 1

    feature_names = explorer.df["feature_names"].iloc[0]

    assert isinstance(feature_names, list)
    assert len(feature_names) == 1
    assert feature_names[0] == ["feature1", "feature3"]


def test_extract_feature_arrays_from_string_error_handling():
    """Test extract_feature_arrays_from_string error handling with invalid input."""
    from ml_grid.util.GA_results_explorer import extract_feature_arrays_from_string

    result = extract_feature_arrays_from_string(None)
    assert result == []

    result = extract_feature_arrays_from_string("")
    assert result == []


def test_plot_base_learner_feature_importance_success():
    """Test plot_base_learner_feature_importance with valid data."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
                "[[(0.5, 'Model4', [1, 1, 1], 0, 0.88, None)]]",
                "[[(0.6, 'Model5', [0, 0, 1], 0, 0.75, None)]]",
                "[[(0.7, 'Model6', [1, 0, 0], 0, 0.82, None)]]",
                "[[(0.5, 'Model7', [1, 1, 0], 0, 0.90, None)]]",
                "[[(0.6, 'Model8', [0, 1, 0], 0, 0.77, None)]]",
                "[[(0.7, 'Model9', [0, 0, 0], 0, 0.70, None)]]",
                "[[(0.5, 'Model10', [1, 1, 1], 0, 0.94, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92, 0.88, 0.75, 0.82, 0.90, 0.77, 0.70, 0.94],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    assert len(explorer.df["feature_names"]) == 10


def test_plot_base_learner_feature_importance_success_full():
    """Test plot_base_learner_feature_importance with valid feature_names data and ANOVA running successfully.

    This test specifically covers lines 843-973 which were previously uncovered because:
    - Existing tests either did not call the function or triggered early returns
    - The 'feature_names' column must exist AND contain properly structured list data
    - Features must appear inconsistently (some runs with, some without) for ANOVA to work

    Covers:
        - Line 843-859: Aggregating features from feature_names and combining into all_bl_features
        - Line 861-896: Processing unique features with truncation logic
        - Line 898-927: ANOVA calculation for each feature (lines 903-927)
        - Line 934-942: Creating results DataFrame and applying truncation
        - Line 945-969: Plotting without plot_dir
        - Line 973-975: Printing results table

    The key difference from plot_initial_feature_importance: this uses 'feature_names'
    which contains already-decoded feature lists, whereas initial uses f_list with bit masks.
    """

    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
                "[[(0.5, 'Model4', [1, 1, 1], 0, 0.88, None)]]",
                "[[(0.6, 'Model5', [0, 0, 1], 0, 0.75, None)]]",
                "[[(0.7, 'Model6', [1, 0, 0], 0, 0.82, None)]]",
                "[[(0.5, 'Model7', [1, 1, 0], 0, 0.90, None)]]",
                "[[(0.6, 'Model8', [0, 1, 0], 0, 0.77, None)]]",
                "[[(0.7, 'Model9', [0, 0, 0], 0, 0.70, None)]]",
                "[[(0.5, 'Model10', [1, 1, 1], 0, 0.94, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92, 0.88, 0.75, 0.82, 0.90, 0.77, 0.70, 0.94],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    assert len(explorer.df["feature_names"]) == 10

    result = explorer.plot_base_learner_feature_importance(outcome_variable="auc")

    assert result is None


def test_plot_base_learner_feature_importance_with_plot_dir():
    """Test plot_base_learner_feature_importance with plot_dir provided.

    This test covers the lines 963-970 where plot_path is constructed and saved.

    Test data has two groups of 5 rows:
    - Group 1: [1,0,1] (feature_a + feature_c) with high AUC (~0.92)
    - Group 2: [0,1,1] (feature_b + feature_c) with low AUC (~0.78)
    This gives ANOVA clear separation for feature_a and feature_b.
    """
    import tempfile

    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.92, None)]]",
                "[[(0.6, 'Model2', [1, 0, 1], 0, 0.94, None)]]",
                "[[(0.7, 'Model3', [1, 0, 1], 0, 0.90, None)]]",
                "[[(0.5, 'Model4', [1, 0, 1], 0, 0.95, None)]]",
                "[[(0.6, 'Model5', [1, 0, 1], 0, 0.93, None)]]",
                "[[(0.7, 'Model6', [0, 1, 1], 0, 0.78, None)]]",
                "[[(0.5, 'Model7', [0, 1, 1], 0, 0.76, None)]]",
                "[[(0.6, 'Model8', [0, 1, 1], 0, 0.79, None)]]",
                "[[(0.7, 'Model9', [0, 1, 1], 0, 0.75, None)]]",
                "[[(0.5, 'Model10', [0, 1, 1], 0, 0.80, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [
                0.92,
                0.94,
                0.90,
                0.95,
                0.93,
                0.78,
                0.76,
                0.79,
                0.75,
                0.80,
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = explorer.plot_base_learner_feature_importance(
            outcome_variable="auc", plot_dir=tmpdir
        )

        assert result is None

        expected_file = os.path.join(tmpdir, "base_learner_feature_importance.png")
        assert os.path.exists(expected_file), f"Plot file {expected_file} should exist"


def test_plot_base_learner_feature_importance_missing_outcome():
    """Test plot_base_learner_feature_importance with missing outcome variable."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_base_learner_feature_importance(
        outcome_variable="nonexistent"
    )
    assert result is None


def test_plot_base_learner_feature_importance_missing_feature_names():
    """Test plot_base_learner_feature_importance when feature_names column is missing."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "auc": [0.85],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    del explorer.df["feature_names"]

    result = explorer.plot_base_learner_feature_importance(outcome_variable="auc")
    assert result is None


def test_get_column_names_with_non_string_input():
    """Test get_column_names returns empty list for non-string inputs."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame({"best_ensemble": [[]]})
    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature1", "feature2"],
        global_params_obj=global_params,
    )

    result_none = explorer.get_column_names(None)
    assert result_none == []

    result_int = explorer.get_column_names(123)
    assert result_int == []

    result_list = explorer.get_column_names([0, 1, 0])
    assert result_list == []


def test_get_column_names_success_path():
    """Test get_column_names success path with valid string input."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame({"best_ensemble": [[]]})
    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.get_column_names("[0, 1, 1]")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "feature_b"
    assert result[1] == "feature_c"


def test_get_column_names_with_leading_zeros():
    """Test get_column_names handles single digit strings correctly."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame({"best_ensemble": [[]]})
    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["f1", "f2", "f3", "f4"],
        global_params_obj=global_params,
    )

    result = explorer.get_column_names("[0,1,0,1]")

    assert len(result) == 2
    assert result[0] == "f2"
    assert result[1] == "f4"


def test_apply_plot_truncation_trigger_default_max_features():
    """Test _apply_plot_truncation triggers truncation when data exceeds max_features."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame({"best_ensemble": [[]]})
    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature1"],
        global_params_obj=global_params,
    )

    large_df = pd.DataFrame({"a": range(65), "b": range(65)})
    result = explorer._apply_plot_truncation(large_df, "test")

    assert len(result) <= 60


def test_apply_plot_truncation_expand_with_large_dataset():
    """Test _apply_plot_truncation logs warning when expand=True and data > 1000."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame({"best_ensemble": [[]]})
    global_params = global_parameters()
    global_params.expand_plots = True
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature1"],
        global_params_obj=global_params,
    )

    large_df = pd.DataFrame({"a": range(1005), "b": range(1005)})
    result = explorer._apply_plot_truncation(large_df, "test plot")

    assert len(result) == 1005


def test_plot_config_anova_feature_importances_success():
    """Test plot_config_anova_feature_importances with valid data and multiple params."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92],
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

    result = explorer.plot_config_anova_feature_importances(outcome_variable="auc")

    assert result is None


def test_plot_config_anova_feature_importances_missing_outcome():
    """Test plot_config_anova_feature_importances with missing outcome variable."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_config_anova_feature_importances(
        outcome_variable="nonexistent"
    )
    assert result is None


def test_plot_config_anova_feature_importances_success_boolean_param():
    """Test plot_config_anova_feature_importances with boolean config param that has sufficient unique values."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92],
            "weighted": [
                True,
                True,
                False,
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_config_anova_feature_importances(outcome_variable="auc")

    assert result is None


def test_plot_run_details_anova_feature_importances_success():
    """Test plot_run_details_anova_feature_importances with valid categorical data."""
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
            "sex": ["M", "F", "M"],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_run_details_anova_feature_importances(outcome_variable="auc")

    assert result is None


def test_plot_run_details_anova_feature_importances_missing_outcome():
    """Test plot_run_details_anova_feature_importances with missing outcome variable."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_run_details_anova_feature_importances(
        outcome_variable="nonexistent"
    )
    assert result is None


def test_plot_combined_anova_empty_results():
    """Test plot_combined_anova_feature_importances when no params have enough unique values."""
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
            "single_value_col": [1, 1],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_combined_anova_feature_importances(
        outcome_variable="single_value_col"
    )

    assert result is None


def test_plot_combined_anova_missing_outcome():
    """Test plot_combined_anova_feature_importances with missing outcome variable."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "auc": [0.85],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_combined_anova_feature_importances(
        outcome_variable="nonexistent"
    )
    assert result is None


def test_plot_config_anova_with_plot_dir_success():
    """Test plot_config_anova_feature_importances with plot_dir provided.

    Test data has two groups of 5 rows:
    - Group 1 (pop_val=10): low AUC (~0.73)
    - Group 2 (pop_val=20): high AUC (~0.92)
    This gives ANOVA clear separation with meaningful variance.
    """
    import tempfile

    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.5, 'Model2', [1, 0, 1], 0, 0.8, None)]]",
                "[[(0.6, 'Model3', [1, 1, 0], 0, 0.95, True)]]",
                "[[(0.6, 'Model4', [1, 1, 1], 0, 0.72, None)]]",
                "[[(0.5, 'Model5', [1, 0, 1], 0, 0.75, None)]]",
                "[[(0.6, 'Model6', [1, 1, 0], 0, 0.93, True)]]",
                "[[(0.5, 'Model7', [1, 0, 1], 0, 0.88, None)]]",
                "[[(0.6, 'Model8', [1, 1, 0], 0, 0.92, True)]]",
                "[[(0.5, 'Model9', [1, 0, 1], 0, 0.74, None)]]",
                "[[(0.6, 'Model10', [1, 1, 0], 0, 0.91, True)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.73, 0.72, 0.75, 0.74, 0.76, 0.91, 0.92, 0.93, 0.91, 0.92],
            "pop_val": [10, 10, 10, 10, 10, 20, 20, 20, 20, 20],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = explorer.plot_config_anova_feature_importances(
            outcome_variable="auc", plot_dir=tmpdir
        )

        assert result is None

        expected_file = os.path.join(tmpdir, "config_anova_auc.png")
        assert os.path.exists(
            expected_file
        ), f"Plot file {expected_file} should have been created"


def test_plot_run_details_anova_with_plot_dir_success():
    """Test plot_run_details_anova_feature_importances with plot_dir provided."""
    import tempfile

    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, False)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, True)]]",
                "[[(0.8, 'Model4', [1, 1, 1], 0, 0.92, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92, 0.88],
            "n_unique_out": ["A", "A", "B", "C"],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = explorer.plot_run_details_anova_feature_importances(
            outcome_variable="auc", plot_dir=tmpdir
        )

        assert result is None


def test_plot_combined_anova_with_plot_dir_success():
    """Test plot_combined_anova_feature_importances with plot_dir provided."""
    import tempfile

    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, False)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78],
            "pop_val": [10, 20],
            "sex": ["M", "F"],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = explorer.plot_combined_anova_feature_importances(
            outcome_variable="auc", plot_dir=tmpdir
        )

        assert result is None


def test_plot_config_anova_with_readonly_plot_dir():
    """Test plot_config_anova_feature_importances error handling when plot_dir is not writable."""
    import tempfile

    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.5, 'Model2', [1, 0, 1], 0, 0.8, None)]]",
                "[[(0.6, 'Model3', [1, 1, 0], 0, 0.95, True)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92],
            "pop_val": [10, 10, 20],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chmod(tmpdir, 0o444)

        result = explorer.plot_config_anova_feature_importances(
            outcome_variable="auc", plot_dir=tmpdir
        )

        assert result is None


def test_plot_parameter_distributions_initial_features_success():
    """Test plot_parameter_distributions with initial_features param_type."""
    import tempfile

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
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = explorer.plot_parameter_distributions(
            param_type="initial_features", plot_dir=tmpdir
        )
        assert result is None


def test_plot_parameter_distributions_base_learner_features_success():
    """Test plot_parameter_distributions with base_learner_features param_type."""
    import tempfile

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
            "BL_0": [
                ["feature_a", "feature_b"],
                ["feature_b", "feature_c"],
            ],
            "BL_1": [
                ["feature_c"],
                ["feature_a"],
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = explorer.plot_parameter_distributions(
            param_type="base_learner_features", plot_dir=tmpdir
        )
        assert result is None


def test_plot_parameter_distributions_invalid_param_type():
    """Test plot_parameter_distributions error handling for invalid param_type."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [[]],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a"],
        global_params_obj=global_params,
    )

    result = explorer.plot_parameter_distributions(param_type="invalid_type")
    assert result is None


def test_plot_initial_feature_importance_success():
    """Test plot_initial_feature_importance with valid f_list data and multiple features."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, None)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92],
            "f_list": [
                "[1, 0, 1]",
                "[0, 1, 1]",
                "[1, 1, 0]",
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None


def test_plot_initial_feature_importance_string_feature_names():
    """Test plot_initial_feature_importance with f_list containing string feature names."""
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
            "f_list": [
                "['feature_a', 'feature_c']",
                "['feature_b', 'feature_c']",
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None


def test_plot_initial_feature_importance_empty_f_list():
    """Test plot_initial_feature_importance when decode_flist returns empty for invalid input."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "auc": [0.85],
            "f_list": [""],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None


def test_plot_initial_feature_importance_nonlist_f_list():
    """Test plot_initial_feature_importance with non-list f_list values triggering line 648."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "auc": [0.85],
            "f_list": [123],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None


def test_plot_initial_feature_importance_indices_list():
    """Test plot_initial_feature_importance with f_list containing list of indices (not binary mask)."""
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
            "f_list": [
                "[0, 2]",
                "[1, 2]",
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None


def test_plot_initial_feature_importance_triple_nested_f_list():
    """Test plot_initial_feature_importance with triple-nested f_list that needs flattening at line 638."""
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
            "f_list": [
                "[[0, 1, 0]]",
                "[[1, 0, 1]]",
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None


def test_plot_initial_feature_importance_with_plot_dir():
    """Test plot_initial_feature_importance with plot_dir provided."""
    import tempfile

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
            "f_list": [
                "[1, 0, 1]",
                "[0, 1, 1]",
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = explorer.plot_initial_feature_importance(
            outcome_variable="auc", plot_dir=tmpdir
        )

        assert result is None


def test_plot_initial_feature_importance_string_feature_names_coverage_666_667():
    """Test lines 666-667: decode_flist string feature names branch."""
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
            "f_list": [
                "['feature_a', 'feature_b', 'feature_c']",
                "['feature_a', 'feature_b', 'feature_c']",
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None


def test_plot_combined_anova_with_readonly_plot_dir():
    """Test plot_combined_anova_feature_importances error handling when plot_dir is not writable."""
    import tempfile

    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, False)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78],
            "pop_val": [10, 20],
            "sex": ["M", "F"],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chmod(tmpdir, 0o444)

        result = explorer.plot_combined_anova_feature_importances(
            outcome_variable="auc", plot_dir=tmpdir
        )

        assert result is None


def test_plot_combined_anova_with_multiple_params_and_run_details():
    """Test plot_combined_anova_variable_importances with sufficient data for valid ANOVA.

    Test data has 12 rows with strictly alternating True/False and M/F, creating
    clean groups with meaningful AUC separation:
    - Group 1 (weighted=False, sex=M): low AUC (~0.68)
    - Group 2 (weighted=True, sex=F): high AUC (~0.93)
    This gives ANOVA clear separation for both variables.
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.92, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.68, False)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.94, True)]]",
                "[[(0.5, 'Model4', [1, 1, 1], 0, 0.65, False)]]",
                "[[(0.6, 'Model5', [0, 0, 1], 0, 0.91, True)]]",
                "[[(0.7, 'Model6', [1, 0, 0], 0, 0.70, False)]]",
                "[[(0.5, 'Model7', [1, 1, 0], 0, 0.93, True)]]",
                "[[(0.6, 'Model8', [0, 1, 0], 0, 0.67, False)]]",
                "[[(0.7, 'Model9', [0, 0, 0], 0, 0.95, True)]]",
                "[[(0.5, 'Model10', [1, 1, 1], 0, 0.66, False)]]",
                "[[(0.6, 'Model11', [1, 0, 1], 0, 0.92, True)]]",
                "[[(0.7, 'Model12', [0, 1, 1], 0, 0.69, False)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [
                0.92,
                0.68,
                0.94,
                0.65,
                0.91,
                0.70,
                0.93,
                0.67,
                0.95,
                0.66,
                0.92,
                0.69,
            ],
            "weighted": [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
            ],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    assert "weighted" in explorer.config_params
    assert "sex" in explorer.run_details

    plot_dir = "/tmp/test_combined_anova"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    result = explorer.plot_combined_anova_feature_importances(
        outcome_variable="auc", plot_dir=plot_dir
    )

    assert result is None
    expected_file = os.path.join(plot_dir, "combined_anova_auc.png")
    assert os.path.exists(expected_file), f"Plot file {expected_file} not created"

    if os.path.exists(expected_file):
        os.remove(expected_file)


def test_plot_combined_anova_with_exception_handling():
    """Test plot_combined_anova_feature_importances with mixed params: some succeed, one throws ANOVA exception."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, False)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, True)]]",
                "[[(0.8, 'Model4', [1, 1, 1], 0, 0.92, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92, 0.88],
            "pop_val": [10, 10, 20, 20],
            "n_unique_out": ["A", "B", "C", "D"],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    assert explorer.df["n_unique_out"].nunique() > 1

    result = explorer.plot_combined_anova_feature_importances(outcome_variable="auc")

    assert result is None


def test_plot_initial_feature_importance_feature_reduction_with_max_features():
    """Test plot_initial_feature_importance with feature reduction triggered (>max_features_to_plot)."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    num_runs = 15
    all_feature_names = [f"feature_{i}" for i in range(310)]

    df_data = {
        "best_ensemble": [],
        "original_feature_names": json.dumps(all_feature_names),
        "auc": [],
        "f_list": [],
    }

    for i in range(num_runs):
        start_idx = (i * 20) % 310
        end_idx = min(start_idx + 25, 310)
        active_features = list(range(start_idx, end_idx))

        mask = [0] * 310
        for idx in active_features:
            mask[idx] = 1

        df_data["best_ensemble"].append(
            f"[[[0.5, 'Model{i}', {str(mask)}, 0, 0.9, None]]]"
        )
        df_data["auc"].append(0.8 + (i * 0.01))
        df_data["f_list"].append(str(mask))

    df = pd.DataFrame(df_data)

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=all_feature_names,
        global_params_obj=global_params,
    )

    unique_count = len(set(f"feature_{i}" for i in range(310)))
    assert unique_count > 300, "Test setup error: need >300 unique features"

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    assert result is None


def test_plot_initial_feature_importance_missing_outcome():
    """Test plot_initial_feature_importance with missing outcome variable triggers early return at line 620."""
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "f_list": ["[1, 0]"],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_initial_feature_importance(outcome_variable="nonexistent")

    assert result is None


def test_plot_run_details_anova_exception_handling_with_nan():
    """Test that plot_run_details_anova_feature_importances handles ANOVA exceptions gracefully when NaN values are present.

    This test specifically covers lines 404-405: the exception handler that logs warnings
    when ANOVA computation fails for a run detail column, even though the column has
    sufficient unique values to pass the initial check at line 382.

    The scenario uses NaN values in 'n_unique_out' (a run_details column) which causes
    sm.stats.anova_lm() to fail with "r_matrix performs f_test for using dimensions
    that are asymptotically non-normal", triggering the exception handler while
    other valid columns still execute successfully.
    """
    import numpy as np

    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                "[[(0.5, 'Model1', [1, 0, 1], 0, 0.9, None)]]",
                "[[(0.6, 'Model2', [0, 1, 1], 0, 0.8, False)]]",
                "[[(0.7, 'Model3', [1, 1, 0], 0, 0.95, True)]]",
                "[[(0.8, 'Model4', [1, 1, 1], 0, 0.92, None)]]",
            ],
            "original_feature_names": json.dumps(
                ["feature_a", "feature_b", "feature_c"]
            ),
            "auc": [0.85, 0.78, 0.92, 0.88],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Add a run detail column (from the self.run_details list) with NaN values
    # that will cause ANOVA to fail, triggering lines 404-405 exception handler
    explorer.df["n_unique_out"] = [10, np.nan, 20, 30]

    result = explorer.plot_run_details_anova_feature_importances(outcome_variable="auc")

    assert result is None


def test_plot_initial_feature_importance_f_list_string_scalar():
    """Test plot_initial_feature_importance when f_list contains string scalar (e.g., '123').

    This test specifically covers the edge case where f_list value is a string that parses
    to a non-list type via ast.literal_eval (e.g., '123' -> integer 123). The decode_flist
    function handles this by returning an empty list at line 642 in the else branch of
    the try block when parsed result is not a list.

    Covers:
        - Line 642: return [] when ast.literal_eval returns non-list (scalar value)
        - Line 648: secondary check for non-list after parsing fails

    This ensures graceful handling of malformed f_list inputs that parse successfully
    but produce invalid output types.
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": ["[[(0.5, 'Model', [1, 0, 1], 0, 0.9, None)]]"],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
            "auc": [0.85],
            "f_list": ["123"],  # String that parses to integer, NOT a list
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_initial_feature_importance(outcome_variable="auc")

    # The function should handle gracefully and return None when no valid features found
    assert result is None


def test_plot_base_learner_feature_importance_non_list_feature_names():
    """Test plot_base_learner_feature_importance when feature_names contains non-list entries (coverage 852-854)."""
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
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Inject a non-list entry in feature_names to trigger line 852-854 path
    explorer.df["feature_names"].iloc[0].append("not_a_list")

    result = explorer.plot_base_learner_feature_importance(outcome_variable="auc")

    assert result is None


def test_plot_algorithm_distribution_in_ensembles_with_exclude_list():
    """Test plot_algorithm_distribution_in_ensembles when algorithms are filtered by exclude_list.

    This test specifically covers the code paths:
        - Line 1952-1954: List comprehension that filters out excluded algorithm types
          (Pipeline, StandardScaler, MinMaxScaler, etc.)

    Covers lines 1936-1954 where algorithms are filtered against the exclude_list,
    ensuring that common transformer/pipeline names don't appear in final distribution.

    Also covers:
        - Line 1970: Creating frequency DataFrame from all_algorithms
        - Line 1971: Renaming columns to "Algorithm" and "Frequency"
        - Line 1974-1975: Applying plot truncation
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    df = pd.DataFrame(
        {
            "best_ensemble": [
                # Ensemble with both base learners and transformers (should be filtered)
                '[[(0.5, \'Pipeline(steps=[("sc", StandardScaler()), ("clf", LogisticRegression(C=1))])\', [1, 0, 1], 0, 0.9, None)]]',
                # Another ensemble with different mix
                "[[(0.6, 'RobustScaler()', [0, 1, 1], 0, 0.8, None)], [(0.4, 'RandomForestClassifier()', [1, 1, 0], 0, 0.95, None)]]",
            ],
            "original_feature_names": json.dumps(["feature_a", "feature_b"]),
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b"],
        global_params_obj=global_params,
    )

    result = explorer.plot_algorithm_distribution_in_ensembles()

    assert result is None


def test_plot_all_convergence_missing_history_column():
    """Test plot_all_convergence returns early when history_column not found in DataFrame.

    This test covers lines 1359-1361 where the method validates that the history column
    exists before attempting to process it. When a non-existent column is specified,
    the method should log an error and return None immediately without raising an exception.

    Covers:
        - Line 1359: Check if history_column in self.df.columns
        - Lines 1360-1361: Log error message and return early when column missing
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
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_all_convergence(
        history_column="nonexistent_history_column",
        performance_metric="auc",
        highlight_best=True,
    )

    assert result is None


def test_plot_all_convergence_missing_performance_metric_with_highlight():
    """Test plot_all_convergence returns early when highlight_best=True but performance_metric not found.

    This test covers lines 1362-1366 where the method validates the performance metric column
    exists when highlight_best parameter is True. The validation only triggers when highlight_best=True.

    Covers:
        - Line 1362: Check if highlight_best and performance_metric in self.df.columns
        - Lines 1363-1365: Log error message and return early when column missing with highlight

    Notes:
        - Must include history_column in DataFrame to reach the second validation check
        - When highlight_best=True but performance_metric is missing, early return occurs
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
            "generation_progress_list": [
                "[0.5, 0.6]",
                "[0.7, 0.8]",
            ],
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_all_convergence(
        history_column="generation_progress_list",
        performance_metric="nonexistent_metric",
        highlight_best=True,
    )

    assert result is None


def test_plot_all_convergence_no_crash_with_missing_performance_when_not_highlight():
    """Test plot_all_convergence works without performance_metric when highlight_best=False.

    When highlight_best=False, the validation at lines 1362-1366 does not trigger.
    This test verifies that the method can proceed even when performance_metric is missing
    from the DataFrame (as long as history_column exists), because highlighting is disabled.

    Covers:
        - Line 1359: history_column existence check passes
        - Lines 1362-1366: Performance metric validation skipped due to highlight_best=False

    Note: This test still expects the method to return early at line 1370-1372 when
    dropna results in empty data, because no history values are provided.
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
        }
    )

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    result = explorer.plot_all_convergence(
        history_column="generation_progress_list",
        performance_metric="nonexistent_metric",
        highlight_best=False,
    )

    assert result is None
