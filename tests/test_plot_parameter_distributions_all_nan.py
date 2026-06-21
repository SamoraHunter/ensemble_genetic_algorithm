"""Test for GA_results_explorer plot_parameter_distributions with all-NaN parameter column."""

import json

import pandas as pd


def test_plot_parameter_distributions_all_nan_column():
    """Test plot_parameter_distributions handles parameter column where all values are NaN.

    This test specifically covers lines 1024-1027 in the plot_parameter_distributions method:

    - Line 1023: data = self.df[param].dropna() - when param column has only NaN values
    - Line 1024: if data.empty: - triggers when dropna() results in empty series
    - Lines 1025-1027: Display "No Data" text on the subplot for missing data

    The method should gracefully handle parameters where all data is missing by:
    - Adding "No Data" centered text to the subplot
    - Setting an appropriate title
    - Continuing with the next parameter (not crashing)

    This is an edge case that can occur with run metadata columns that may not be present
    in all datasets.
    """
    from ml_grid.util import GA_results_explorer
    from ml_grid.util.global_params import global_parameters

    # Create a DataFrame where we explicitly add config param columns but set them to all NaN
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

    # Add a config param column where ALL values are NaN
    df["weighted"] = pd.NA

    global_params = global_parameters()
    explorer = GA_results_explorer.GA_results_explorer(
        df=df,
        original_feature_names=["feature_a", "feature_b", "feature_c"],
        global_params_obj=global_params,
    )

    # Verify that config params like 'weighted' exist in the dataframe but have all NaN
    assert "weighted" in explorer.df.columns

    # Get the 'weighted' column and check it's all NaN after initialization
    weighted_col = explorer.df["weighted"]
    assert weighted_col.isna().all(), "Test setup: weighted column should be all NaN"

    # Call plot_parameter_distributions with param_type="config"
    # This will iterate through config_params including 'weighted'
    result = explorer.plot_parameter_distributions(param_type="config")

    # Should complete without error and return None
    assert result is None
