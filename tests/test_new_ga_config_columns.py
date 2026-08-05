import os

import numpy as np
import pandas as pd

from ml_grid.util.project_score_save import project_score_save_class


class MockMLGridObjectWithConfig:
    def __init__(self, base_dir):
        self.base_project_dir = base_dir
        self.verbose = 0
        self.local_param_dict = {
            "data": {"age": True, "sex": True},
            "resample": None,
            "scale": True,
            "n_features": "all",
            "param_space_size": "medium",
            "cxpb": 0.5,
            "mutpb": 0.2,
            "indpb": 0.05,
            "t_size": 3,
            "weighted": "unweighted",
            "use_stored_base_learners": False,
            "store_base_learners": False,
            "n_unique_out": 10,
            "outcome_var_n": "1",
            "div_p": 0,
            "percent_missing": 99,
            "corr": 0.98,
            # Test that these are logged correctly
            "mutpb_adaptive": {"enabled": True},
            "niche_params": {"enabled": False},
        }
        self.X_train = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        self.y_train = pd.Series([0, 1])
        self.X_test = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        self.y_test = pd.Series([0, 1])
        self.X_test_orig = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        self.y_test_orig = pd.Series([0, 1])
        self.param_space_index = 0
        self.original_feature_names = ["col_a", "col_b"]
        self.nb_val = 4
        self.pop_val = 8
        self.g_val = 4
        self.g = 4


def test_new_ga_config_columns(tmp_path):
    """
    Tests that mutpb_adaptive_enabled and niche_params_enabled columns are correctly
    logged to the CSV file.
    """
    project_dir = tmp_path
    saver = project_score_save_class(base_project_dir=project_dir)

    mock_ml_grid_object = MockMLGridObjectWithConfig(base_dir=project_dir)

    # Run update_score_log
    saver.update_score_log(
        ml_grid_object=mock_ml_grid_object,
        best_pred_orig=np.array([0, 1]),
        current_algorithm="test_algo",
        method_name="test_method",
        pg=10,
        start=0,
        n_iter_v=10,
        valid=False,
        generation_progress_list=[],
        best_ensemble="test_ensemble",
        original_feature_names=["col_a", "col_b"],
    )

    # Read the CSV
    log_file_path = os.path.join(project_dir, "final_grid_score_log.csv")
    df = pd.read_csv(log_file_path)

    # Check that the new columns exist
    assert (
        "mutpb_adaptive_enabled" in df.columns
    ), "mutpb_adaptive_enabled column missing"
    assert "niche_params_enabled" in df.columns, "niche_params_enabled column missing"

    # Check the values are correct
    assert df["mutpb_adaptive_enabled"].iloc[
        0
    ], f"Expected mutpb_adaptive_enabled=True, got {df['mutpb_adaptive_enabled'].iloc[0]}"
    assert not df["niche_params_enabled"].iloc[
        0
    ], f"Expected niche_params_enabled=False, got {df['niche_params_enabled'].iloc[0]}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
