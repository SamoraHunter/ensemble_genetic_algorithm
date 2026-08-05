import sys

sys.path.insert(0, "/workspaces/ensemble_genetic_algorithm")

import os
import shutil

import numpy as np
import pandas as pd

from ml_grid.util.project_score_save import project_score_save_class

tmpdir = "/workspaces/ensemble_genetic_algorithm/test_run"
os.makedirs(tmpdir, exist_ok=True)

saver = project_score_save_class(tmpdir)


class MockMLGridObject:
    def __init__(self):
        self.base_project_dir = tmpdir
        self.local_param_dict = {
            "cx_type": "twopoint",
            "cxpb": 0.5,
            "mutpb": 0.2,
            "indpb": 0.05,
            "t_size": 3,
            "weighted": "unweighted",
            "data": {"age": True, "sex": True},
            "mutpb_adaptive": {"enabled": False},
            "niche_params": {"enabled": False},
        }
        self.verbose = 0
        # Create dummy dataframes for X_train, X_test etc.
        self.X_train = pd.DataFrame({"age": [1, 2, 3], "sex": [1, 1, 0]})
        self.y_train = pd.Series([0, 1, 0])
        self.X_test = pd.DataFrame({"age": [4, 5], "sex": [1, 0]})
        self.y_test = pd.Series([1, 0])
        self.X_test_orig = pd.DataFrame({"age": [6, 7], "sex": [1, 0]})
        self.y_test_orig = pd.Series([1, 0])
        self.param_space_index = 0
        self.nb_val = 4
        self.pop_val = 32
        self.g_val = 128
        self.g = 128
        self.original_feature_names = ["age", "sex"]


# Test with valid=True to use y_test_orig
try:
    saver.update_score_log(
        ml_grid_object=MockMLGridObject(),
        best_pred_orig=np.array(
            [1, 0]
        ),  # Discrete predictions for classification metrics
        current_algorithm=None,
        method_name="test",
        pg=1,
        start=0,
        n_iter_v=10,
        valid=True,
    )

    df = pd.read_csv(os.path.join(tmpdir, "final_grid_score_log.csv"))

    print("Columns from CSV file:")
    print(list(df.columns))
    print()
    print("DataFrame contents:")
    print(df.to_string())
    print()
    print(f"mutpb_adaptive_enabled: {df['mutpb_adaptive_enabled'].iloc[0]}")
    print(f"niche_params_enabled: {df['niche_params_enabled'].iloc[0]}")

except Exception as e:
    print(f"Error: {e}")
    # Also check if the CSV was created even if update failed
    csv_path = os.path.join(tmpdir, "final_grid_score_log.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("\nExisting columns (from initial create):")
        print(list(df.columns))

# Cleanup
shutil.rmtree(tmpdir)
