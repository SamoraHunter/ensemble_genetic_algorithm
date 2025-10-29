import os
import pandas as pd
import pytest
import numpy as np
import logging

from ml_grid.util.project_score_save import project_score_save_class

# A mock class to simulate the ml_grid_object
class MockMLGridObject:
    def __init__(self, base_dir):
        self.base_project_dir = base_dir
        self.verbose = 2
        self.local_param_dict = {
            'data': {'age': True, 'sex': True, 'bmi': True, 'ethnicity': True, 'bloods': True, 'diagnostic_order': True, 'drug_order': True, 'annotation_n': True, 'meta_sp_annotation_n': True, 'meta_sp_annotation_mrc_n': False, 'annotation_mrc_n': True, 'core_02': True, 'bed': False, 'vte_status': False, 'hosp_site': False, 'core_resus': True, 'news': True, 'date_time_stamp': False, 'appointments': False},
            'resample': None, 'scale': True, 'n_features': 'all', 'param_space_size': 'medium',
            'cxpb': 0.5, 'mutpb': 0.2, 'indpb': 0.05, 't_size': 3,
            'weighted': 'unweighted', 'use_stored_base_learners': False, 'store_base_learners': False,
            'n_unique_out': 10, 'outcome_var_n': '1', 'div_p': 0, 'percent_missing': 99, 'corr': 0.98
        }
        # Use dummy data that won't cause issues
        self.X_train = pd.DataFrame({'col_a': [1, 2], 'col_b': [3, 4]})
        self.y_train = pd.Series([0, 1])
        self.X_test = pd.DataFrame({'col_a': [1, 2], 'col_b': [3, 4]})
        self.y_test = pd.Series([0, 1])
        self.X_test_orig = pd.DataFrame({'col_a': [1, 2], 'col_b': [3, 4]})
        self.y_test_orig = pd.Series([0, 1])
        self.param_space_index = 0
        self.original_feature_names = ['col_a', 'col_b']
        self.nb_val = 4
        self.pop_val = 8
        self.g_val = 4
        self.g = 4

def test_column_misalignment_detection(tmp_path, caplog):
    """
    Tests that a column misalignment between the data to be saved and the
    existing CSV header is detected and logged correctly.
    """
    # 1. Setup: Create a temporary directory and a dummy log file with specific headers.
    project_dir = tmp_path
    log_file_path = os.path.join(project_dir, "final_grid_score_log.csv")

    # These are the "correct" columns expected in the log file.
    expected_columns = [
        "nb_size", "f_list", "auc", "mcc", "f1", "precision", "recall", "accuracy",
        "nb_val", "pop_val", "g_val", "g", "weighted", "use_stored_base_learners",
        "store_base_learners", "resample", "scale", "n_features", "param_space_size",
        "n_unique_out", "outcome_var_n", "div_p", "percent_missing", "corr", "age",
        "sex", "bmi", "ethnicity", "bloods", "diagnostic_order", "drug_order",
        "annotation_n", "meta_sp_annotation_n", "meta_sp_annotation_mrc_n",
        "annotation_mrc_n", "core_02", "bed", "vte_status", "hosp_site", "core_resus",
        "news", "date_time_stamp", "X_train_size", "X_test_orig_size", "X_test_size",
        "run_time", "cxpb", "mutpb", "indpb", "t_size", "valid",
        "generation_progress_list", "best_ensemble", "original_feature_names"
    ]
    
    # Create a dummy CSV with one extra column and one missing column
    # to simulate misalignment.
    csv_header_with_mismatch = expected_columns.copy()
    csv_header_with_mismatch.remove("t_size")  # Remove a column
    csv_header_with_mismatch.append("extra_column_in_csv") # Add an extra one
    
    pd.DataFrame(columns=csv_header_with_mismatch).to_csv(log_file_path, index=False)

    # 2. The Test Logic: Attempt to save data with the "correct" columns.
    mock_ml_grid_object = MockMLGridObject(base_dir=project_dir)
    
    # The project_score_save_class will use its own internal column list,
    # which will now be different from the header in the dummy CSV.
    saver = project_score_save_class(base_project_dir=project_dir)

    with caplog.at_level(logging.ERROR):
        saver.update_score_log(
            ml_grid_object=mock_ml_grid_object,
            scores={},
            best_pred_orig=np.array([0, 1]),
            current_algorithm="test_algo",
            method_name="test_method",
            pg=10,
            start=0,
            n_iter_v=10,
            valid=False,
            generation_progress_list=[],
            best_ensemble="test_ensemble",
            original_feature_names=['col_a', 'col_b']
        )

    # 3. Assertions: Check if the correct error messages were logged.
    log_text = caplog.text
    assert "COLUMN MISALIGNMENT DETECTED!" in log_text
    
    # Check that the log correctly identifies the missing and extra columns
    assert "Columns missing from the new data: ['extra_column_in_csv']" in log_text
    assert "Extra columns in the new data: ['t_size']" in log_text
    
    # Check that it logs the expected vs actual columns for debugging
    assert "Expected columns (from CSV):" in log_text
    assert "Actual columns (to be saved):" in log_text