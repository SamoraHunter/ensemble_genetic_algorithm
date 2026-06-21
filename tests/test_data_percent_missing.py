import logging
import pickle
import shutil
import tempfile
import unittest

from ml_grid.pipeline.data_percent_missing import handle_percent_missing


class TestHandlePercentMissing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_handle_percent_missing_corrupt_pickle_file(self):
        """Test handling of corrupt pickle file."""
        # Create a corrupted pickle file in current directory
        pickle_path = "percent_missing_dict.pickle"

        with open(pickle_path, "w") as f:
            f.write("not valid pickle data")

        try:
            local_param_dict = {"percent_missing": 0.5}
            all_df_columns = ["col1", "col2"]
            drop_list = []

            result = handle_percent_missing(
                local_param_dict,
                all_df_columns,
                drop_list.copy(),
            )

            self.assertEqual(result, [])
        finally:
            import os

            if os.path.exists(pickle_path):
                os.remove(pickle_path)

    def test_handle_percent_missing_non_numeric_values(self):
        """Test handling of non-numeric values in percent_missing_dict."""
        pickle_path = "percent_missing_dict.pickle"

        # Create a dict with mixed valid and invalid values
        with open(pickle_path, "wb") as f:
            pickle.dump({"col1": 0.5, "col2": "not_a_number"}, f)

        try:
            local_param_dict = {"percent_missing": 0.3}
            all_df_columns = ["col1", "col2"]
            drop_list = []

            result = handle_percent_missing(
                local_param_dict,
                all_df_columns,
                drop_list.copy(),
            )

            self.assertIn("col1", result)
        finally:
            import os

            if os.path.exists(pickle_path):
                os.remove(pickle_path)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()
