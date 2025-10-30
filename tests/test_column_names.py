import logging
import unittest

from ml_grid.pipeline.column_names import get_pertubation_columns

# Suppress logging during tests to keep the output clean
logging.disable(logging.CRITICAL)


class TestGetPertubationColumns(unittest.TestCase):
    def setUp(self):
        """Set up common variables for all test cases."""
        # A dataset with columns that match the specific suffix logic
        self.all_df_columns_specific = [
            "age",
            "male",
            "blood_test_mean",
            "blood_test_median",
            "vte_status_mean",  # This has a blood suffix but is also a category
            "outcome_var_1",
            "outcome_var_2",
            "id_to_drop",
            "Unnamed: 0",
        ]
        # A generic dataset with no matching suffixes
        self.all_df_columns_generic = [
            "feature1",
            "feature2",
            "feature3",
            "outcome_var_1",
            "id_to_drop",
        ]

        self.drop_term_list = ["id_to_drop"]

        # Config to select specific features
        self.local_param_dict_specific = {
            "data": {
                "age": True,
                "sex": True,
                "bloods": True,
                "vte_status": False,  # Explicitly disable vte_status
                "diagnostic_order": False,
                "drug_order": False,
                "bmi": False,
                "ethnicity": False,
                "annotation_n": False,
                "meta_sp_annotation_n": False,
                "annotation_mrc_n": False,
                "meta_sp_annotation_mrc_n": False,
                "core_02": False,
                "bed": False,
                "hosp_site": False,
                "core_resus": False,
                "news": False,
                "date_time_stamp": False,
                "appointments": False,
            },
            "outcome_var_n": 1,
        }

        # Config that will select no features, triggering the fallback
        self.local_param_dict_fallback = {
            "data": {key: False for key in self.local_param_dict_specific["data"]},
            "outcome_var_n": 1,
        }

    def tearDown(self):
        """Re-enable logging after tests are done."""
        logging.disable(logging.NOTSET)

    def test_no_fallback_with_specific_data(self):
        """Test that fallback is NOT triggered with a specific dataset."""
        pert_cols, _ = get_pertubation_columns(
            self.all_df_columns_specific,
            self.local_param_dict_specific,
            self.drop_term_list,
        )
        # Should select 'age', 'male', 'blood_test_mean', 'blood_test_median'
        # 'vte_status_mean' is excluded because its category is false and the post-processing removes it from bloods
        self.assertCountEqual(
            pert_cols, ["age", "male", "blood_test_mean", "blood_test_median"]
        )

    def test_fallback_triggered_with_generic_data(self):
        """Test that fallback IS triggered for a generic dataset."""
        # Use a config that selects nothing to force the fallback
        pert_cols, _ = get_pertubation_columns(
            self.all_df_columns_generic, self.local_param_dict_fallback, []
        )
        # Should select all columns except the outcome variable
        # and columns that would be in the initial drop list (like 'id_to_drop').
        # The initial drop list is built inside the function, so we test its effect here.
        self.assertCountEqual(
            pert_cols, ["feature1", "feature2", "feature3", "id_to_drop"]
        )

    def test_fallback_respects_initial_drop_list(self):
        """Test that the fallback excludes columns from the provided drop_term_list."""
        # The function internally finds 'id_to_drop' from drop_term_list and adds it to its drop_list.
        # The fallback logic should then exclude it.
        pert_cols, drop_list = get_pertubation_columns(
            self.all_df_columns_generic,
            self.local_param_dict_fallback,
            self.drop_term_list,  # Contains 'id_to_drop'
        )

        # The internal drop_list should contain 'id_to_drop'
        self.assertIn("id_to_drop", drop_list)

        # The final perturbation columns should contain the features, but not the outcome or the dropped ID.
        self.assertCountEqual(pert_cols, ["feature1", "feature2", "feature3"])
        self.assertNotIn("id_to_drop", pert_cols)

    def test_fallback_with_empty_columns(self):
        """Test that the function handles an empty list of columns gracefully."""
        pert_cols, _ = get_pertubation_columns([], self.local_param_dict_fallback, [])
        self.assertEqual(pert_cols, [])


if __name__ == "__main__":
    unittest.main()
