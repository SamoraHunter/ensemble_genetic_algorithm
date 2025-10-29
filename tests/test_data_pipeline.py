import unittest
import pandas as pd
import numpy as np
import os
import shutil
import logging
import pickle
from unittest.mock import patch

from ml_grid.pipeline.data import pipe, NoFeaturesError
from ml_grid.util.global_params import global_parameters

# Suppress logging for cleaner test output
logging.basicConfig(level=logging.CRITICAL)

class TestDataPipeline(unittest.TestCase):

    # We patch at the class level or manually in setUp
    # Manual setup is cleaner for complex mocks like 'open'

    def setUp(self):
        """Set up a temporary directory and dummy data for each test."""
        self.test_dir = "temp_test_data_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.file_path = os.path.join(self.test_dir, "test_data.csv")

        # Create a diverse dummy dataset
        data = {
            'age': np.random.randint(20, 80, 100),
            'sex_male': np.random.randint(0, 2, 100),
            'some_blood_test_mean': np.random.rand(100),
            'correlated_blood_test_mean': np.random.rand(100) * 0.95,
            'constant_feature': [1] * 100,
            'missing_feature_std': [np.nan] * 80 + list(np.random.rand(20)),
            'categorical_feature': ['A'] * 50 + ['B'] * 50,
            'outcome_var_1': np.random.randint(0, 2, 100),
            'outcome_var_2': np.random.randint(0, 2, 100),
            'id_column': range(100)
        }
        data['correlated_blood_test_mean'] = data['some_blood_test_mean'] * 0.9 + np.random.rand(100) * 0.1

        self.df = pd.DataFrame(data)
        self.df.to_csv(self.file_path, index=False)

        # Mock the loading of percent_missing_dict.pickle
        self.mock_percent_missing_dict = {
            'missing_feature_std': 80,
            'age': 0,
            'sex_male': 0,
            'some_blood_test_mean': 0,
            'correlated_blood_test_mean': 0,
            'constant_feature': 0,
            'categorical_feature': 0,
            'outcome_var_1': 0,
            'outcome_var_2': 0,
            'id_column': 0,
        }

        # --- Start Patches ---        
        # 1. Patch 'os.path.exists' to return True for the pickle file and config,
        # but let it check the real filesystem for everything else.
        self.original_exists = os.path.exists
        self.patcher_exists = patch('os.path.exists')
        self.mock_exists = self.patcher_exists.start()
        def side_effect_exists(path):
            if os.path.normpath(path) in ["percent_missing_dict.pickle", "config.yml"]:
                return True
            return self.original_exists(path)
        self.mock_exists.side_effect = side_effect_exists

        # 2. Patch 'pickle.load' to return our mock dictionary.
        # We don't need to mock 'open' for this, as 'pickle.load' is what we care about.
        self.patcher_pickle = patch('pickle.load')
        self.mock_pickle_load = self.patcher_pickle.start()
        self.mock_pickle_load.return_value = self.mock_percent_missing_dict

        # 3. Patch 'builtins.open' only to handle the config file gracefully,
        # preventing a FileNotFoundError, without creating a recursive loop.
        self.original_open = open
        self.patcher_open = patch('builtins.open', self.mock_open_custom)
        self.mock_open = self.patcher_open.start()

        # Mock global_params
        self.global_params = global_parameters(testing=True)
        self.global_params.verbose = 1

        # Default local_param_dict
        self.local_param_dict = {
            'outcome_var_n': 1,
            'corr': 0.8,
            'percent_missing': 50,
            'scale': False,
            'n_features': 'all',
            'resample': None,
            'data': {
                'age': True,
                'sex': True,
                'bloods': True,
                'vte_status': False,
            }
        }

        self.drop_term_list = ['id']
        self.base_project_dir = self.test_dir

    def mock_open_custom(self, file, mode='r', **kwargs):
        """A safe mock for 'open' that only intercepts the config file."""
        if file in ['config.yml', 'percent_missing_dict.pickle']:
            # If the config is opened, return an empty file mock.
            # This is enough to prevent FileNotFoundError in the config loader.
            # For the pickle file, this provides a valid file handle for the mocked pickle.load().
            return unittest.mock.mock_open(read_data="")(file, mode, **kwargs)
        # For all other files (like our test CSV), use the real 'open'.
        return self.original_open(file, mode, **kwargs)

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        self.patcher_exists.stop()
        self.patcher_pickle.stop()
        self.patcher_open.stop()
        shutil.rmtree(self.test_dir)

    def test_happy_path_pipeline_execution(self):
        """Test a standard, successful run of the entire pipeline."""

        pipeline = pipe(
            global_params=self.global_params,
            file_name=self.file_path, # This will be read by the REAL 'open'
            drop_term_list=self.drop_term_list,
            local_param_dict=self.local_param_dict,
            base_project_dir=self.base_project_dir,
            param_space_index=0
        )
        self.assertIsInstance(pipeline, pipe)
        self.assertFalse(pipeline.X_train.empty)

    def test_scaling_works(self):
        """Test if the scaling option correctly scales the data."""
        self.local_param_dict['scale'] = True
        pipeline = pipe(
            global_params=self.global_params,
            file_name=self.file_path,
            drop_term_list=self.drop_term_list,
            local_param_dict=self.local_param_dict,
            base_project_dir=self.base_project_dir,
            param_space_index=0
        )

        self.assertAlmostEqual(pipeline.X_train.mean().mean(), 0, places=1)
        self.assertAlmostEqual(pipeline.X_train.std().mean(), 1, places=1)

    def test_feature_importance_selection(self):
        """Test if feature importance selection reduces features to the target number."""
        self.local_param_dict['n_features'] = 1
        pipeline = pipe(
            global_params=self.global_params,
            file_name=self.file_path,
            drop_term_list=self.drop_term_list,
            local_param_dict=self.local_param_dict,
            base_project_dir=self.base_project_dir,
            param_space_index=0
        )
        self.assertEqual(pipeline.X_train.shape[1], 1)

    def test_safety_net_prevents_no_features_error(self):
        """Test that the safety net activates when all features are pruned, preventing an error."""
        self.local_param_dict['corr'] = 0.001
        self.local_param_dict['percent_missing'] = 0
        for key in self.local_param_dict['data']:
            self.local_param_dict['data'][key] = False

        pipeline = pipe(
            global_params=self.global_params,
            file_name=self.file_path,
            drop_term_list=['some', 'correlated', 'constant', 'missing', 'id', 'sex_male', 'categorical_feature'], # Drop everything by name EXCEPT 'age'
            local_param_dict=self.local_param_dict,
            base_project_dir=self.base_project_dir,
            param_space_index=0
        )
        self.assertEqual(pipeline.X_train.shape[1], 1, "Safety net should have retained exactly one feature.")
        self.assertEqual(pipeline.X_train.columns[0], 'age', "Safety net should retain 'age' as the only available numeric column.")

    def test_safety_net_activation(self):
        """Test that the safety net retains features if all would be pruned."""
        self.local_param_dict['corr'] = 0.01 
        self.local_param_dict['percent_missing'] = 0 
        for key in self.local_param_dict['data']:
            self.local_param_dict['data'][key] = False
        
        pipeline = pipe(
            global_params=self.global_params,
            file_name=self.file_path,
            drop_term_list=['categorical_feature'], 
            local_param_dict=self.local_param_dict,
            base_project_dir=self.base_project_dir,
            param_space_index=0
        )
        self.assertGreater(pipeline.X_train.shape[1], 0)
        self.assertIn(pipeline.X_train.shape[1], [1, 2])

    def test_string_column_raises_error_when_not_dropped(self):
        """Test that a string/object column raises a ValueError if it reaches the final processing step."""
        for key in self.local_param_dict['data']:
            self.local_param_dict['data'][key] = False
        
        self.local_param_dict['corr'] = 0.99
        self.local_param_dict['percent_missing'] = 90

        with self.assertRaises(ValueError) as context:
            pipe(
                global_params=self.global_params,
                file_name=self.file_path,
                drop_term_list=['id_column'], 
                local_param_dict=self.local_param_dict,
                base_project_dir=self.base_project_dir,
                param_space_index=0
            )
        self.assertTrue("non-numeric" in str(context.exception) or "string" in str(context.exception))

    def test_no_features_error_raised_on_empty_data(self):
        """Test that NoFeaturesError is raised if the dataset is fundamentally empty of features."""
        empty_df = pd.DataFrame({'outcome_var_1': [0, 1, 0, 1]})
        empty_df.to_csv(self.file_path, index=False)

        for key in self.local_param_dict['data']:
            self.local_param_dict['data'][key] = False

        with self.assertRaises(NoFeaturesError):
            pipe(
                global_params=self.global_params,
                file_name=self.file_path,
                drop_term_list=[],
                local_param_dict=self.local_param_dict,
                base_project_dir=self.base_project_dir,
                param_space_index=0
            )

    def test_feature_transformation_log(self):
        """Test that the feature transformation log is created and populated."""
        pipeline = pipe(
            global_params=self.global_params,
            file_name=self.file_path,
            drop_term_list=self.drop_term_list,
            local_param_dict=self.local_param_dict,
            base_project_dir=self.base_project_dir,
            param_space_index=0
        )

        self.assertIsInstance(pipeline.feature_transformation_log, pd.DataFrame)
        self.assertFalse(pipeline.feature_transformation_log.empty)
        self.assertIn('Initial Load', pipeline.feature_transformation_log['step'].values)
        self.assertIn('Drop Correlated', pipeline.feature_transformation_log['step'].values)
        self.assertIn('Drop Missing', pipeline.feature_transformation_log['step'].values)
        self.assertIn('Drop Constants', pipeline.feature_transformation_log['step'].values)

if __name__ == '__main__':
    unittest.main()