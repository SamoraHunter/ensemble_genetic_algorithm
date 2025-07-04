import sklearn.feature_selection
from ml_grid.model_classes_ga.logistic_regression_model import (
    logisticRegressionModelGenerator,
)
from ml_grid.pipeline import read_in
from ml_grid.pipeline.column_names import get_pertubation_columns
from ml_grid.pipeline.data_clean_up import clean_up_class
from ml_grid.pipeline.data_constant_columns import (
    remove_constant_columns,
    remove_constant_columns_with_debug,
)
from ml_grid.pipeline.data_correlation_matrix import handle_correlation_matrix
from ml_grid.pipeline.data_feature_importance_methods import feature_importance_methods
from ml_grid.pipeline.data_outcome_list import handle_outcome_list
from ml_grid.pipeline.data_percent_missing import handle_percent_missing
from ml_grid.pipeline.data_plot_split import plot_pie_chart_with_counts
from ml_grid.pipeline.data_scale import data_scale_methods
from ml_grid.pipeline.data_train_test_split import *
from ml_grid.pipeline.logs_project_folder import log_folder
from ml_grid.util.global_params import global_parameters
from ml_grid.util.project_score_save import project_score_save_class
from sklearn.exceptions import ConvergenceWarning
from tabulate import tabulate

ConvergenceWarning("ignore")

from warnings import filterwarnings

filterwarnings("ignore")

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


class pipe:
    """
    function should take settings iteration...
    _Function takes input csv of type KCH cogstack, outputs _

    Function returns ml_grid_[data]_object, this is a permutation from the feature space

    This object can be used to pass to classifier methods
    """

    def __init__(
        self,
        file_name,
        drop_term_list,
        local_param_dict,
        base_project_dir,
        param_space_index,
        additional_naming=None,
        test_sample_n=0,
        column_sample_n=0,
        config_dict=None,
        testing=False,
        multiprocessing_ensemble=False,
    ):  # kwargs**
        """
        This class is the main entry point for the ml_grid pipeline. It takes in a number of parameters which specify the
        data, the desired outcome, and the global parameters for the project. It then does the following:

            1. Read in the input data
            2. Remove any columns specified in the drop_term_list
            3. Remove any columns which have a percentage of missing values greater than the threshold specified in
               local_param_dict['missing_data_threshold']
            4. Remove any duplicate columns
            5. Handle any correlation between columns by removing any columns which have a correlation greater than the
               threshold specified in local_param_dict['correlation_threshold']
            6. Split the data into training and testing sets
            7. Optionally performs data scaling (e.g. standardization)
            8. Optionally performs feature selection to target a specific number of features
            9. Saves the results of each of the above steps

        The resulting object is then used to pass to the classifier methods.

        :param file_name: The path to the input data
        :param drop_term_list: A list of terms to be removed from the data
        :param local_param_dict: A dictionary containing the local parameters for the project
        :param base_project_dir: The base directory of the project
        :param param_space_index: The index of the parameter space iteration
        :param additional_naming: Any additional strings to be added to the naming of the output files
        :param test_sample_n: The number of rows to sample from the data for testing purposes
        :param column_sample_n: The number of columns to sample from the data for testing purposes
        :param config_dict: A dictionary of configuration options
        :param testing: Whether to run in testing mode or not
        :param multiprocessing_ensemble: Whether to use multiprocessing for ensemble methods
        """

        self.testing = testing

        self.multiprocessing_ensemble = multiprocessing_ensemble

        self.base_project_dir = base_project_dir

        self.additional_naming = additional_naming

        self.local_param_dict = local_param_dict

        self.global_params = global_parameters()

        self.verbose = self.global_params.verbose

        self.param_space_index = param_space_index

        self.project_score_save_object = project_score_save_class(
            base_project_dir=base_project_dir
        )

        self.config_dict = config_dict
        if self.config_dict == None:
            self.config_dict = {
                "use_stored_base_learners": False,
            }
            if self.verbose >= 1:
                print(f"Using default config_dict... {self.config_dict}")

        if self.verbose >= 1:
            print(f"Starting... {self.local_param_dict}")

        self.logging_paths_obj = log_folder(
            local_param_dict=local_param_dict,
            additional_naming=additional_naming,
            base_project_dir=base_project_dir,
        )
        read_in_sample = True

        if read_in_sample and test_sample_n > 0 or column_sample_n > 0:
            self.df = read_in.read_sample(
                file_name, test_sample_n, column_sample_n
            ).raw_input_data
        else:
            self.df = read_in.read(file_name, use_polars=True).raw_input_data

        if test_sample_n > 0 and read_in_sample == False:
            print("sampling 200 for debug/trial purposes...")
            self.df = self.df.sample(test_sample_n)

        if column_sample_n > 0 and read_in_sample == False:
            # Check if 'age' and 'male' columns are in the original DataFrame
            if (
                "age" in self.df.columns
                and "male" in self.df.columns
                and "outcome_var_1" in self.df.columns
            ):
                original_columns = ["age", "male", "outcome_var_1"]
            else:
                original_columns = []

            print(
                "Sampling",
                column_sample_n,
                "columns for additional debug/trial purposes...",
            )

            # Sample the columns
            sampled_columns = self.df.sample(n=column_sample_n, axis=1).columns

            # Ensure original columns are retained
            new_columns = list(set(sampled_columns) | set(original_columns))

            # Reassign DataFrame with sampled columns
            self.df = self.df[new_columns].copy()

            print("Result df shape", self.df.shape)

        self.all_df_columns = list(self.df.columns)

        self.orignal_feature_names = self.all_df_columns.copy()

        self.pertubation_columns, self.drop_list = get_pertubation_columns(
            all_df_columns=self.all_df_columns,
            local_param_dict=local_param_dict,
            drop_term_list=drop_term_list,
        )

        self.outcome_variable = f'outcome_var_{local_param_dict.get("outcome_var_n")}'

        print(
            f"Using {len(self.pertubation_columns)}/{len(self.all_df_columns)} columns for {self.outcome_variable} outcome"
        )

        list_2 = self.df.columns
        list_1 = self.pertubation_columns.copy()

        difference_list = list(set(list_2) - set(list_1))
        print(f"Omitting {len(difference_list)} :...")
        print(f"{difference_list[0:5]}...")

        self.drop_list = handle_correlation_matrix(
            local_param_dict=local_param_dict, drop_list=self.drop_list, df=self.df
        )

        self.drop_list = handle_percent_missing(
            local_param_dict=local_param_dict,
            all_df_columns=self.all_df_columns,
            drop_list=self.drop_list,
        )

        self.drop_list = handle_outcome_list(
            drop_list=self.drop_list, outcome_variable=self.outcome_variable
        )

        self.drop_list = remove_constant_columns(
            X=self.df, drop_list=self.drop_list, verbose=self.verbose
        )

        self.final_column_list = [
            self.X
            for self.X in self.pertubation_columns
            if (self.X not in self.drop_list)
        ]

        # Add safety mechanism to retain minimum features
        min_required_features = 5  # Set your minimum threshold
        core_protected_columns = ["age", "male", "client_idcode"]  # Columns to protect

        if not self.final_column_list:
            print("WARNING: All features pruned! Activating safety retention...")

            # Try to keep protected columns first
            safety_columns = [
                col
                for col in core_protected_columns
                if col in self.df.columns and col in self.pertubation_columns
            ]

            # If no protected columns, use first available columns
            if not safety_columns:
                safety_columns = [
                    col for col in self.pertubation_columns if col in self.df.columns
                ][:min_required_features]

            # Update final columns and drop list
            self.final_column_list = safety_columns
            self.drop_list = [
                col for col in self.drop_list if col not in self.final_column_list
            ]

            print(f"Retaining minimum features: {self.final_column_list}")

            # Add two random features if list still empty
            if not self.final_column_list:
                print("Warning no feature columns retained, selecting two at random")
                final_column_list = []
                final_column_list.append(random.choice(self.orignal_feature_names))
                final_column_list.append(random.choice(self.orignal_feature_names))

        # Ensure we still have at least 1 feature
        if not self.final_column_list:
            raise ValueError(
                "CRITICAL: Unable to retain any features despite safety measures"
            )

        if not self.final_column_list:
            raise ValueError(
                "All features pruned. No columns remaining in final_column_list."
            )

        self.final_column_list = [
            col for col in self.final_column_list if col in self.df.columns
        ]

        self.X = self.df[self.final_column_list].copy()

        self.X = clean_up_class().handle_duplicated_columns(self.X)

        clean_up_class().screen_non_float_types(self.X)

        self.y = self.df[self.outcome_variable].copy()

        clean_up_class().handle_column_names(self.X)

        scale = self.local_param_dict.get("scale")

        if scale:

            self.X = data_scale_methods().standard_scale_method(self.X)

        if self.verbose >= 1:
            print(
                f"len final droplist: {len(self.drop_list)} \ {len(list(self.df.columns))}"
            )
            # print('\n'.join(map(str, self.drop_list[0:5])))

        # screen for nan and string variables, raise error if found in self.X

        # Check for NaNs
        if self.X.isnull().values.any():
            raise ValueError("DataFrame contains NaN values.")

        # Check for string (object or string) columns or values
        if self.X.select_dtypes(include=["object", "string"]).shape[1] > 0:
            raise ValueError("DataFrame contains string (non-numeric) columns.")

        # Optionally: check for any string values hidden in numeric-looking columns
        if self.X.applymap(lambda x: isinstance(x, str)).any().any():
            raise ValueError("DataFrame contains string values within numeric columns.")

        print("------------------------")

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.X_test_orig,
            self.y_test_orig,
        ) = get_data_split(X=self.X, y=self.y, local_param_dict=self.local_param_dict)

        # Handle columns made constant by splitting
        self.X_train, self.X_test, self.X_test_orig = (
            remove_constant_columns_with_debug(
                self.X_train, self.X_test, self.X_test_orig, verbosity=self.verbose
            )
        )
        target_n_features = self.local_param_dict.get("n_features")

        if target_n_features != "all":

            self.X_train, self.X_test, self.X_test_orig = (
                feature_importance_methods.handle_feature_importance_methods(
                    self,
                    target_n_features,
                    X_train=self.X_train,
                    X_test=self.X_test,
                    y_train=self.y_train,
                    X_test_orig=self.X_test_orig,
                    ml_grid_object=self,
                )
            )

        if self.verbose >= 2:
            print(
                f"Data Split Information:\n"
                f"Number of rows in self.X_train: {len(self.X_train)}, Columns: {self.X_train.shape[1]}\n"
                f"Number of rows in self.X_test: {len(self.X_test)}, Columns: {self.X_test.shape[1]}\n"
                f"Number of rows in self.y_train: {len(self.y_train)}\n"
                f"Number of rows in self.y_test: {len(self.y_test)}\n"
                f"Number of rows in self.X_test_orig: {len(self.X_test_orig)}, Columns: {self.X_test_orig.shape[1]}\n"
                f"Number of rows in self.y_test_orig: {len(self.y_test_orig)}"
            )

        if self.verbose >= 3:

            plot_pie_chart_with_counts(self.X_train, self.X_test, self.X_test_orig)

        # load in here for passing

        self.model_class_list = [
            # LogisticRegression_class(X=self.ml_grid_object.X_train, y=self.ml_grid_object.y_train,
            #                       parameter_space_size=self.parameter_space_size),
            logisticRegressionModelGenerator(self, self.local_param_dict)
        ]
