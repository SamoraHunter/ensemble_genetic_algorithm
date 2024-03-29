import sklearn.feature_selection
from ml_grid.model_classes_ga.logistic_regression_model import (
    logisticRegressionModelGenerator,
)
from ml_grid.pipeline import read_in
from ml_grid.pipeline.column_names import get_pertubation_columns
from ml_grid.pipeline.data_clean_up import clean_up_class
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
        config_dict=None,
        testing=False,
        multiprocessing_ensemble=False,
    ):  # kwargs**

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

        self.df = read_in.read(file_name).raw_input_data

        if test_sample_n > 0:
            print("sampling 200 for debug/trial purposes...")
            self.df = self.df.sample(test_sample_n)

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

        final_column_list = [
            self.X
            for self.X in self.pertubation_columns
            if (self.X not in self.drop_list)
        ]

        self.X = self.df[final_column_list].copy()

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

        print("------------------------")

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.X_test_orig,
            self.y_test_orig,
        ) = get_data_split(X=self.X, y=self.y, local_param_dict=self.local_param_dict)

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
