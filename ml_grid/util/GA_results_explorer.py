import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import math
import itertools
import numpy as np
import ast
import re
import os
from typing import List, Optional, Tuple

# the plots should use the feature names which we extract in the init method. these are a nested list of lists of features. each corresponds to each base learner in order they appear in best_ensemble


# Example ensemble
# '[[(0.6591241898175125, "LogisticRegression(C=100000.0, class_weight=\'balanced\', max_iter=12,\\n                   solver=\'sag\')", [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 0, 0.8308, array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1,\n       0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1])), (0.3670372447278536, \'Perceptron(eta0=0.1, max_iter=5)\', [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 0, 0.654, array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,\n       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1]))]]'
# binary feature map [0, 1, 0 ...1] maps to original feature names
class GA_results_explorer:
    """A comprehensive toolkit for analyzing and visualizing GA results.

    This class takes a DataFrame of GA results and provides numerous methods
    to plot and analyze the data, helping to understand feature importance,
    hyperparameter sensitivity, performance trade-offs, and ensemble composition.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing the GA run results.
        original_feature_names (List[str]): A list of the original feature names
            from the dataset, used for decoding feature masks.
        config_params (List[str]): A predefined list of hyperparameter names
            to be used in importance analysis.
        run_details (List[str]): A predefined list of run metadata column names
            to be used in importance analysis.
    """

    def __init__(self, df: pd.DataFrame, original_feature_names: List[str]):
        """Initializes the GA_results_explorer object.

        This constructor processes the input DataFrame to decode the feature
        sets used in each ensemble, making them available for analysis.

        Args:
            df: A DataFrame containing the results of a genetic algorithm search.
                It must contain a 'best_ensemble' column.
            original_feature_names: The complete list of original feature
                names from the dataset, used to map binary feature masks back
                to their string names.
        """
        self.df = df
        self.original_feature_names = original_feature_names

        # Extract feature arrays from the 'best_ensemble' column
        self.df["feature_arrays"] = self.df["best_ensemble"].apply(
            extract_feature_arrays_from_string
        )

        # Use the feature arrays as a map on the original feature names
        self.df["feature_names"] = self.df["feature_arrays"].apply(
            lambda x: [
                [
                    f
                    for i, f in enumerate(self.original_feature_names)
                    if i < len(arr) and arr[i] == 1
                ]
                for arr in x
            ]
        )

        print("Feature names extracted for all ensembles.")

        self.config_params = [
            "nb_size",
            "nb_val",
            "pop_val",
            "g_val",
            "g",
            "weighted",
            "use_stored_base_learners",
            "store_base_learners",
            "resample",
            "scale",
            "n_features",
            "param_space_size",
            "cxpb",
            "mutpb",
            "indpb",
            "t_size",
        ]

        self.run_details = [
            "n_unique_out",
            "outcome_var_n",
            "div_p",
            "percent_missing",
            "corr",
            "age",
            "sex",
            "bmi",
            "ethnicity",
            "bloods",
            "diagnostic_order",
            "drug_order",
            "annotation_n",
            "meta_sp_annotation_n",
            "meta_sp_annotation_mrc_n",
            "annotation_mrc_n",
            "core_02",
            "bed",
            "vte_status",
            "hosp_site",
            "core_resus",
            "news",
            "date_time_stamp",
            "X_train_size",
            "X_test_orig_size",
            "X_test_size",
            "run_time",
        ]

    def get_column_names(self, raw_string_vector: str) -> List[str]:
        """Decodes a string representation of a binary feature mask.

        This method is a utility for manually inspecting feature masks from the
        results DataFrame.

        Args:
            raw_string_vector: A string representing a list of binary integers,
                e.g., '[0, 1, 0, 1]'.

        Returns:
            A list of feature names corresponding to the '1's in the mask.
            Returns an empty list if the input is not a valid string.
        """
        if isinstance(raw_string_vector, str):
            int_f_list = list(
                map(
                    int,
                    raw_string_vector.strip("[").strip("]").replace(" ", "").split(","),
                )
            )
            res = [b for a, b in zip(int_f_list, self.original_feature_names) if a]
            return res
        else:
            return []

    def plot_config_anova_feature_importances(
        self, outcome_variable: str = "auc", plot_dir: Optional[str] = None
    ) -> None:
        """Plots the importance of configuration parameters using ANOVA.

        This method iterates through the hyperparameters defined in self.config_params,
        treating each as a categorical independent variable and the specified
        outcome_variable as the dependent variable. It calculates the F-statistic
        and p-value for each parameter.

        The results are visualized as a sorted horizontal bar plot, showing the
        relative importance of each configuration parameter.

        Args:
            outcome_variable: The column name of the outcome
                variable in self.df to be used as the dependent variable in the
                ANOVA test. Defaults to 'auc'.
            plot_dir: Directory to save the plot. If None,
                the plot is only displayed. Defaults to None.
        """
        # Check if the outcome variable exists in the dataframe
        if outcome_variable not in self.df.columns:
            print(
                f"❌ Error: Outcome variable '{outcome_variable}' not found in the DataFrame."
            )
            return

        anova_results = []
        print(
            f"📊 Performing ANOVA F-tests for parameters against '{outcome_variable}'..."
        )

        # Iterate over each configuration parameter to perform ANOVA
        for param in self.config_params:
            # ANOVA is only possible if the parameter exists and has at least two groups to compare
            if param in self.df.columns and self.df[param].nunique() > 1:
                try:
                    # Create the model formula using the C() function to treat the parameter as categorical
                    formula = f"{outcome_variable} ~ C({param})"

                    # Fit the Ordinary Least Squares (OLS) model
                    model = ols(formula, data=self.df).fit()

                    # Perform ANOVA on the fitted model
                    anova_table = sm.stats.anova_lm(model, typ=2)

                    # Extract F-statistic and p-value for the parameter from the anova table
                    f_value = anova_table["F"][0]
                    p_value = anova_table["PR(>F)"][0]

                    anova_results.append(
                        {"Parameter": param, "F-statistic": f_value, "p-value": p_value}
                    )
                except Exception as e:
                    print(f"⚠️ Could not perform ANOVA for parameter '{param}': {e}")
            else:
                print(f"⏩ Skipping '{param}': Not enough unique values for ANOVA.")

        # Check if any results were generated
        if not anova_results:
            print(
                "No ANOVA results to plot. This may be because no parameters had sufficient unique values."
            )
            return

        # Convert results to a DataFrame for easier handling and plotting
        results_df = pd.DataFrame(anova_results)

        # Sort the parameters by F-statistic in descending order for the plot
        results_df = results_df.sort_values(
            by="F-statistic", ascending=False
        ).reset_index(drop=True)

        # --- Plotting ---
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(12, 10))

        # Create the bar plot
        barplot = sns.barplot(
            x="F-statistic",
            y="Parameter",
            data=results_df,
            palette="viridis",
            orient="h",
        )

        # Set plot title and labels
        plt.title(
            f"Importance of Configuration Parameters on {outcome_variable.upper()} (ANOVA F-test)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("F-statistic (Higher = More Important)", fontsize=12)
        plt.ylabel("Hyperparameter", fontsize=12)

        # Add data labels (F-statistic values) to the bars for clarity
        for patch in barplot.patches:
            plt.text(
                patch.get_width() * 1.01,  # x-coordinate
                patch.get_y() + patch.get_height() / 2,  # y-coordinate
                f"{patch.get_width():.2f}",  # Text label
                va="center",  # Vertical alignment
                fontsize=10,
                color="dimgray",
            )

        plt.tight_layout()
        if plot_dir:
            try:
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig(
                    os.path.join(plot_dir, f"config_anova_{outcome_variable}.png"),
                    bbox_inches="tight",
                )
                print(
                    f"📈 Plot saved to {os.path.join(plot_dir, f'config_anova_{outcome_variable}.png')}"
                )
            except Exception as e:
                print(f"⚠️ Could not save plot to '{plot_dir}': {e}")
        plt.show()

        # Print the detailed results table
        print("\nANOVA F-test Results (sorted by importance):")
        print(results_df.to_string())

    def plot_run_details_anova_feature_importances(
        self, outcome_variable: str = "auc", plot_dir: Optional[str] = None
    ) -> None:
        """Plots the importance of run metadata using ANOVA.

        This method iterates through the dataset and run metadata defined in
        self.run_details, treating each as a categorical independent variable
        and the specified outcome_variable as the dependent variable. It calculates
        the F-statistic and p-value for each detail.

        The results are visualized as a sorted horizontal bar plot.

        Args:
            outcome_variable: The column name of the outcome
                variable in self.df to be used as the dependent variable in the
                ANOVA test. Defaults to 'auc'.
        """
        # Check if the outcome variable exists in the dataframe
        if outcome_variable not in self.df.columns:
            print(
                f"❌ Error: Outcome variable '{outcome_variable}' not found in the DataFrame."
            )
            return

        anova_results = []
        print(
            f"📊 Performing ANOVA F-tests for run details against '{outcome_variable}'..."
        )

        # Iterate over each run detail to perform ANOVA
        for detail in self.run_details:
            # ANOVA is only possible if the detail exists and has at least two groups to compare
            if detail in self.df.columns and self.df[detail].nunique() > 1:
                try:
                    # Create the model formula using the C() function to treat the detail as categorical
                    formula = f"{outcome_variable} ~ C({detail})"

                    # Fit the Ordinary Least Squares (OLS) model
                    model = ols(formula, data=self.df).fit()

                    # Perform ANOVA on the fitted model
                    anova_table = sm.stats.anova_lm(model, typ=2)

                    # Extract F-statistic and p-value
                    f_value = anova_table["F"][0]
                    p_value = anova_table["PR(>F)"][0]

                    anova_results.append(
                        {
                            "Run Detail": detail,
                            "F-statistic": f_value,
                            "p-value": p_value,
                        }
                    )
                except Exception as e:
                    print(f"⚠️ Could not perform ANOVA for run detail '{detail}': {e}")
            else:
                print(
                    f"⏩ Skipping '{detail}': Not in DataFrame or not enough unique values for ANOVA."
                )

        # Check if any results were generated
        if not anova_results:
            print(
                "No ANOVA results to plot. This may be because no run details had sufficient unique values."
            )
            return

        # Convert results to a DataFrame for easier handling and plotting
        results_df = pd.DataFrame(anova_results)

        # Sort by F-statistic in descending order for the plot
        results_df = results_df.sort_values(
            by="F-statistic", ascending=False
        ).reset_index(drop=True)

        # --- Plotting ---
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(12, 10))

        # Create the bar plot
        barplot = sns.barplot(
            x="F-statistic", y="Run Detail", data=results_df, palette="mako", orient="h"
        )

        # Set plot title and labels
        plt.title(
            f"Impact of Run Details on {outcome_variable.upper()} (ANOVA F-test)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("F-statistic (Higher = More Impact)", fontsize=12)
        plt.ylabel("Run Detail / Metadata", fontsize=12)

        # Add data labels (F-statistic values) to the bars
        for patch in barplot.patches:
            plt.text(
                patch.get_width() * 1.01,
                patch.get_y() + patch.get_height() / 2,
                f"{patch.get_width():.2f}",
                va="center",
                fontsize=10,
                color="dimgray",
            )

        plt.tight_layout()

        if plot_dir:
            plt.savefig(
                os.path.join(plot_dir, f"anova_run_details_{outcome_variable}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        plt.show()

        # Print the detailed results table
        print("\nANOVA F-test Results (sorted by importance):")
        print(results_df.to_string())

    def plot_combined_anova_feature_importances(
        self, outcome_variable: str = "auc", plot_dir: Optional[str] = None
    ) -> None:
        """Plots the combined importance of all parameters using ANOVA.

        This method calculates the F-statistic for every item in both
        self.config_params and self.run_details against the outcome variable.
        The results are displayed on one sorted bar chart, with colors
        distinguishing between Hyperparameters and Run Details.

        Args:
            outcome_variable: The column name of the outcome
                variable in self.df to use for the ANOVA test. Defaults to 'auc'.
        """
        if outcome_variable not in self.df.columns:
            print(
                f"❌ Error: Outcome variable '{outcome_variable}' not found in the DataFrame."
            )
            return

        all_results = []
        param_lists = {
            "Hyperparameter": self.config_params,
            "Run Detail": self.run_details,
        }

        print(f"📊 Performing combined ANOVA F-tests against '{outcome_variable}'...")

        # Loop through both parameter types (Hyperparameter and Run Detail)
        for param_type, param_list in param_lists.items():
            for param_name in param_list:
                if param_name in self.df.columns and self.df[param_name].nunique() > 1:
                    try:
                        formula = f"{outcome_variable} ~ C({param_name})"
                        model = ols(formula, data=self.df).fit()
                        anova_table = sm.stats.anova_lm(model, typ=2)

                        all_results.append(
                            {
                                "Parameter": param_name,
                                "F-statistic": anova_table["F"][0],
                                "p-value": anova_table["PR(>F)"][0],
                                "Type": param_type,  # Assign the type for color-coding
                            }
                        )
                    except Exception as e:
                        print(f"⚠️ Could not process '{param_name}': {e}")
                else:
                    print(
                        f"⏩ Skipping '{param_name}': Not in DataFrame or not enough unique values."
                    )

        if not all_results:
            print("No ANOVA results to plot.")
            return

        # Create and sort the combined DataFrame
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(
            by="F-statistic", ascending=False
        ).reset_index(drop=True)

        # --- Plotting ---
        plt.style.use("seaborn-v0_8-whitegrid")
        # Increase figure height to accommodate all parameters
        plt.figure(figsize=(14, 16))

        # Create the bar plot, using 'hue' to color-code the bars by 'Type'
        barplot = sns.barplot(
            x="F-statistic",
            y="Parameter",
            hue="Type",
            data=results_df,
            orient="h",
            dodge=False,  # Prevents bars from being shifted side-by-side
            palette={
                "Hyperparameter": "#3498db",
                "Run Detail": "#f1c40f",
            },  # Custom colours
        )

        # Set plot title and labels
        plt.title(
            f"Combined Importance of All Parameters on {outcome_variable.upper()} (ANOVA F-test)",
            fontsize=18,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("F-statistic (Higher = More Impact)", fontsize=14)
        plt.ylabel("Parameter / Run Detail", fontsize=14)
        plt.tick_params(axis="y", labelsize=12)
        plt.tick_params(axis="x", labelsize=12)

        # Enhance the legend
        plt.legend(title="Parameter Type", fontsize=12, title_fontsize=14)

        plt.tight_layout()

        if plot_dir:
            plt.savefig(
                os.path.join(plot_dir, f"combined_anova_{outcome_variable}.png"),
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()
        plt.close()

        # Print the detailed results table
        print("\nCombined ANOVA F-test Results (sorted by importance):")
        print(results_df.to_string())

    def plot_initial_feature_importance(self, outcome_variable="auc", plot_dir=None):
        """
        Analyzes the importance of each initial feature based on its impact on the
        outcome variable.

        For each unique feature across all 'f_list' sets, this method performs
        an ANOVA test. It compares the outcome variable of runs that included
        the feature against those that did not. The resulting F-statistic is used
        as the importance score.

        Args:
            outcome_variable (str, optional): The performance metric to use as the
                dependent variable in the ANOVA test. Defaults to 'auc'.
        """
        if outcome_variable not in self.df.columns:
            print(
                f"❌ Error: Outcome variable '{outcome_variable}' not found in the DataFrame."
            )
            return

        print(
            f"📊 Calculating importance for initial features against '{outcome_variable}'..."
        )

        import ast

        def decode_flist(flist_row):
            # If already a list of feature names (all str, not numbers or brackets), return as is
            if isinstance(flist_row, list) and all(
                isinstance(x, str) and x not in {",", "[", "]", " "} for x in flist_row
            ):
                # Defensive: filter out any non-feature tokens
                return [x for x in flist_row if x in self.original_feature_names]
            # If it's a string, try to parse as a list of ints (mask) or as a space-separated 0/1 string
            if isinstance(flist_row, str):
                # Try to parse as a list (e.g. '[0, 1, 0, 1]')
                try:
                    parsed = ast.literal_eval(flist_row)
                    if isinstance(parsed, list):
                        flist_row = parsed
                except Exception:
                    # If not a list, try to parse as space-separated 0/1 string
                    try:
                        mask = [
                            int(x) for x in flist_row.strip().split() if x in {"0", "1"}
                        ]
                        if len(mask) == len(self.original_feature_names):
                            return [
                                f
                                for i, f in enumerate(self.original_feature_names)
                                if mask[i]
                            ]
                    except Exception:
                        return []
            # If it's a nested list (e.g., [[0, 1, 1, ...]]), flatten it
            if (
                isinstance(flist_row, list)
                and len(flist_row) == 1
                and isinstance(flist_row[0], list)
            ):
                flist_row = flist_row[0]
            # If it's a list of ints (mask) and matches the length of original_feature_names
            if isinstance(flist_row, list) and len(flist_row) == len(
                self.original_feature_names
            ):
                try:
                    # If the list is a mask of 0/1 integers
                    if all(isinstance(x, (int, np.integer)) for x in flist_row):
                        return [
                            f
                            for i, f in enumerate(self.original_feature_names)
                            if flist_row[i]
                        ]
                    # If the list is a mask of 0/1 strings
                    if all(isinstance(x, str) and x in {"0", "1"} for x in flist_row):
                        return [
                            f
                            for i, f in enumerate(self.original_feature_names)
                            if int(flist_row[i])
                        ]
                except Exception:
                    return []
            return []

        decoded_feature_lists = self.df["f_list"].apply(decode_flist)

        # 2. Get a flat list of all unique decoded features
        try:
            all_features_flat = [
                feature for sublist in decoded_feature_lists for feature in sublist
            ]
            unique_features = sorted(list(set(all_features_flat)))
            if not unique_features:
                print("No features found in the 'f_list' column after decoding.")
                return
        except TypeError:
            print(
                "❌ Error: Could not process 'f_list'. Ensure it contains lists of binary/int masks or feature names."
            )
            return

        # 3. Perform ANOVA for each feature
        anova_results = []
        temp_df = self.df[[outcome_variable]].copy()
        temp_df["decoded_f_list"] = decoded_feature_lists

        for feature in unique_features:
            # Create a temporary boolean column: True if feature is in decoded_f_list, else False
            temp_df["has_feature"] = temp_df["decoded_f_list"].apply(
                lambda x: feature in x
            )

            # A test is only possible if we have both groups (with and without the feature)
            if temp_df["has_feature"].nunique() < 2:
                print(
                    f"⏩ Skipping '{feature}': Feature is always present or always absent."
                )
                continue

            try:
                formula = f"{outcome_variable} ~ C(has_feature)"
                model = ols(formula, data=temp_df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)

                anova_results.append(
                    {
                        "Feature": feature,
                        "F-statistic": anova_table["F"][0],
                        "p-value": anova_table["PR(>F)"][0],
                    }
                )
            except Exception as e:
                print(f"⚠️ Could not perform ANOVA for '{feature}': {e}")

        if not anova_results:
            print("No feature importance results to plot.")
            return

        # 4. Create DataFrame and plot the results
        results_df = pd.DataFrame(anova_results)
        results_df = results_df.sort_values(
            by="F-statistic", ascending=False
        ).reset_index(drop=True)

        # --- Plotting ---
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(12, max(8, len(results_df) * 0.4)))  # Dynamic height

        barplot = sns.barplot(
            x="F-statistic", y="Feature", data=results_df, palette="crest", orient="h"
        )

        plt.title(
            f"Importance of Initial Features on {outcome_variable.upper()} (ANOVA F-test)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("F-statistic (Higher = More Impact)", fontsize=12)
        plt.ylabel("Initial Feature", fontsize=12)

        plt.tight_layout()

        if plot_dir is not None:
            plot_path = os.path.join(plot_dir, "initial_feature_importance.png")
            plt.savefig(plot_path)
            print(f"✅ Feature importance plot saved to: {plot_path}")

        plt.show()
        plt.close()

        # 5. Print the results table
        print("\nInitial Feature Importance (sorted):")
        print(results_df.to_string())

    def plot_base_learner_feature_importance(
        self, outcome_variable: str = "auc", plot_dir: Optional[str] = None
    ) -> None:
        """Plots the importance of features used in base learners using ANOVA.

        This method first aggregates the feature sets from all 'BL_n' columns
        for each run. It then performs an ANOVA test for each unique feature,
        comparing the outcome of runs where the feature was used in at least one
        base learner against runs where it was not. The resulting F-statistic
        is used as the importance score.

        Args:
            outcome_variable: The performance metric to use as the
                dependent variable. Defaults to 'auc'.
        """
        if outcome_variable not in self.df.columns:
            print(
                f"❌ Error: Outcome variable '{outcome_variable}' not found in the DataFrame."
            )
            return

        feature_names_col = "feature_names"
        if feature_names_col not in self.df.columns:
            print(
                f"❌ Error: Feature column '{feature_names_col}' not found. Was the explorer initialized correctly?"
            )
            return

        print(f"📊 Aggregating features from 'feature_names' column...")
        temp_df = self.df[[outcome_variable, feature_names_col]].copy()

        # For each run, combine all features from its base learners into a single set
        def combine_bl_features_from_names(row):
            feature_lists = row[feature_names_col]
            if not isinstance(feature_lists, list):
                return set()
            all_features = set()
            for feature_list in feature_lists:
                if isinstance(feature_list, list):
                    all_features.update(feature_list)
            return all_features

        temp_df["all_bl_features"] = temp_df.apply(
            combine_bl_features_from_names, axis=1
        )

        # Get a unique list of all features used across all runs and all base learners
        try:
            all_features_flat = [
                feature
                for feature_set in temp_df["all_bl_features"]
                for feature in feature_set
            ]
            unique_features = sorted(list(set(all_features_flat)))
            if not unique_features:
                print("No features found in any 'feature_names' entries.")
                return
        except TypeError:
            print(
                "❌ Error: Could not process 'feature_names'. Ensure it contains lists of strings."
            )
            return

        # 4. Perform ANOVA for each unique feature
        print(
            f"📈 Calculating importance for {len(unique_features)} unique base learner features..."
        )
        anova_results = []
        for feature in unique_features:
            # Create a temporary boolean column: True if feature was used in any BL
            temp_df["has_feature"] = temp_df["all_bl_features"].apply(
                lambda f_set: feature in f_set
            )

            if temp_df["has_feature"].nunique() < 2:
                print(
                    f"⏩ Skipping '{feature}': Feature is always present or always absent in base learners."
                )
                continue

            try:
                formula = f"{outcome_variable} ~ C(has_feature)"
                model = ols(formula, data=temp_df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                anova_results.append(
                    {
                        "Feature": feature,
                        "F-statistic": anova_table["F"][0],
                        "p-value": anova_table["PR(>F)"][0],
                    }
                )
            except Exception as e:
                print(f"⚠️ Could not perform ANOVA for '{feature}': {e}")

        if not anova_results:
            print("No feature importance results to plot.")
            return

        # 5. Create DataFrame and plot the results
        results_df = pd.DataFrame(anova_results)
        results_df = results_df.sort_values(
            by="F-statistic", ascending=False
        ).reset_index(drop=True)

        # --- Plotting ---
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(12, max(8, len(results_df) * 0.4)))  # Dynamic height

        barplot = sns.barplot(
            x="F-statistic", y="Feature", data=results_df, palette="rocket", orient="h"
        )

        plt.title(
            f"Importance of Base Learner Features on {outcome_variable.upper()} (ANOVA F-test)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("F-statistic (Higher = More Impact)", fontsize=12)
        plt.ylabel("Base Learner Feature", fontsize=12)

        plt.tight_layout()

        if plot_dir is not None:
            plot_path = os.path.join(plot_dir, "base_learner_feature_importance.png")
            plt.savefig(plot_path)
            print(f"✅ Feature importance plot saved to: {plot_path}")
            plt.close()

        plt.show()
        plt.close()

        # 6. Print the results table
        print("\nBase Learner Feature Importance (sorted):")
        print(results_df.to_string())

    def plot_parameter_distributions(self, param_type="config", plot_dir=None):
        """
        Visualizes the distribution of values for a specified group of parameters.

        - For 'config' and 'run_details', it creates a grid of plots showing the
        distribution for each parameter/detail. It uses histograms for
        continuous data and count plots for categorical data.
        - For 'initial_features' and 'base_learner_features', it creates a
        single bar chart showing the selection frequency of each feature
        across all runs.

        Args:
            param_type (str, optional): The group of parameters to plot.
                Valid options: 'config', 'run_details', 'initial_features',
                'base_learner_features'. Defaults to 'config'.
        """
        plt.style.use("seaborn-v0_8-whitegrid")

        # --- Case 1 & 2: Plot distributions for columns in the DataFrame ---
        if param_type in ["config", "run_details"]:
            param_map = {
                "config": (
                    self.config_params,
                    "Distribution of Configuration Hyperparameters",
                ),
                "run_details": (
                    self.run_details,
                    "Distribution of Run Details & Metadata",
                ),
            }
            params_to_plot, title = param_map[param_type]
            params_to_plot = [p for p in params_to_plot if p in self.df.columns]

            if not params_to_plot:
                print(f"No columns found for type '{param_type}'.")
                return

            # Setup subplot grid
            n_cols = min(3, len(params_to_plot))
            n_rows = math.ceil(len(params_to_plot) / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
            axes = axes.flatten()

            print(f"📊 Generating distribution plots for {param_type}...")
            for i, param in enumerate(params_to_plot):
                ax = axes[i]
                data = self.df[param].dropna()
                if data.empty:
                    ax.text(0.5, 0.5, "No Data", ha="center", va="center")
                    ax.set_title(f"Distribution of {param}", fontsize=12)
                    continue

                # Decide plot type: countplot for categorical, histplot for continuous
                if data.nunique() < 20 and data.dtype in ["object", "int64", "bool"]:
                    sns.countplot(
                        x=data, ax=ax, palette="viridis", order=sorted(data.unique())
                    )
                    # --- FIX IS HERE ---
                    if data.nunique() > 4:  # Rotate labels if there are many categories
                        plt.setp(
                            ax.get_xticklabels(),
                            rotation=45,
                            ha="right",
                            rotation_mode="anchor",
                        )
                    # --- END OF FIX ---
                else:
                    sns.histplot(data, ax=ax, kde=True, color="darkcyan")

                ax.set_title(f"{param}", fontsize=14)
                ax.set_xlabel("")
                ax.set_ylabel("Count")

            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)  # Hide unused subplots

            fig.suptitle(title, fontsize=20, fontweight="bold")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # --- Case 3 & 4: Plot frequency of features from lists ---
        elif param_type in ["initial_features", "base_learner_features"]:
            if param_type == "initial_features":
                try:
                    features_flat = self.df["f_list"].explode().dropna().tolist()
                    title = "Selection Frequency of Initial Features"
                except Exception as e:
                    print(f"Could not process 'f_list': {e}")
                    return
            else:  # 'base_learner_features'
                bl_cols = [col for col in self.df.columns if col.startswith("BL_")]
                if not bl_cols:
                    print("No 'BL_' columns found.")
                    return

                all_bl_features = []
                for col in bl_cols:
                    all_bl_features.extend(self.df[col].explode().dropna())
                features_flat = all_bl_features
                title = "Selection Frequency of Base Learner Features"

            if not features_flat:
                print(f"No features found for type '{param_type}'.")
                return

            print(f"📊 Generating frequency plot for {param_type}...")
            freq_df = pd.Series(features_flat).value_counts().reset_index()
            freq_df.columns = ["Feature", "Frequency"]

            plt.figure(figsize=(12, max(8, len(freq_df) * 0.35)))
            sns.barplot(
                x="Frequency", y="Feature", data=freq_df, palette="magma", orient="h"
            )
            plt.title(title, fontsize=18, fontweight="bold", pad=20)
            plt.xlabel("Frequency (Total Count Across All Runs)", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.tight_layout()

            if plot_dir is not None:
                plot_path = os.path.join(plot_dir, f"{param_type}_frequency.png")
                plt.savefig(plot_path)
                print(f"✅ Frequency plot saved to: {plot_path}")
            plt.show()

        else:
            print(
                f"❌ Error: Invalid 'param_type'. Choose from: 'config', 'run_details', 'initial_features', or 'base_learner_features'."
            )

    def plot_ensemble_feature_diversity(self, outcome_variable="auc", plot_dir=None):
        """
        Calculates and plots the internal feature diversity of each ensemble
        against its performance.

        This method computes the average Jaccard similarity between the feature
        sets of all pairs of base learners within each run. A lower score
        indicates higher diversity. This diversity score is then plotted
        against a specified outcome variable to see if there is a correlation.

        Args:
            outcome_variable (str, optional): The performance metric. Defaults to 'auc'.
        """
        # Prefer using 'feature_names' column if available
        if "feature_names" in self.df.columns:
            print(
                "📊 Calculating internal ensemble diversity for each run using 'feature_names' column..."
            )

            def get_avg_jaccard(row):
                feature_lists = row["feature_names"]
                if not isinstance(feature_lists, list) or len(feature_lists) < 2:
                    return None
                feature_sets = [
                    set(flist) for flist in feature_lists if isinstance(flist, list)
                ]
                if len(feature_sets) < 2:
                    return None
                pairs = list(itertools.combinations(feature_sets, 2))
                similarities = []
                for set_a, set_b in pairs:
                    intersection_len = len(set_a.intersection(set_b))
                    union_len = len(set_a.union(set_b))
                    similarities.append(
                        intersection_len / union_len if union_len > 0 else 0
                    )
                return sum(similarities) / len(similarities) if similarities else 0

            temp_df = self.df.copy()
            temp_df["avg_jaccard_similarity"] = temp_df.apply(get_avg_jaccard, axis=1)
            temp_df.dropna(
                subset=["avg_jaccard_similarity", outcome_variable], inplace=True
            )
        else:
            bl_cols = [col for col in self.df.columns if col.startswith("BL_")]
            if len(bl_cols) < 2:
                print(
                    "❌ Error: Need at least 2 base learner columns (e.g., 'BL_0', 'BL_1') to compare."
                )
                return
            print(
                "📊 Calculating internal ensemble diversity for each run using BL_ columns..."
            )

            def decode_bl_features(bl_entry):
                if isinstance(bl_entry, list) and all(
                    isinstance(x, str) for x in bl_entry
                ):
                    return set(x for x in bl_entry if x in self.original_feature_names)
                if isinstance(bl_entry, list) and len(bl_entry) == len(
                    self.original_feature_names
                ):
                    try:
                        if all(isinstance(x, (int, np.integer)) for x in bl_entry):
                            return set(
                                f
                                for i, f in enumerate(self.original_feature_names)
                                if bl_entry[i]
                            )
                        if all(
                            isinstance(x, str) and x in {"0", "1"} for x in bl_entry
                        ):
                            return set(
                                f
                                for i, f in enumerate(self.original_feature_names)
                                if int(bl_entry[i])
                            )
                    except Exception:
                        return set()
                return set()

            def get_avg_jaccard(row):
                feature_sets = [
                    decode_bl_features(row[col])
                    for col in bl_cols
                    if isinstance(row[col], list) and row[col]
                ]
                if len(feature_sets) < 2:
                    return None
                pairs = list(itertools.combinations(feature_sets, 2))
                similarities = []
                for set_a, set_b in pairs:
                    intersection_len = len(set_a.intersection(set_b))
                    union_len = len(set_a.union(set_b))
                    similarities.append(
                        intersection_len / union_len if union_len > 0 else 0
                    )
                return sum(similarities) / len(similarities) if similarities else 0

            temp_df = self.df.copy()
            temp_df["avg_jaccard_similarity"] = temp_df.apply(get_avg_jaccard, axis=1)
            temp_df.dropna(
                subset=["avg_jaccard_similarity", outcome_variable], inplace=True
            )

        # --- Plotting ---
        plt.figure(figsize=(10, 7))
        sns.regplot(
            data=temp_df,
            x="avg_jaccard_similarity",
            y=outcome_variable,
            scatter_kws={"alpha": 0.6, "color": "darkcyan"},
            line_kws={"color": "red", "linestyle": "--"},
        )

        plt.title(
            f"Ensemble Feature Diversity vs. {outcome_variable.upper()}",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel(
            "Average Jaccard Similarity (Low score = High Diversity)", fontsize=12
        )
        plt.ylabel(f"Performance ({outcome_variable.upper()})", fontsize=12)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        if plot_dir is not None:
            plot_path = os.path.join(plot_dir, "ensemble_feature_diversity.png")
            plt.savefig(plot_path)
            print(f"✅ Diversity plot saved to: {plot_path}")

        plt.show()
        plt.close()

    def plot_performance_tradeoff(
        self,
        performance_metric: str = "auc",
        cost_metric: str = "run_time",
        hue_parameter: str = "pop_val",
        plot_dir: Optional[str] = None,
    ) -> None:
        """Plots the trade-off between performance and a cost metric.

        This helps identify configurations that offer the best performance for an
        acceptable cost.

        Args:
            performance_metric: The column for the performance metric (y-axis).
            cost_metric: The column for the cost metric (x-axis).
            hue_parameter: The hyperparameter column to use for color-coding.
            plot_dir: Directory to save the plot. Defaults to None.
        """
        # --- 1. Validate Inputs ---
        required_cols = [performance_metric, cost_metric, hue_parameter]
        for col in required_cols:
            if col not in self.df.columns:
                print(f"❌ Error: Column '{col}' not found in the DataFrame.")
                return

        print(
            f"📊 Plotting {performance_metric} vs. {cost_metric}, colored by {hue_parameter}..."
        )

        # --- 2. Prepare Data ---
        plot_df = self.df[required_cols].dropna()
        if plot_df.empty:
            print("No data available to plot after removing missing values.")
            return

        # Ensure the hue parameter is treated as a category for distinct colors
        plot_df[hue_parameter] = plot_df[hue_parameter].astype("category")

        # --- 3. Create Plot ---
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(12, 8))

        ax = sns.scatterplot(
            data=plot_df,
            x=cost_metric,
            y=performance_metric,
            hue=hue_parameter,
            palette="plasma",
            alpha=0.8,
            s=100,  # Size of points
        )

        # --- 4. Customize Labels and Title ---
        plt.title(
            f"Trade-off: {performance_metric.replace('_', ' ').title()} vs. {cost_metric.replace('_', ' ').title()}",
            fontsize=18,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel(f"Cost: {cost_metric.replace('_', ' ').title()}", fontsize=14)
        plt.ylabel(
            f"Performance: {performance_metric.replace('_', ' ').title()}", fontsize=14
        )

        # Place legend outside the plot
        ax.legend(
            title=hue_parameter.replace("_", " ").title(),
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        plt.tight_layout(rect=[0, 0, 0.88, 1])  # Adjust layout to make space for legend

        if plot_dir is not None:
            plot_path = os.path.join(plot_dir, "performance_tradeoff.png")
            plt.savefig(plot_path)
            print(f"✅ Trade-off plot saved to: {plot_path}")

        plt.show()
        plt.close()

    def plot_all_convergence(
        self,
        history_column: str = "generation_progress_list",
        performance_metric: str = "auc",
        highlight_best: bool = True,
        plot_dir: Optional[str] = None,
    ) -> None:
        """Plots the convergence curves for all GA runs on a single graph.

        Each line represents one run. All runs are plotted with transparency
        to show the density of solutions, and the single best run (based on the
        final performance_metric) can be highlighted for emphasis.

        Args:
            history_column: The column containing the list of fitness scores.
            performance_metric: The column used to identify the best run.
            highlight_best: If True, highlights the best-performing run.
            plot_dir: Directory to save the plot. Defaults to None.
        """
        # --- 1. Validate Inputs & Prepare Data ---
        if history_column not in self.df.columns:
            print(f"❌ Error: History column '{history_column}' not found.")
            return
        if highlight_best and performance_metric not in self.df.columns:
            print(
                f"❌ Error: Performance metric column '{performance_metric}' not found."
            )
            return

        plot_df = self.df.dropna(subset=[history_column]).reset_index()

        if plot_df.empty:
            print(f"No data to plot in '{history_column}'.")
            return

        # --- Helper function to safely parse string-lists ---
        def parse_history(data):
            if isinstance(data, list):
                return data
            if isinstance(data, str):
                try:
                    # Safely evaluate the string representation of the list
                    return ast.literal_eval(data)
                except (ValueError, SyntaxError):
                    return None  # Return None if parsing fails
            return None

        # --- 2. Create Plot ---
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 8))
        print(f"📊 Plotting {len(plot_df)} convergence curves...")
        min_fitness = float("inf")

        # Plot all runs with transparency
        for index, row in plot_df.iterrows():
            # **This is the key fix: parse the string into a list**
            history = parse_history(row[history_column])
            if history:  # Checks if parsing was successful and list is not empty
                ax.plot(history, color="cornflowerblue", alpha=0.4, linewidth=1.5)
                current_min = min(history)
                if current_min < min_fitness:
                    min_fitness = current_min

        # Highlight the best run
        if highlight_best:
            try:
                best_run_idx = self.df[performance_metric].idxmax()
                best_run_history_raw = self.df.loc[best_run_idx, history_column]
                # **Also apply the fix here**
                best_run_history = parse_history(best_run_history_raw)

                if best_run_history:
                    label_text = (
                        f"Best Run (Final {performance_metric.upper()}: "
                        f"{self.df[performance_metric].max():.4f})"
                    )
                    ax.plot(
                        best_run_history,
                        color="#FF4136",
                        linewidth=2.5,
                        label=label_text,
                    )
                    ax.legend(fontsize=12)
            except (ValueError, KeyError) as e:
                print(f"⚠️ Could not highlight best run: {e}")

        # --- 3. Customize Labels and Title ---
        ax.set_title("GA Convergence for All Runs", fontsize=18, fontweight="bold")
        ax.set_xlabel("Generation", fontsize=14)
        ax.set_ylabel("Best Fitness Score", fontsize=14)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5)

        if min_fitness != float("inf"):
            ax.set_ylim(bottom=max(0, min_fitness * 0.95))

        plt.tight_layout()

        if plot_dir is not None:
            plot_path = os.path.join(plot_dir, "convergence_curves.png")
            plt.savefig(plot_path)
            print(f"✅ Convergence plot saved to: {plot_path}")

        plt.show()
        plt.close()

    def plot_interaction_heatmap(
        self,
        param1: str,
        param2: str,
        performance_metric: str = "auc",
        plot_dir: Optional[str] = None,
    ) -> None:
        """Creates a heatmap to visualize hyperparameter interactions.

        Args:
            param1: The name of the first hyperparameter (y-axis).
            param2: The name of the second hyperparameter (x-axis).
            performance_metric: The metric to display in the heatmap cells.
            plot_dir: Directory to save the plot. Defaults to None.
        """
        # --- 1. Validate Inputs ---
        required_cols = [param1, param2, performance_metric]
        for col in required_cols:
            if col not in self.df.columns:
                print(f"❌ Error: Column '{col}' not found in the DataFrame.")
                return

        # Warn if parameters have too many unique values, which can make the heatmap unreadable
        if self.df[param1].nunique() > 15 or self.df[param2].nunique() > 15:
            print(
                f"⚠️ Warning: '{param1}' or '{param2}' has many unique values. The heatmap may be large."
            )

        print(f"📊 Generating interaction heatmap for '{param1}' and '{param2}'...")

        # --- 2. Create Pivot Table ---
        # This groups the data by the two parameters and calculates the mean performance for each combination.
        try:
            pivot_df = pd.pivot_table(
                self.df,
                values=performance_metric,
                index=param1,
                columns=param2,
                aggfunc="mean",
            )
        except Exception as e:
            print(f"❌ Error creating pivot table: {e}")
            return

        # --- 3. Create Heatmap ---
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            pivot_df,
            annot=True,  # Display the mean value in each cell
            fmt=".4f",  # Format numbers to 4 decimal places (standard for AUC)
            cmap="viridis",  # A visually appealing color map
            linewidths=0.5,
        )

        # --- 4. Customize Labels and Title ---
        plt.title(
            f"Interaction: {param1.replace('_', ' ').title()} vs {param2.replace('_', ' ').title()} on {performance_metric.upper()}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel(param2.replace("_", " ").title(), fontsize=12)
        plt.ylabel(param1.replace("_", " ").title(), fontsize=12)
        plt.tight_layout()

        if plot_dir is not None:
            plot_path = os.path.join(plot_dir, "interaction_heatmap.png")
            plt.savefig(plot_path)
            print(f"✅ Interaction heatmap saved to: {plot_path}")

        plt.show()
        plt.close()

    def plot_feature_stability(
        self,
        performance_metric="auc",
        top_percent=10.0,
        feature_type="base_learner",
        plot_dir=None,
    ):
        """
        Analyzes and plots the selection frequency of features within the top-performing runs.

        This helps identify which features are consistently part of the best solutions,
        indicating their robust importance.

        Args:
            performance_metric (str): The column used to rank runs. Defaults to 'auc'.
            top_percent (float): The percentage of top runs to analyze (e.g., 10 for top 10%).
                                Defaults to 10.0.
            feature_type (str): The feature set to analyze. Either 'initial' (from f_list)
                                or 'base_learner' (from BL_ columns). Defaults to 'base_learner'.
        """
        # --- 1. Validate Inputs ---
        if performance_metric not in self.df.columns:
            print(f"❌ Error: Performance metric '{performance_metric}' not found.")
            return
        if not 0 < top_percent <= 100:
            print(f"❌ Error: top_percent must be between 0 and 100.")
            return
        if feature_type not in ["initial", "base_learner"]:
            print(f"❌ Error: feature_type must be 'initial' or 'base_learner'.")
            return

        # --- 2. Filter for Top Runs ---
        # Calculate the performance threshold for the top N percent
        threshold = self.df[performance_metric].quantile(1 - (top_percent / 100.0))
        top_runs_df = self.df[self.df[performance_metric] >= threshold]

        if top_runs_df.empty:
            print(f"⚠️ No runs found in the top {top_percent}%. Cannot generate plot.")
            return

        print(
            f"📊 Analyzing features from {len(top_runs_df)} runs in the top {top_percent}% by {performance_metric.upper()}..."
        )

        # --- 3. Calculate Feature Frequency on Filtered Data ---
        if feature_type == "initial":
            source_title = "Initial Feature"
            feature_source_col = "f_list"
            if feature_source_col not in top_runs_df.columns:
                print(f"❌ Error: Feature column '{feature_source_col}' not found.")
                return
            features_flat = top_runs_df[feature_source_col].explode().dropna().tolist()
        else:  # 'base_learner'
            source_title = "Base Learner Feature"
            feature_names_col = "feature_names"
            if feature_names_col not in top_runs_df.columns:
                print(
                    f"❌ Error: Feature column '{feature_names_col}' not found. Was the explorer initialized correctly?"
                )
                return
            # feature_names is a list of lists of feature names per ensemble
            features_flat = []
            for feature_lists in top_runs_df[feature_names_col]:
                if isinstance(feature_lists, list):
                    for feature_list in feature_lists:
                        if isinstance(feature_list, list):
                            features_flat.extend(feature_list)

        if not features_flat:
            print("No features found in the selected top runs.")
            return

        freq_df = pd.Series(features_flat).value_counts().reset_index()
        freq_df.columns = ["Feature", "Frequency"]

        # --- 4. Plotting ---
        plt.style.use("seaborn-v0_8-whitegrid")
        # Dynamically adjust figure height based on number of features
        plt.figure(figsize=(12, max(8, len(freq_df) * 0.35)))

        sns.barplot(
            x="Frequency", y="Feature", data=freq_df, palette="cividis", orient="h"
        )

        # --- 5. Labels and Title ---
        plt.title(
            f"{source_title} Stability (Top {top_percent}% of Runs)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel(f"Frequency in Top Performing Runs", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()

        if plot_dir is not None:
            plot_path = os.path.join(plot_dir, "feature_stability.png")
            plt.savefig(plot_path)
            print(f"✅ Stability plot saved to: {plot_path}")

        plt.show()
        plt.close()

    def plot_performance_vs_size(
        self,
        performance_metric: str = "auc",
        feature_type: str = "base_learner",
        plot_dir: Optional[str] = None,
    ) -> None:
        """Plots performance vs. the number of features used.

        A 2nd order polynomial trend line is fitted to highlight the point of
        diminishing returns.

        Args:
            performance_metric: The column name for the performance metric (y-axis).
            feature_type: The feature set to use for calculating size.
                                Either 'initial' (from f_list) or 'base_learner'
                                (union of features in all BL_ columns).
                                Defaults to 'base_learner'.
            plot_dir: Directory to save the plot. Defaults to None.
        """
        # 1. Validate Inputs
        if performance_metric not in self.df.columns:
            print(f"❌ Error: Performance metric '{performance_metric}' not found.")
            return
        if feature_type not in ["initial", "base_learner"]:
            print(f"❌ Error: feature_type must be 'initial' or 'base_learner'.")
            return

        print(
            f"📊 Plotting {performance_metric} vs. number of {feature_type} features..."
        )

        # 2. Calculate Solution Size for each run
        temp_df = self.df[[performance_metric]].copy()

        if feature_type == "initial":
            source_col, title_frag = "f_list", "Initial"
            if source_col not in self.df.columns:
                print(f"❌ Error: Feature column '{source_col}' not found.")
                return
            temp_df["solution_size"] = self.df[source_col].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        else:  # 'base_learner'
            source_col, title_frag = "feature_names", "Base Learner"
            if source_col not in self.df.columns:
                print(
                    f"❌ Error: Feature column '{source_col}' not found. Was the explorer initialized correctly?"
                )
                return

            def count_unique_bl_features(row):
                # row[source_col] is a list of lists of feature names, e.g., [['f1', 'f2'], ['f2', 'f3']]
                feature_lists = row[source_col]
                if not isinstance(feature_lists, list):
                    return 0
                # Flatten the list of lists of features for the ensemble and count unique features
                all_features = set(itertools.chain.from_iterable(feature_lists))
                return len(all_features)

            temp_df["solution_size"] = self.df.apply(count_unique_bl_features, axis=1)

        temp_df.dropna(inplace=True)

        # 3. Create Plot
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(12, 8))

        try:
            # Use regplot to show the scatter and a fitted trend line
            sns.regplot(
                data=temp_df,
                x="solution_size",
                y=performance_metric,
                order=2,  # Fit a 2nd order polynomial to show diminishing returns
                line_kws={"color": "#E11D74", "linestyle": "--"},
                scatter_kws={"alpha": 0.5, "color": "darkslateblue"},
            )
        except np.linalg.LinAlgError:
            print(
                "⚠️ Warning: Could not fit a 2nd-order polynomial due to a numerical issue (e.g., insufficient unique data points)."
                "\nDisplaying a scatter plot without the trend line."
            )
            sns.scatterplot(
                data=temp_df,
                x="solution_size",
                y=performance_metric,
                alpha=0.5,
                color="darkslateblue",
            )

        # 4. Customize Labels and Title
        plt.title(
            f"Performance vs. Solution Size ({title_frag} Features)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Number of Unique Features in Solution", fontsize=12)
        plt.ylabel(f"Performance ({performance_metric.upper()})", fontsize=12)
        plt.tight_layout()

        if plot_dir is not None:
            plot_path = os.path.join(plot_dir, "performance_vs_size.png")
            plt.savefig(plot_path)
            print(f"✅ Performance vs. Size plot saved to: {plot_path}")

        plt.show()
        plt.close()

    def plot_feature_cooccurrence(
        self,
        performance_metric: str = "auc",
        top_percent: float = 10.0,
        top_n_features: int = 15,
        feature_type: str = "base_learner",
        plot_dir: Optional[str] = None,
    ) -> None:
        """Creates a heatmap of feature co-occurrence in top runs.

        Args:
            performance_metric: The column used to rank runs. Defaults to 'auc'.
            top_percent: The percentage of top runs to analyze. Defaults to 10.0.
            top_n_features: The number of top stable features to include.
            feature_type: The feature set to analyze ('initial' or 'base_learner').
            plot_dir: Directory to save the plot. Defaults to None.
        """
        # 1. Validate Inputs
        if performance_metric not in self.df.columns:
            print(f"❌ Error: Performance metric '{performance_metric}' not found.")
            return
        if not 0 < top_percent <= 100:
            print(f"❌ Error: top_percent must be between 0 and 100.")
            return
        if not isinstance(top_n_features, int) or top_n_features <= 0:
            print(f"❌ Error: top_n_features must be a positive integer.")
            return
        if feature_type not in ["initial", "base_learner"]:
            print(f"❌ Error: feature_type must be 'initial' or 'base_learner'.")
            return

        # 2. Filter for Top Runs and Identify Top N Features
        threshold = self.df[performance_metric].quantile(1 - (top_percent / 100.0))
        top_runs_df = self.df[self.df[performance_metric] >= threshold]

        if top_runs_df.empty:
            print(f"⚠️ No runs found in the top {top_percent}%. Cannot generate plot.")
            return

        # Calculate co-occurrence matrix
        cooccurrence_matrix = self.calculate_cooccurrence_matrix(
            top_runs_df, top_n_features, feature_type
        )

        # 3. Create Plot
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(12, 10))

        if cooccurrence_matrix.empty:
            print("⚠️ Co-occurrence matrix is empty. Cannot generate plot.")
            return

        sns.heatmap(
            cooccurrence_matrix,
            annot=True,
            cmap="Blues",
            square=True,
            linewidths=0.5,
            linecolor="white",
        )

        plt.title("Co-occurrence of Top Features in Best Ensembles")
        plt.xlabel("Feature")
        plt.ylabel("Feature")

        if plot_dir is not None:
            plt.savefig(
                os.path.join(plot_dir, f"feature_cooccurrence_{feature_type}.png"),
                bbox_inches="tight",
                dpi=300,
            )
        plt.show()
        plt.close()

    def calculate_cooccurrence_matrix(self, top_runs_df, top_n_features, feature_type):
        """
        Calculate the co-occurrence matrix of top features in the best-performing runs.

        Parameters
        ----------
        top_runs_df : pandas DataFrame
            A DataFrame containing the top-performing runs.
        top_n_features : int
            The number of top features to include in the co-occurrence matrix.
        feature_type : str
            The type of features to analyze ('initial' or 'base_learner').

        Returns
        -------
        cooccurrence_matrix : pandas DataFrame
            A DataFrame representing the co-occurrence matrix of top features.
        """
        # Extract feature names from the top runs
        feature_names = []
        for index, row in top_runs_df.iterrows():
            feature_arrays = row["feature_arrays"]
            for feature_array in feature_arrays:
                feature_names.extend(
                    [
                        f
                        for i, f in enumerate(self.original_feature_names)
                        if feature_array[i] == 1
                    ]
                )

        # Get the top N features
        top_features = (
            pd.Series(feature_names).value_counts().head(top_n_features).index.tolist()
        )

        # Create a co-occurrence matrix
        cooccurrence_matrix = pd.DataFrame(0, index=top_features, columns=top_features)

        # Iterate over the top runs and update the co-occurrence matrix
        for index, row in top_runs_df.iterrows():
            feature_arrays = row["feature_arrays"]
            for feature_array in feature_arrays:
                features = [
                    f
                    for i, f in enumerate(self.original_feature_names)
                    if feature_array[i] == 1
                ]
                for feature1 in features:
                    for feature2 in features:
                        if (
                            feature1 != feature2
                            and feature1 in top_features
                            and feature2 in top_features
                        ):
                            cooccurrence_matrix.loc[feature1, feature2] += 1

        return cooccurrence_matrix

    def plot_algorithm_distribution_in_ensembles(
        self, plot_dir: Optional[str] = None
    ) -> None:
        """Plots the distribution of algorithms in the final ensembles.

        This method visualizes how frequently each type of base learner
        (e.g., LogisticRegression, RandomForestClassifier) appears in the
        best-performing ensembles across all runs.
        """
        if "best_ensemble" not in self.df.columns:
            print(self.df["best_ensemble"].iloc[0])

            print("❌ Error: 'best_ensemble' column not found in the DataFrame.")
            return

        print("📊 Analyzing algorithm distribution in best ensembles...")

        all_algorithms = []
        # Iterate over each run's best ensemble string
        for ensemble_str in self.df["best_ensemble"].dropna():
            try:
                # Use regex to find all algorithm names like "LogisticRegression("
                # The pattern looks for a word starting with a capital letter followed by an opening parenthesis.
                # This is a robust way to find algorithm names without full parsing.
                algorithms_in_run = re.findall(r"([A-Z]\w+)\(", ensemble_str)
                all_algorithms.extend(algorithms_in_run)
            except Exception as e:
                # This is a fallback, but the regex should be quite safe.
                print(f"Could not parse algorithms from an ensemble string: {e}")
                continue

        if not all_algorithms:
            print("No algorithms could be extracted from the 'best_ensemble' column.")
            return

        # Calculate the frequency of each algorithm
        freq_df = pd.Series(all_algorithms).value_counts().reset_index()
        freq_df.columns = ["Algorithm", "Frequency"]

        # --- Plotting ---
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(12, 8))

        barplot = sns.barplot(
            x="Frequency", y="Algorithm", data=freq_df, palette="cubehelix", orient="h"
        )

        # --- Customize Labels and Title ---
        plt.title(
            "Frequency of Algorithms in Final Ensembles",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Total Count in Ensembles", fontsize=12)
        plt.ylabel("Algorithm", fontsize=12)

        # Add data labels to the bars
        for patch in barplot.patches:
            plt.text(
                patch.get_width() * 1.01,
                patch.get_y() + patch.get_height() / 2,
                f"{int(patch.get_width())}",
                va="center",
                fontsize=10,
                color="dimgray",
            )

        plt.tight_layout()

        if plot_dir is not None:
            plt.savefig(
                os.path.join(plot_dir, "algorithm_distribution_in_ensembles.png"),
                bbox_inches="tight",
                dpi=300,
            )
        plt.show()
        plt.close()

    def run_all_plots(
        self,
        plot_dir: Optional[str] = None,
        outcome_variable: str = "auc",
        interaction_params: Tuple[str, str] = ("pop_val", "g_val"),
        tradeoff_params: Tuple[str, str] = ("run_time", "pop_val"),
    ) -> None:
        """Runs all available plotting methods for a comprehensive analysis.

        This is a convenience method to generate a full suite of analysis plots from the
        GA results.

        Args:
            plot_dir: Directory to save the plots. Defaults to None.
            outcome_variable: The main performance metric. Defaults to "auc".
            interaction_params: The two parameters for the interaction heatmap.
            tradeoff_params: The cost and hue parameters for the tradeoff plot.
        """
        print("--- Running All Plots ---")

        # Parameter and Run Detail Importance
        self.plot_config_anova_feature_importances(
            outcome_variable=outcome_variable, plot_dir=plot_dir
        )
        self.plot_run_details_anova_feature_importances(
            outcome_variable=outcome_variable, plot_dir=plot_dir
        )
        self.plot_combined_anova_feature_importances(
            outcome_variable=outcome_variable, plot_dir=plot_dir
        )

        # Distributions
        self.plot_parameter_distributions(param_type="config", plot_dir=plot_dir)
        self.plot_parameter_distributions(param_type="run_details", plot_dir=plot_dir)
        self.plot_algorithm_distribution_in_ensembles(plot_dir=plot_dir)

        # Feature Analysis
        self.plot_initial_feature_importance(
            outcome_variable=outcome_variable, plot_dir=plot_dir
        )
        self.plot_base_learner_feature_importance(
            outcome_variable=outcome_variable, plot_dir=plot_dir
        )
        self.plot_feature_stability(
            performance_metric=outcome_variable, plot_dir=plot_dir
        )
        self.plot_feature_cooccurrence(
            performance_metric=outcome_variable, plot_dir=plot_dir
        )

        # Performance Analysis
        self.plot_all_convergence(
            performance_metric=outcome_variable, plot_dir=plot_dir
        )
        self.plot_ensemble_feature_diversity(
            outcome_variable=outcome_variable, plot_dir=plot_dir
        )
        self.plot_performance_vs_size(
            performance_metric=outcome_variable, plot_dir=plot_dir
        )

        # Specific Interaction/Tradeoff Plots
        if len(interaction_params) == 2 and all(
            isinstance(p, str) for p in interaction_params
        ):
            self.plot_interaction_heatmap(
                param1=interaction_params[0],
                param2=interaction_params[1],
                performance_metric=outcome_variable,
                plot_dir=plot_dir,
            )

        if len(tradeoff_params) == 2 and all(
            isinstance(p, str) for p in tradeoff_params
        ):
            self.plot_performance_tradeoff(
                performance_metric=outcome_variable,
                cost_metric=tradeoff_params[0],
                hue_parameter=tradeoff_params[1],
                plot_dir=plot_dir,
            )

        print("--- All Plots Finished ---")


def extract_feature_arrays_from_string(raw_ensemble_string: str) -> List[List[int]]:
    """Extracts binary feature arrays from a complex ensemble string.

    This function is designed to parse strings that may contain non-standard Python
    literals like `array(...)`, which `ast.literal_eval` cannot handle directly.
    It uses a regular expression to robustly find all list-like substrings of digits
    and then identifies and parses only the ones corresponding to feature arrays.

    Args:
        raw_ensemble_string: A string containing the nested ensemble data.

    Returns:
        A list of lists, where each inner list is a binary feature array.
        Returns an empty list if parsing fails.
    """
    try:
        # Use a regular expression to find all substrings that look like a list of numbers.
        # This pattern finds a '[' followed by digits, commas, and whitespace, ending with a ']'.
        # It correctly captures both the binary feature vectors and the numpy array predictions.
        all_list_substrings = re.findall(r"(\[[\d,\s]+\])", raw_ensemble_string)

        # The feature arrays are the first, third, fifth, etc., lists found.
        # We select them by taking every other element starting from the first (index 0).
        feature_array_strings = all_list_substrings[0::2]

        # Safely parse each of these clean substrings into a Python list.
        # This works because each string in feature_array_strings is a valid list literal.
        feature_arrays = [ast.literal_eval(s) for s in feature_array_strings]

        return feature_arrays

    except Exception as e:
        print(f"❌ An error occurred during extraction: {e}")
        return []


# --- Example Usage ---

## The problematic string from the DataFrame.
# raw_ensemble_string = """[[(0.5833333333333334, "LogisticRegression(C=1, class_weight='balanced', max_iter=5, solver='sag')", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0.9537, array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0])), (0.6871842709362768, 'Perceptron(eta0=0.1, max_iter=7)', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0.9722, array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))]]"""

## Call the function with the raw string
# extracted_arrays = extract_feature_arrays_from_string(raw_ensemble_string)
