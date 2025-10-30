import logging
import random
import string
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

logger = logging.getLogger("ensemble_ga")


class SyntheticDataGenerator:
    """
    Generates a synthetic dataset with controlled imperfections to test data cleaning pipelines.

    This class uses sklearn's `make_classification` to create a base dataset with a
    known signal, then introduces common data quality issues such as:
    - Missing values (NaNs)
    - Highly correlated features
    - Constant value columns
    - Duplicate columns
    - Categorical (string) columns

    The feature names are generated with prefixes to mimic real-world data structures.
    """

    n_samples: int
    """The number of samples (rows) to generate."""
    n_features: int
    """The number of base features to generate before adding imperfect ones."""
    n_informative: int
    """The number of informative features."""
    n_redundant: int
    """The number of redundant features."""
    class_sep: float
    """The separation between classes. Higher values make the task easier."""
    missing_pct: float
    """The percentage of cells to replace with NaN."""
    n_constant_cols: int
    """The number of constant-value columns to add."""
    n_duplicate_cols: int
    """The number of columns to duplicate."""
    n_correlated_pairs: int
    """The number of highly correlated feature pairs to add."""
    n_categorical_cols: int
    """The number of non-numeric (string) columns to add."""
    outcome_name: str
    """The name for the target variable column."""
    random_state: int
    """The random seed for reproducibility."""
    feature_name_templates: Dict
    """A dictionary holding templates for generating realistic feature names."""

    def __init__(
        self,
        n_samples: int = 2000,
        n_features: int = 100,
        n_informative_ratio: float = 0.5,
        n_redundant_ratio: float = 0.1,
        class_sep: float = 1.5,
        missing_pct: float = 0.05,
        n_constant_cols: int = 3,
        n_duplicate_cols: int = 3,
        n_correlated_pairs: int = 3,
        n_categorical_cols: int = 2,
        outcome_name: str = "outcome_var_1",
        random_state: int = 42,
    ):
        """Initializes the SyntheticDataGenerator with specified parameters."""
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = int(n_features * n_informative_ratio)
        self.n_redundant = int(self.n_informative * n_redundant_ratio)
        self.class_sep = class_sep
        self.missing_pct = missing_pct
        self.n_constant_cols = n_constant_cols
        self.n_duplicate_cols = n_duplicate_cols
        self.n_correlated_pairs = n_correlated_pairs
        self.n_categorical_cols = n_categorical_cols
        self.outcome_name = outcome_name
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)

        self._setup_feature_templates()

    def _setup_feature_templates(self) -> None:
        """Initializes templates for generating realistic feature names based on column_names.py."""
        self.feature_name_templates = {
            "age": {"prefix": ["age"], "suffix": [""]},
            "sex": {"prefix": ["male"], "suffix": [""]},
            "bmi": {"prefix": ["bmi"], "suffix": ["_value"]},
            "ethnicity": {"prefix": ["census"], "suffix": ["_code"]},
            "bloods": {
                "prefix": [
                    "hemoglobin",
                    "platelet",
                    "wbc",
                    "rbc",
                    "sodium",
                    "potassium",
                    "creatinine",
                ],
                "suffix": [
                    "_mean",
                    "_median",
                    "_std",
                    "_num-tests",
                    "_max",
                    "_min",
                    "_most-recent",
                    "_days-since-last-test",
                    "_contains-extreme-high",
                ],
            },
            "diagnostic_order": {
                "prefix": ["xray", "ct_scan", "mri", "ultrasound"],
                "suffix": [
                    "_num-diagnostic-order",
                    "_days-since-last-diagnostic-order",
                    "_days-between-first-last-diagnostic",
                ],
            },
            "drug_order": {
                "prefix": [
                    "paracetamol",
                    "ibuprofen",
                    "aspirin",
                    "morphine",
                    "antibiotic",
                ],
                "suffix": [
                    "_num-drug-order",
                    "_days-since-last-drug-order",
                    "_days-between-first-last-drug",
                ],
            },
            "annotation_n": {
                "prefix": ["pain_level", "symptom_severity", "procedure_note"],
                "suffix": ["_count"],
            },
            "meta_sp_annotation_n": {
                "prefix": ["family_history_of_cancer", "personal_history_of_stroke"],
                "suffix": [
                    "_count_subject_present",
                    "_count_subject_not_present",
                    "_count_relative_present",
                    "_count_relative_not_present",
                ],
            },
            "annotation_mrc_n": {
                "prefix": ["nlp_concept_a", "nlp_concept_b"],
                "suffix": ["_count_mrc_cs"],
            },
            "meta_sp_annotation_mrc_n": {
                "prefix": ["nlp_meta_concept_x", "nlp_meta_concept_y"],
                "suffix": [
                    "_count_subject_present_mrc_cs",
                    "_count_subject_not_present_mrc_cs",
                ],
            },
            "core_02": {"prefix": ["core_02"], "suffix": ["_status"]},
            "bed": {"prefix": ["bed"], "suffix": ["_location"]},
            "vte_status": {"prefix": ["vte_status"], "suffix": ["_active"]},
            "hosp_site": {"prefix": ["hosp_site"], "suffix": ["_code"]},
            "core_resus": {"prefix": ["core_resus"], "suffix": ["_event"]},
            "news": {"prefix": ["news_resus"], "suffix": ["_score"]},
            "date_time_stamp": {"prefix": ["date_time_stamp"], "suffix": ["_value"]},
            "appointments": {
                "prefix": ["ConsultantCode", "ClinicCode", "AppointmentType"],
                "suffix": ["_id"],
            },
        }

    def _generate_feature_names(self, n: int) -> List[str]:
        """Generates n feature names distributed among all defined categories.

        Args:
            n: The total number of feature names to generate.

        Returns:
            A list of unique, realistically named features.
        """
        names = set()
        if n >= 2:
            names.add("age")
            names.add("male")
        available_categories = list(self.feature_name_templates.keys())
        while len(names) < n:
            category = random.choice(available_categories)
            if category in ["age", "sex"]:
                continue
            template = self.feature_name_templates[category]
            prefix = random.choice(template["prefix"])
            suffix = random.choice(template["suffix"])
            name = f"{prefix}{suffix}"
            if name in names:
                i = 1
                while f"{name}_{i}" in names:
                    i += 1
                name = f"{name}_{i}"
            names.add(name)
        return list(names)[:n]

    def _introduce_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Randomly introduces NaN values into the DataFrame.

        Args:
            df: The input DataFrame.

        Returns:
            The DataFrame with NaN values introduced.
        """
        df_copy = df.copy()
        mask = np.random.choice(
            [True, False],
            size=df_copy.shape,
            p=[self.missing_pct, 1 - self.missing_pct],
        )
        df_copy[mask] = np.nan
        return df_copy

    def _add_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds columns with a single constant value.

        Args:
            df: The input DataFrame.

        Returns:
            The DataFrame with constant columns added.
        """
        for i in range(self.n_constant_cols):
            df[f"constant_col_{i}"] = np.random.rand() * 10
        return df

    def _add_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds columns that are exact copies of other columns.

        Args:
            df: The input DataFrame.

        Returns:
            The DataFrame with duplicate columns added.
        """
        if not df.columns.to_list():
            logger.warning(
                "Cannot add duplicate columns to a DataFrame with no columns."
            )
            return df

        if self.n_duplicate_cols > len(df.columns):
            logger.warning(
                "More duplicate columns requested (%s) than available features (%s). Capping at available.",
                self.n_duplicate_cols,
                len(df.columns),
            )
            self.n_duplicate_cols = len(df.columns)

        if self.n_duplicate_cols == 0:
            return df

        cols_to_dupe = random.sample(list(df.columns), self.n_duplicate_cols)
        for i, col in enumerate(cols_to_dupe):
            df[f"duplicate_of_{col}_{i}"] = df[col]
        return df

    def _add_correlated_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds columns that are highly correlated with existing columns.

        Args:
            df: The input DataFrame.

        Returns:
            The DataFrame with correlated columns added.
        """
        if not df.columns.to_list():
            logger.warning(
                "Cannot add correlated columns to a DataFrame with no columns."
            )
            return df

        if self.n_correlated_pairs > len(df.columns):
            logger.warning(
                "More correlated columns requested (%s) than available features (%s). Capping at available.",
                self.n_correlated_pairs,
                len(df.columns),
            )
            self.n_correlated_pairs = len(df.columns)

        if self.n_correlated_pairs == 0:
            return df

        cols_to_corrupt = random.sample(list(df.columns), self.n_correlated_pairs)
        for i, col in enumerate(cols_to_corrupt):
            noise = np.random.normal(0, 0.1, size=self.n_samples)
            df[f"correlated_to_{col}_{i}"] = df[col] * 0.9 + noise
        return df

    def _add_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds non-numeric columns.

        Args:
            df: The input DataFrame.

        Returns:
            The DataFrame with categorical columns added.
        """
        for i in range(self.n_categorical_cols):
            # Create a high-cardinality string column
            choices = [
                "".join(random.choices(string.ascii_uppercase, k=5)) for _ in range(20)
            ]
            df[f"categorical_col_{i}"] = random.choices(choices, k=self.n_samples)
        return df

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values in numeric columns using the mean.

        Args:
            df: The input DataFrame, potentially with NaNs.

        Returns:
            The DataFrame with NaNs in numeric columns imputed.
        """
        df_copy = df.copy()
        # Select only numeric columns for imputation
        numeric_cols = df_copy.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            mean_val = df_copy[col].mean()
            df_copy[col].fillna(mean_val, inplace=True)

        logger.info("    > Imputed NaNs in %s numeric columns.", len(numeric_cols))
        return df_copy

    def generate_data(self) -> pd.DataFrame:
        """
        Orchestrates the data generation process and returns the final DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the synthetic data with all
                          the specified imperfections.
        """
        logger.info("--- Generating Synthetic Data with the following settings: ---")
        logger.info("  - n_samples: %s", self.n_samples)
        logger.info("  - n_features: %s", self.n_features)
        logger.info("  - n_informative: %s", self.n_informative)
        logger.info("  - n_redundant: %s", self.n_redundant)
        logger.info("  - class_sep: %s", self.class_sep)
        logger.info("  - missing_pct: %s", self.missing_pct)
        logger.info("  - n_constant_cols: %s", self.n_constant_cols)
        logger.info("  - n_duplicate_cols: %s", self.n_duplicate_cols)
        logger.info("  - n_correlated_pairs: %s", self.n_correlated_pairs)
        logger.info("  - n_categorical_cols: %s", self.n_categorical_cols)
        logger.info("  - random_state: %s", self.random_state)
        logger.info("-------------------------------------------------------------")

        # 1. Generate base data with a clear signal
        logger.info("Step 1: Generating base data with %s features...", self.n_features)
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=2,
            weights=[0.9, 0.1],  # Imbalanced classes
            flip_y=0.01,
            class_sep=self.class_sep,
            random_state=self.random_state,
        )

        # 2. Create DataFrame with proper feature names
        feature_names = self._generate_feature_names(self.n_features)
        df = pd.DataFrame(X, columns=feature_names)

        # 3. Introduce imperfections
        logger.info("Step 2: Introducing data quality issues...")
        df = self._add_constant_columns(df)
        df = self._add_duplicate_columns(df)
        df = self._add_correlated_columns(df)
        df = self._add_categorical_columns(df)

        # 4. Introduce missing values across the entire dataset
        logger.info("Step 3: Introducing missing values...")
        df = self._introduce_missing_values(df)

        # 5. Mean impute numeric columns, leaving categorical NaNs for the pipeline to handle
        logger.info("Step 4: Mean-imputing numeric columns...")
        df = self._impute_missing_values(df)

        # 6. Add outcome variable
        df[self.outcome_name] = y

        # 7. Shuffle columns to make it less obvious which are "good" or "bad"
        logger.info("Step 5: Finalizing and shuffling columns...")
        final_cols = list(df.columns)
        final_cols.remove(self.outcome_name)
        random.shuffle(final_cols)
        df = df[[self.outcome_name] + final_cols]

        logger.info(
            "--- Synthetic data generated successfully. Shape: %s ---", df.shape
        )
        return df


# # Example of how to use the class
# if __name__ == "__main__":
#     # --- Example of how to use the class ---
#     # You can customize the dataset by changing the parameters below.
#     generator = SyntheticDataGenerator(
#         n_samples=10000,              # Total number of rows in the dataset.
#         n_features=500,              # The number of base features to generate before adding imperfect ones.
#         n_informative_ratio=0.2,     # The ratio of features that are predictive of the outcome. Higher is easier.
#         n_redundant_ratio=0.1,       # The ratio of informative features that are linear combinations of other informative features.
#         class_sep=2.5,               # The separation between classes. Higher values make the classification task easier.
#         missing_pct=0.05,            # The percentage of cells to be replaced with NaN before imputation.
#         n_constant_cols=3,           # Number of columns with a single, constant value.
#         n_duplicate_cols=3,          # Number of columns that are exact copies of other columns.
#         n_correlated_pairs=3,        # Number of new columns that are highly correlated with existing columns.
#         n_categorical_cols=2,        # Number of non-numeric (string) columns to add.
#         outcome_name="outcome_var_1",# The name of the target variable column.
#         random_state=42,             # Seed for reproducibility.
#     )
#     synthetic_df = generator.generate_data()

#     logger.info("\nGenerated DataFrame head:\n%s", synthetic_df.head())

#     logger.info("\nInfo:")
#     # .info() prints to stdout, so we can't directly capture it with the logger easily.
#     # For debugging, this is often acceptable.

#     # Save to a CSV file for use in the pipeline
#     output_path = "synthetic_data_for_testing.csv"
#     synthetic_df.to_csv(output_path, index=False)
#     logger.info("\nData saved to '%s'", output_path)
