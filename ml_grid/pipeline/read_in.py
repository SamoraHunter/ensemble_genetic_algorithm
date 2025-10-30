import logging
import random

import numpy as np
import pandas as pd
import polars as pl


class read:
    """Reads a CSV file into a pandas DataFrame, with an option to use Polars.

    Attributes:
        raw_input_data (pd.DataFrame): The loaded data as a pandas DataFrame.
    """

    def __init__(self, input_filename: str, use_polars: bool = False):
        """Initializes the read class and loads data from a CSV file.

        This class attempts to read a CSV file using the fast Polars library
        first (if `use_polars` is True), and falls back to pandas if an error
        occurs. If Polars is not used, it reads directly with pandas.

        Args:
            input_filename: The path to the CSV file to be read.
            use_polars: If True, attempts to use Polars for faster reading.
                Defaults to False.
        """
        self.logger = logging.getLogger("ensemble_ga")
        self.logger.info("Init main >read on %s", input_filename)
        if use_polars:
            try:
                self.raw_input_data = pl.read_csv(input_filename, ignore_errors=True)
                self.raw_input_data = self.raw_input_data.to_pandas()
            except Exception as e:
                self.logger.warning("Error reading with Polars: %s", e)
                self.logger.info("Trying to read with Pandas...")
                try:
                    self.raw_input_data = pd.read_csv(input_filename)
                except Exception as e:
                    self.logger.error("Error reading with Pandas: %s", e)
                    self.raw_input_data = pd.DataFrame()
        else:
            try:
                self.raw_input_data = pd.read_csv(input_filename)
            except Exception as e:
                self.logger.error("Error reading with Pandas: %s", e)
                self.raw_input_data = pd.DataFrame()


class read_sample:
    def __init__(
        self, input_filename: str, test_sample_n: int, column_sample_n: int
    ) -> None:
        """Reads a sampled subset of a CSV file into a pandas DataFrame.

        This class is designed to efficiently read a sample of rows and/or
        columns from a large CSV file. It ensures that certain 'necessary'
        columns are always included if they exist.

        The sampling logic is as follows:
        - If `test_sample_n` > 0, it randomly samples that many rows.
        - If `column_sample_n` > 0, it randomly samples that many columns,
          always including a predefined set of necessary columns.

        After sampling, it validates that the 'outcome_var_1' column, if
        present, contains more than one unique class.

        Args:
            input_filename: The path to the input CSV file.
            test_sample_n: The number of rows to randomly sample. If 0, all
                rows are read.
            column_sample_n: The number of columns to randomly sample. If 0,
                all columns are read.

        Raises:
            ValueError: If, after sampling, the 'outcome_var_1' column
                contains fewer than two unique classes.
        """
        self.filename = input_filename
        self.logger = logging.getLogger("ensemble_ga")
        self.logger.info("Init main > read_sample on %s", self.filename)

        necessary_columns = ["outcome_var_1", "age", "male"]
        read_csv_args = {}

        if test_sample_n > 0:
            total_rows = sum(1 for line in open(self.filename))
            if test_sample_n < total_rows:
                read_csv_args["skiprows"] = np.random.choice(
                    np.arange(1, total_rows), total_rows - test_sample_n, replace=False
                )

        if column_sample_n > 0:
            all_columns = pd.read_csv(self.filename, nrows=1).columns.tolist()
            if column_sample_n < len(all_columns):
                # Ensure necessary columns are included if they exist
                selected_necessary = [
                    col for col in necessary_columns if col in all_columns
                ]
                remaining_columns = [
                    col for col in all_columns if col not in selected_necessary
                ]

                # Calculate how many more columns to sample
                n_additional_cols = column_sample_n - len(selected_necessary)
                n_additional_cols = max(0, n_additional_cols)

                # Sample additional columns
                selected_additional = random.sample(
                    remaining_columns, min(len(remaining_columns), n_additional_cols)
                )

                read_csv_args["usecols"] = selected_necessary + selected_additional

        try:
            self.raw_input_data = pd.read_csv(self.filename, **read_csv_args)
        except Exception as e:
            self.logger.error("Error reading sampled CSV: %s", e)
            self.raw_input_data = pd.DataFrame()

        if (
            self.raw_input_data is not None
            and "outcome_var_1" in self.raw_input_data.columns
        ):
            classes = self.raw_input_data["outcome_var_1"].unique()
            if len(classes) < 2:
                raise ValueError(
                    "Outcome variable does not have both classes post sampling."
                )
