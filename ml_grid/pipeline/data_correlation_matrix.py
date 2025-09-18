import traceback
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple


def correlation_coefficient(col1: pd.Series, col2: pd.Series) -> float:
    """Calculates the Pearson correlation coefficient between two pandas Series.

    Args:
        col1: The first pandas Series.
        col2: The second pandas Series.

    Returns:
        The correlation coefficient as a float.
    """
    return col1.corr(col2)


def handle_correlation_matrix(
    local_param_dict: Dict, drop_list: List, df: pd.DataFrame, chunk_size: int = 50
) -> List[Tuple[str, str]]:
    """Identifies highly correlated column pairs and adds them to a drop list.

    This function calculates the correlation matrix of a DataFrame in chunks to
    manage memory usage. It identifies pairs of columns where the absolute
    correlation coefficient exceeds a specified threshold and returns a list
    of these pairs.

    Args:
        local_param_dict: A dictionary containing local parameters, including
            the 'corr' threshold.
        drop_list: A list to which correlated column pairs `(col1, col2)`
            will be appended.
        df: The input DataFrame to analyze.
        chunk_size: The number of columns to process in each chunk.
            Defaults to 50.

    Returns:
        A list of unique tuples, where each tuple contains a pair of
        column names that are correlated above the threshold.
    """

    # Define the correlation threshold
    threshold = local_param_dict.get("corr", 0.25)

    # Ensure chunk_size is at least 1
    if chunk_size <= 0:
        chunk_size = 1

    # Remove non-numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns
    df_numeric = df[numeric_columns]

    # Split columns into chunks
    column_chunks = [
        df_numeric.columns[i : i + chunk_size]
        for i in range(0, len(df_numeric.columns), chunk_size)
    ]

    # Iterate through each column chunk
    for chunk in tqdm(column_chunks, desc="Calculating Correlations"):
        # Calculate the correlation coefficients for the current chunk
        try:
            # Using abs() to consider both positive and negative correlations
            correlations = df_numeric[chunk].corr().abs()
        except Exception as e:
            print(
                "Encountered exception while calculating correlations for chunk", chunk
            )
            print(e)
            continue

        # Iterate through each column in the chunk
        for col in chunk:
            # Filter columns with correlation coefficient greater than the threshold
            try:
                # Exclude self-correlation (which is always 1)
                correlated_cols = correlations[col][
                    (correlations[col] > threshold) & (correlations[col] <= 1.0)
                ].index.tolist()
                # Explicitly remove self-correlation if present
                if col in correlated_cols:
                    correlated_cols.remove(col)
            except KeyError:
                print(
                    "Encountered KeyError while calculating correlations for column",
                    col,
                )
                print("Continuing with an empty list of correlated columns")
                correlated_cols = []

            # Add the correlated columns to the list
            drop_list.extend([(col, corr_col) for corr_col in correlated_cols])

    # Remove duplicates from the list
    # A frozenset is used to make pairs order-independent, e.g., (a, b) is same as (b, a)
    unique_pairs = {frozenset(pair) for pair in drop_list}
    # Convert back to list of tuples
    drop_list = [tuple(pair) for pair in unique_pairs]

    return drop_list
