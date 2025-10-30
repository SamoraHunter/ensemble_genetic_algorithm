import logging
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("ensemble_ga")


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
    local_param_dict: Dict, drop_list: List[str], df: pd.DataFrame, chunk_size: int = 50
) -> List[str]:
    """Identifies highly correlated column pairs and adds them to a drop list.

    This function calculates the correlation matrix of a DataFrame in chunks to
    manage memory usage. It identifies pairs of columns where the absolute
    correlation coefficient exceeds a specified threshold and returns a list
    of these pairs.

    Args:
        local_param_dict: A dictionary containing local parameters, including
            the 'corr' threshold.
        drop_list: A list of column names already marked for dropping. This
            argument is included for signature consistency but is not used.
        df: The input DataFrame to analyze.
        chunk_size: The number of columns to process in each chunk.
            Defaults to 50.

    Returns:
        An updated list of columns to drop, including one column from each
        highly correlated pair.
    """

    # Define the correlation threshold
    threshold = local_param_dict.get("corr", 0.25)

    # Ensure chunk_size is at least 1
    if chunk_size <= 0:
        chunk_size = 1

    # Remove non-numeric columns
    numeric_columns = df.select_dtypes(include=["number"]).columns
    df_numeric = df[numeric_columns]

    if df_numeric.empty:
        return []

    n_cols = len(df_numeric.columns)
    to_drop = set()

    # Split columns into chunks for memory efficiency
    column_chunks = [
        df_numeric.columns[i : i + chunk_size]
        for i in range(0, n_cols, chunk_size)
    ]

    with tqdm(total=n_cols, desc="Calculating Correlations") as pbar:
        for i, chunk_cols in enumerate(column_chunks):
            # Define the columns to correlate against: the current chunk and all subsequent chunks
            remaining_cols = df_numeric.columns[i * chunk_size :]

            # Calculate correlation for the current slice of the matrix
            corr_matrix_chunk = df_numeric[remaining_cols].corr(numeric_only=True).abs()

            # We only need to check correlations of the current chunk against all remaining columns
            # This is equivalent to the top-left block of the chunk's correlation matrix
            sub_matrix = corr_matrix_chunk.loc[chunk_cols, :]

            # Find highly correlated pairs
            for col1 in chunk_cols:
                # Find correlations above the threshold, excluding self-correlation
                correlated_series = sub_matrix.loc[col1][sub_matrix.loc[col1] > threshold]
                for col2, _ in correlated_series.items():
                    # If two columns are correlated, and we haven't already decided to drop col1, drop col2.
                    if col1 != col2 and col1 not in to_drop:
                        to_drop.add(col2)
            pbar.update(len(chunk_cols))

    logger.info(f"Identified {len(to_drop)} columns to drop due to high correlation.")

    # Return a list of unique columns to drop
    return sorted(list(to_drop))
