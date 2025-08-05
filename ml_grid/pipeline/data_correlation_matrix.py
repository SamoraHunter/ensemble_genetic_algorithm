import traceback
import pandas as pd
import numpy as np
from tqdm import tqdm


def correlation_coefficient(col1, col2):
    return col1.corr(col2)


def handle_correlation_matrix(local_param_dict, drop_list, df, chunk_size=50):
    """
    Calculate correlated columns in chunks.

    Calculates the correlation coefficient between each column in the input DataFrame
    using chunks to avoid memory issues. The correlation threshold is defined by
    the 'corr' key in the local_param_dict dictionary.

    Args:
        local_param_dict (dict): Dictionary containing local parameters, including the correlation threshold.
        drop_list (list): List to which correlated columns will be appended.
        df (pandas.DataFrame): Input DataFrame.
        chunk_size (int, optional): Size of each chunk for correlation calculation. Default is 50.

    Returns:
        list: List of correlated columns.
    """

    if chunk_size >= len(df):
        chunk_size = len(df) - 1
    # Define the correlation threshold
    threshold = local_param_dict.get("corr", 0.25)

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
            correlations = df_numeric[chunk].corr()
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
                correlated_cols = correlations[col][
                    (correlations[col] > threshold) & (correlations[col] != 1)
                ].index.tolist()
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
    drop_list = list(set(drop_list))

    return drop_list

