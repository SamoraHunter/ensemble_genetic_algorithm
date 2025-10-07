# Data Preparation Guide

This guide provides detailed instructions and best practices for preparing your dataset for use with the **Ensemble Genetic Algorithm** project. Correctly formatting your data is the most critical step to ensure a successful experiment.

---

## Core Requirements

The framework is designed to work with a specific data structure. Please ensure your dataset meets the following criteria.

### 1. File Format

-   **CSV (Comma-Separated Values)**: Your dataset must be saved in a standard CSV file.

### 2. Data Types

-   **All Numeric**: All columns in your dataset, including features and the outcome variable, must be numeric (integers or floats).
-   **No Missing Headers**: Every column must have a header name.
-   **No Non-Numeric Data**: The framework is not designed to handle string, categorical (unless one-hot encoded), or object data types. Please preprocess your data to convert all columns to a numeric format.

### 3. Outcome Variable

This is the most important requirement.

-   **Binary**: The outcome variable must be binary, containing only two distinct values (e.g., `0` and `1`).
-   **Column Name Suffix**: The name of the outcome variable column **must** end with the suffix `_outcome_var_1`.
    -   Correct: `my_outcome_outcome_var_1`, `disease_outcome_var_1`
    -   Incorrect: `outcome`, `my_outcome_variable`

### 4. Feature Columns

-   There are no strict naming requirements for feature columns, but they must be unique and should not contain special characters that might interfere with processing.

### 5. Missing Values (NaNs)

-   The framework has a built-in mechanism to handle missing data. The `percent_missing` parameter in the configuration grid (`grid_param_space_ga.py`) allows you to specify a threshold for column removal. For example, a value of `99.8` will remove any column that has more than 99.8% missing values.
-   The framework has a built-in mechanism to handle missing data. The `percent_missing` parameter, configured in your `config.yml` under `grid_params`, allows you to specify a threshold for column removal. For example, a value of `99.8` will remove any column that has more than 99.8% missing values.
-   Remaining missing values in the data are typically handled by the base learners, many of which (like XGBoost) can handle NaNs natively. For others, you may need to perform imputation before running the experiment if you encounter errors.

---

## Example of a Valid CSV File

Here is a small example of what a correctly formatted CSV file (`my_data.csv`) would look like:

```csv
feature_A,feature_B,feature_C,readmission_outcome_var_1
10.5,2.3,1,0
8.2,4.1,0,1
9.1,3.3,1,0
7.6,2.9,0,1
11.0,1.5,1,0
```

In this example:
-   All values are numeric.
-   The outcome column, `readmission_outcome_var_1`, ends with the required suffix.
-   The outcome is binary (`0` or `1`).

By adhering to these guidelines, you can ensure that your data is correctly ingested and processed by the experiment pipeline.