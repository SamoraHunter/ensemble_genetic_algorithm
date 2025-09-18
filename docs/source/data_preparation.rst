Data Preparation Guide
======================

Preparing your dataset correctly is the most important step to ensure the experiment pipeline runs smoothly. This guide outlines the required format and conventions for your input data.

Core Requirements
-----------------

-   **File Format**: The input data must be a single CSV (Comma-Separated Values) file.
-   **Data Type**: All columns, including features and the outcome variable, must be numeric. Categorical variables must be pre-processed into a numeric format (e.g., through one-hot encoding or label encoding).
-   **Missing Values**: The pipeline can handle missing values, but they should be represented as standard empty fields or ``NaN``.

Outcome Variable Naming Convention
----------------------------------

This is a strict requirement. The target or outcome variable for the binary classification task **must** be named with the suffix ``_outcome_var_1``.

-   **Correct Name**: ``my_outcome_outcome_var_1``, ``has_disease_outcome_var_1``
-   **Incorrect Name**: ``outcome``, ``target``, ``y_value``

The column should contain only two unique numeric values, typically `0` and `1`.

Feature Columns
---------------

All other columns in the CSV will be treated as input features.

-   **Naming**: Feature names should not contain special characters that might interfere with processing. Use letters, numbers, and underscores.
-   **Data Types**: As mentioned, all feature columns must be numeric.

Example CSV Structure
---------------------

Here is a simplified example of a valid input CSV file:

.. code-block:: text

   feature_a,feature_b,age,is_smoker_outcome_var_1
   0.5,1.2,45,1
   0.8,0.9,32,0
   0.2,1.5,67,1
   ...

Data Pre-processing in the Pipeline
-----------------------------------

While the initial data must be numeric, the experiment pipeline itself performs several pre-processing steps based on the selected hyperparameters for each run. These include:

-   **Missing Value Imputation**: The pipeline can remove columns with a high percentage of missing values.
-   **Feature Scaling**: Features can be scaled (e.g., using StandardScaler).
-   **Feature Selection**: Methods like ANOVA F-test or correlation thresholds can be applied to select a subset of features.
-   **Resampling**: Techniques like oversampling or undersampling can be used to handle imbalanced datasets.

By adhering to the initial format, you ensure that these automated steps can be applied correctly across all experiment iterations.