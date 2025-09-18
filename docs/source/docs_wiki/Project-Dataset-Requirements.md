# Project Dataset Requirements

For the **Ensemble Genetic Algorithm** project to function correctly, your input dataset must adhere to specific requirements:

-   **Format**: The data should be a numeric data matrix, typically represented as a Pandas DataFrame.
-   **Outcome Variable**: It must include a binary outcome variable (e.g., `0` or `1`).
-   **Naming Convention**: The outcome variable column **must** have the suffix `_outcome_var_1`. For example, if your outcome is `disease_status`, the column name should be `disease_status_outcome_var_1`.

For a practical example of the expected feature column naming conventions and overall data structure, please refer to the synthetic dataset generated for the unit tests within this project, or explore the `pat2vec` project:
https://github.com/SamoraHunter/pat2vec/tree/main