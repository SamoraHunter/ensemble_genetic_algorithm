Evaluating Final Models
=======================

After running an experiment and analyzing the results, the final and most critical step is to evaluate the performance of the best-discovered ensembles on a completely unseen hold-out dataset. This process provides an unbiased measure of how well your models will generalize to new data.

The ``EnsembleEvaluator`` class is designed specifically for this purpose.

The Importance of a Hold-Out Set
--------------------------------

Throughout the genetic algorithm process, models are evaluated on a "test" set (a split from the training data) to calculate their fitness. While this is necessary for the evolutionary search, the models can become implicitly overfit to this test set.

The **hold-out validation set** is a portion of data that was set aside at the very beginning of the experiment and was **never** used during training, feature selection, or GA fitness evaluation. Performance on this set is the true measure of your model's generalization capability.

Using the ``EnsembleEvaluator``
-------------------------------

The last section of the ``notebooks/example_usage.ipynb`` notebook demonstrates how to use the ``EnsembleEvaluator``.

1.  **Initialization**: The evaluator is initialized with the path to the original dataset.

2.  **Evaluation**: The ``evaluate_on_test_set_from_df`` and ``validate_on_holdout_set_from_df`` methods are called. These methods:
    -   Identify the best-performing ensembles from the ``final_grid_score_log.csv`` results file.
    -   Load the corresponding saved models.
    -   Re-evaluate them on the test and hold-out validation sets, respectively.

Example Code
~~~~~~~~~~~~

The following code snippet from the example notebook shows this process in action:

.. code-block:: python

   from ml_grid.util.evaluate_ensemble_methods import EnsembleEvaluator
   import pandas as pd

   # 1. Define paths and parameters
   results_csv_path = ml_grid_object.base_project_dir + "final_grid_score_log.csv"
   outcome_variable = "outcome_var_1"

   # 2. Initialize the evaluator
   evaluator = EnsembleEvaluator(
       input_csv_path=input_csv_path,
       outcome_variable=outcome_variable,
   )

   # 3. Load experiment results
   results_df = pd.read_csv(results_csv_path)

   # 4. Evaluate on the test set (used during GA)
   test_results_df = evaluator.evaluate_on_test_set_from_df(
       results_df, weighting_methods_to_test=["unweighted", "de", "ann"]
   )
   print("\\n--- Results on TEST SET ---")
   display(test_results_df)

   # 5. Evaluate on the hold-out validation set (unbiased performance)
   validation_results_df = evaluator.validate_on_holdout_set_from_df(
       results_df, weighting_methods_to_test=["unweighted", "de", "ann"]
   )
   print("\\n--- Results on VALIDATION (HOLD-OUT) SET ---")
   display(validation_results_df)

Interpreting the Output
-----------------------

The evaluator produces two Pandas DataFrames:

-   **Results on TEST SET**: This shows the performance of the best models on the data used for fitness evaluation during the GA. These scores are often slightly optimistic.
-   **Results on VALIDATION (HOLD-OUT) SET**: This shows the performance on the completely unseen data. **These are the final, reportable scores** that represent the true generalization performance of your evolved ensembles.