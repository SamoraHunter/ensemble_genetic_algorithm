# Evaluating Final Models with EnsembleEvaluator

This guide explains how to use the `EnsembleEvaluator` class to perform a final, unbiased evaluation of the best ensembles discovered by the genetic algorithm.

---

## Purpose of the `EnsembleEvaluator`

The genetic algorithm identifies promising ensembles by evaluating them on a validation set that is part of the training data split. However, to get a true measure of how these models will perform in the real world, they must be tested on completely unseen data.

The `EnsembleEvaluator` class is designed for this exact purpose. It:

1.  Loads the best-performing ensembles from your experiment's results log (`final_grid_score_log.csv`).
2.  Reconstructs these ensembles, including their specific base learners and weights.
3.  Evaluates them on a hold-out **test set** and a separate **validation set** that were not used during the GA training process.

This provides an unbiased assessment of generalization performance.

---

## How it Works

The final cell in the `notebooks/example_usage.ipynb` notebook demonstrates the standard workflow.

### 1. Initialization

First, an instance of `EnsembleEvaluator` is created:

```python
from ml_grid.util.evaluate_ensemble_methods import EnsembleEvaluator

evaluator = EnsembleEvaluator(
    input_csv_path=input_csv_path,
    outcome_variable="outcome_var_1",
    initial_param_dict={"resample": None},
    debug=False
)
```

-   `input_csv_path`: Path to the *original, full dataset*. The evaluator handles splitting the data internally.
-   `outcome_variable`: The name of the target column.
-   `initial_param_dict`: A dictionary for initial data processing parameters.

### 2. Loading Results

The evaluator needs the results log from the main experiment to know which models to test.

```python
results_df = pd.read_csv(results_csv_path)
```

### 3. Evaluation on Test and Validation Sets

The evaluator has two main methods:

-   `evaluate_on_test_set_from_df(...)`: This method identifies the best ensemble for each grid search iteration (based on the validation performance during the GA run) and evaluates it on the hold-out **test set**. The test set is created from the `test_sample_n` samples specified during the experiment setup.

-   `validate_on_holdout_set_from_df(...)`: This method evaluates the same top ensembles on a *second* hold-out set, which it calls the "validation (hold-out) set". This provides an additional layer of validation on data that was unseen by both the GA and the initial test set evaluation.

```python
# Evaluate on the test set
test_results_df = evaluator.evaluate_on_test_set_from_df(
    results_df, weighting_methods_to_test
)

# Evaluate on the validation (hold-out) set
validation_results_df = evaluator.validate_on_holdout_set_from_df(
    results_df, weighting_methods_to_test
)
```

### 4. Interpreting the Output

The methods return Pandas DataFrames containing the performance metrics (AUC, accuracy, etc.) of the best ensembles on the respective unseen datasets.

By comparing these scores to the scores reported during the GA run, you can check for overfitting. If the performance on the test/validation sets is significantly lower than the performance reported in `final_grid_score_log.csv`, it suggests the ensembles may have overfit to the training data.

---

## Summary

The `EnsembleEvaluator` is a critical final step in the experimental pipeline. It ensures that the models you select are not just good at memorizing the training data but can generalize effectively to new, unseen examples, which is the ultimate goal of any predictive modeling task.