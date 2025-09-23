# Troubleshooting Guide

This guide provides solutions to common errors and issues you might encounter while using the **Ensemble Genetic Algorithm** project.

---

## Environment and Setup Issues

#### Problem: `ModuleNotFoundError: No module named '...'`

This is the most common issue and usually means the project's virtual environment is not active or was not set up correctly.

**Solutions:**
1.  **Activate the Environment**: Make sure you have activated the correct virtual environment before running any scripts.
    ```bash
    source ga_env/bin/activate  # If you used setup.sh
    # OR
    source .venv/bin/activate   # If you installed manually
    ```
2.  **Re-install Dependencies**: If the error persists, your installation might be incomplete. Re-install the dependencies.
    ```bash
    pip install .
    ```

#### Problem: GPU is not being used by PyTorch models.

**Solutions:**
1.  **Check Installation**: Ensure you installed the GPU-enabled version of PyTorch. The easiest way is to re-run the setup script with the `--gpu` flag: `./setup.sh --gpu --force`.
2.  **Verify CUDA**: From your terminal, run `nvidia-smi`. This command should list your NVIDIA GPU. If it doesn't, you may have an issue with your NVIDIA drivers.
3.  **Check Environment Variables**: Make sure the `CUDA_VISIBLE_DEVICES` environment variable is not set to `-1`, as this explicitly disables GPU access.

---

## Data-Related Errors

#### Problem: The experiment fails immediately with a `KeyError` or `ValueError` related to a column name.

**Solutions:**
1.  **Check Outcome Variable Name**: This is a strict requirement. The column name for your target variable **must** end with the suffix `_outcome_var_1`. Please review the Data Preparation Guide.
2.  **Check for Non-Numeric Data**: The framework expects all columns to be numeric. Ensure you have preprocessed your data to remove or encode any string or categorical columns.

#### Problem: A specific base learner fails with `ValueError: Input contains NaN`.

This happens because some models (like scikit-learn's `LogisticRegression` or `SVC`) cannot natively handle missing values, while others (like `XGBoost`) can.

**Solutions:**
1.  **Perform Imputation**: Preprocess your dataset to impute (fill in) missing values before running the experiment.
2.  **Adjust `percent_missing`**: In `grid_param_space_ga.py`, lower the `percent_missing` threshold (e.g., from `99.8` to `90.0`) to be more aggressive about removing columns that have any missing data.
3.  **Prune `modelFuncList`**: Remove the specific model generator that is causing the error from the `model_class_list` in your experiment script.

---

## Runtime and Performance Issues

#### Problem: The experiment runs out of memory (`MemoryError`).

**Note**: Most performance parameters are defined in `ml_grid/util/grid_param_space_ga.py` and can be overridden with a `config.yml` file.

**Solutions:**
1.  **Reduce Population Size**: The most effective solution is often to reduce the `pop_params` (population size) in your configuration.
2.  **Reduce Data Size**: For testing, use a smaller `test_sample_n` or `column_sample_n` in the `ml_grid.pipeline.data.pipe` call.
3.  **Disable Model Caching**: If `store_base_learners` is `True`, the run can consume a lot of disk space and memory. Set it to `False` if you are memory-constrained.

#### Problem: The experiment is running very slowly.

**Solutions:**
1.  **Start Small**: For initial runs, set `n_iter` to a low number (e.g., 1-3) and use `testing=True` in your experiment script.
2.  **Use Model Caching**: For subsequent runs, use the model caching feature (`use_stored_base_learners=True`) to avoid retraining models. See Best Practices.
3.  **Simplify Weighting**: The `ensemble_weighting_method` has a huge impact. Use `'unweighted'` for fast runs. `'de'` and `'ann'` are much slower.

#### Problem: The genetic algorithm's fitness score is not improving (the convergence plot is flat).

**Solutions:**
1.  **Increase Mutation/Crossover**: The search might be stuck. Try increasing the `mutpb` (mutation rate) or `cxpb` (crossover rate) in `grid_param_space_ga.py` to encourage more exploration.
2.  **Increase Population Size**: A larger population (`pop_params`) can introduce more diversity, helping the algorithm escape local optima.
3.  **Check Model Suitability**: The base learners in your `model_class_list` may not be a good fit for your data. Try adding or swapping in different types of models.