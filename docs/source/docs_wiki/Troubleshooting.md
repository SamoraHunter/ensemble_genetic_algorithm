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
2.  **Adjust `percent_missing`**: In your `config.yml`, lower the `percent_missing` threshold under `grid_params` (e.g., from `99.8` to `90.0`) to be more aggressive about removing columns that have any missing data.
3.  **Prune `model_list`**: In your `config.yml`, remove the model that is causing the error from the `model_list` under `global_params`.

---

## Runtime and Performance Issues

#### Problem: The experiment runs out of memory (`MemoryError`).

**Solutions:**
1.  **Reduce Population Size**: In your `config.yml`, use smaller values for `pop_params` under the `ga_params` section (e.g., `[32]`).
2.  **Reduce Data Size**: For testing, either use a smaller input CSV file or set `testing: True` in your `config.yml` under `global_params`. You can also set `test_sample_n` to a small number (e.g., `1000`) to sample your data.
3.  **Disable Model Caching**: In your `config.yml`, set `store_base_learners: [False]` under `grid_params`.

#### Problem: The experiment is running very slowly.

**Solutions:**
1.  **Start Small**: For initial runs, set `n_iter` to a low number (e.g., 1-3) and `testing: True` in your `config.yml` under `global_params`.
2.  **Use Model Caching**: For subsequent runs, set `use_stored_base_learners: [True]` in `grid_params` to avoid retraining models. See Best Practices.
3.  **Simplify Weighting**: In your `config.yml`, limit the `weighted` list under `grid_params` to `["unweighted"]` for fast runs. `'de'` and `'ann'` are much slower.

#### Problem: The genetic algorithm's fitness score is not improving (the convergence plot is flat).

**Solutions:**
1.  **Increase Mutation/Crossover**: The search might be stuck. In your `config.yml`, try increasing the `mutpb` (mutation rate) or `cxpb` (crossover rate) under `grid_params` to encourage more exploration.
2.  **Increase Population Size**: In `config.yml`, use larger values for `pop_params` under `ga_params` to introduce more diversity.
3.  **Check Model Suitability**: The base learners in your `model_list` (in `config.yml`) may not be a good fit for your data. Try adding or swapping in different types of models.