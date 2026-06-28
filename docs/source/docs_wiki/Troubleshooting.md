# Troubleshooting Guide

This guide provides solutions to common errors and issues you might encounter while using the **Ensemble Genetic Algorithm** project. Please also refer to the {doc}`Best_Practices` for proactive tips on running experiments efficiently.

---

## Environment and Setup Issues

### Problem: `ModuleNotFoundError: No module named '...'`

This is the most common issue and usually means the project's virtual environment is not active or was not set up correctly.

**Solutions:**
1.  **Activate the Environment**: Make sure you have activated the correct virtual environment before running any scripts.
    ```bash
    source ga_env/bin/activate  # If you used setup.sh
    # OR
    source .venv/bin/activate   # If you installed manually
    ```
2.  **Verify Python Version**: Ensure you are using Python >=3.12 as required by `pyproject.toml`.
    ```bash
    python --version  # Should show Python 3.12.x or higher
    ```
3.  **Re-install Dependencies**: If the error persists, your installation might be incomplete. Re-install the dependencies.
    ```bash
    pip install .
    ```
4.  **Check for Missing Optional Dependencies**: If you're using GPU support, ensure PyTorch with CUDA is installed:
    ```bash
    pip list | grep torch
    ```

### Problem: `ImportError: cannot import name '...' from 'ml_grid'`

This indicates that the package is not properly installed or your Python path is incorrect.

**Solutions:**
1.  **Reinstall in Editable Mode**: Run from the project root:
    ```bash
    pip install -e .
    ```
2.  **Verify Installation**: Check if the module can be imported:
    ```bash
    python -c "from ml_grid.pipeline import main_ga; print('Import successful')"
    ```

### Problem: GPU is not being used by PyTorch models

**Solutions:**
1.  **Check Installation**: Ensure you installed the GPU-enabled version of PyTorch. The easiest way is to re-run the setup script with the `--gpu` flag: `./setup.sh --gpu`.
2.  **Verify CUDA**: From your terminal, run `nvidia-smi`. This command should list your NVIDIA GPU. If it doesn't, you may have an issue with your NVIDIA drivers.
3.  **Check Environment Variables**: Make sure the `CUDA_VISIBLE_DEVICES` environment variable is not set to `-1`, as this explicitly disables GPU access.
4.  **Verify PyTorch CUDA**:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```
    Should output `True`.

---

## Data-Related Errors

### Problem: The experiment fails immediately with a `KeyError` or `ValueError` related to a column name.

**Solutions:**
1.  **Check Outcome Variable Name**: This is a strict requirement. The column name for your target variable **must** end with the suffix `_outcome_var_1`. Please review the {doc}`Data_Preparation_Guide`.
2.  **Check for Non-Numeric Data**: The framework expects all columns to be numeric. Ensure you have preprocessed your data to remove or encode any string or categorical columns.
3.  **Verify CSV Format**: Check that there are no hidden characters or malformed rows in your CSV file.

### Problem: A specific base learner fails with `ValueError: Input contains NaN`.

This happens because some models (like scikit-learn's `LogisticRegression` or `SVC`) cannot natively handle missing values, while others (like `XGBoost`) can.

**Solutions:**
1.  **Perform Imputation**: Preprocess your dataset to impute (fill in) missing values before running the experiment.
2.  **Adjust `percent_missing`**: In your `config.yml`, lower the `percent_missing` threshold under `grid_params` (e.g., from `99.8` to `90.0`) to be more aggressive about removing columns that have any missing data.
3.  **Prune `model_list`**: In your `config.yml`, remove the model that is causing the error from the `model_list` under `global_params`.

### Problem: `ValueError: Number of classes does not match number of labels`

This occurs when a model trained on one set of classes is evaluated on data with different class distributions.

**Solutions:**
1.  **Stratified Split**: Ensure your train/test split uses stratification to preserve class distribution.
2.  **Check Outcome Variable**: Verify that both training and test sets contain samples from both classes (binary classification).

---

## Runtime and Performance Issues

### Problem: The experiment runs out of memory (`MemoryError` or `CUDA out of memory`)

**Solutions:**
1.  **Reduce Population Size**: In your `config.yml`, use smaller values for `pop_params` under the `ga_params` section (e.g., `[32]` instead of `[64, 128]`).
2.  **Reduce Data Size**: For testing, either use a smaller input CSV file or set `testing: True` in your `config.yml` under `global_params`. You can also set `test_sample_n` to a small number (e.g., `1000`) to sample your data.
3.  **Disable Model Caching**: In your `config.yml`, set `store_base_learners: False` to avoid storing trained models in memory.
4.  **GPU-Specific**: If using GPU, reduce batch sizes in PyTorch model hyperparameters or use gradient accumulation.

### Problem: The experiment is running very slowly

**Solutions:**
1.  **Start Small**: For initial runs, set `n_iter` to a low number (e.g., 1-3) and `testing: True` in your `config.yml` under `global_params`.
2.  **Use Model Caching**: For subsequent runs, set `use_stored_base_learners: True` in `grid_params` to avoid retraining models.
3.  **Simplify Weighting**: In your `config.yml`, limit the `weighted` list under `grid_params` to `["unweighted"]` for fast runs. `'de'` and `'ann'` are much slower.
4.  **Reduce Generations**: Lower the values in `g_params` (e.g., `[50]`) for quicker experiments.

### Problem: The genetic algorithm's fitness score is not improving (the convergence plot is flat)

**Solutions:**
1.  **Increase Mutation/Crossover**: The search might be stuck. In your `config.yml`, try increasing the `mutpb` (mutation rate) or `cxpb` (crossover rate) under `grid_params` to encourage more exploration.
2.  **Increase Population Size**: In `config.yml`, use larger values for `pop_params` under `ga_params` to introduce more diversity.
3.  **Check Model Suitability**: The base learners in your `model_list` (in `config.yml`) may not be a good fit for your data. Try adding or swapping in different types of models.
4.  **Feature Selection**: Adjust the `feature_selection_method` or correlation threshold (`corr`) to allow different feature subsets.

---

## GA Convergence Failures

When the genetic algorithm fails to converge, it means the evolutionary process does not find an optimal or satisfactory solution within the allocated generations. This can occur due to various reasons related to population diversity, fitness landscape, or parameter settings.

### Understanding Convergence

A genetic algorithm is considered to have converged when:
- The best-performing individual's fitness score plateaus over multiple generations
- The population stabilizes around a small set of high-fitness solutions
- No significant improvement occurs after a defined number of generations

### Common Signs of Convergence Failure

| Symptom | Description |
|---------|-------------|
| **Flat Fitness Plot** | Best fitness score remains constant or fluctuates randomly across generations without trend |
| **Rapid premature convergence** | Population converges too quickly to suboptimal solutions (typically within first 10-20 generations) |
| **No improvement after burn-in** | No performance gains despite extended generation counts (>50) |
| **High diversity with no direction** | Constant population reorganization without upward trend in fitness |

### Diagnosing Convergence Failures

#### 1. Check Initial Population Diversity
```python
from ml_grid.pipeline import data, main_ga
from ml_grid.util.global_params import global_parameters

# Run a short experiment and inspect population diversity
global_params = global_parameters(config_path='config.yml')
ml_grid_object = data.pipe(
    global_params=global_params,
    file_name="data/dataset.csv",
    drop_term_list=[],
    local_param_dict={'pop': 64},
    param_space_index=0,
)

result = main_ga.run(
    ml_grid_object,
    local_param_dict={'cxpb': 0.8, 'mutpb': 0.2},
    global_params=global_params
).execute()

# Check logs for population statistics
# Look for: "Population diversity" and "Best fitness" logs
```

#### 2. Analyze Feature Transformation Log
Low feature counts can limit solution quality:
```python
print(ml_grid_object.feature_transformation_log)

# If features_after is very low (<3), consider relaxing thresholds
# or removing the n_features parameter to use all available features
```

#### 3. Inspect Genetic Operator Parameters
Verify your configuration includes balanced exploration/exploitation:

```yaml
grid_params:
  cxpb: 0.6-0.9   # Crossover rate - too low reduces exploration
  mutpb: 0.1-0.5  # Mutation rate - too high prevents convergence
```

### Diagnostic Commands and Checks

1. **Enable Verbose Logging**:
   ```yaml
   global_params:
     verbose: 3  # or 4 for maximum GA debugging output
   ```

2. **Monitor Key Metrics in Logs**:
   - Population diversity (should remain >30% throughout)
   - Best fitness score trend (should show gradual improvement)
   - Number of unique individuals per generation

3. **Check Early Generation Performance**:
   If fitness plateaus after <10 generations, the population likely lacks sufficient genetic diversity.

### Solutions for Convergence Failures

#### Solution 1: Adjust Genetic Operators
```yaml
ga_params:
  pop_params: [64, 128]  # increase to 256 if memory allows
  g_params: [100, 200]   # extend generations with no improvement
grid_params:
  cxpb: 0.7              # moderate crossover for balance
  mutpb: 0.3             # sufficient mutation to maintain diversity
```

#### Solution 2: Enable Weighted Ensemble Search
```yaml
grid_params:
  weighted: ["unweighted", "de"]  # allow ensemble weighting exploration
```
Unweighted mode may explore slower but provides more stable convergence.

#### Solution 3: Modify Feature Selection Strategy
```yaml
grid_params:
  feature_selection_method: null  # use all features
  corr: 0.75                     # reduce from default 0.95
```

#### Solution 4: Use Multiple Starting Points
Run several independent experiments with different random seeds:

```python
import os
os.environ['PYTHONHASHSEED'] = '42'  # Set seed for reproducibility

# Try multiple starting configurations
for seed in [42, 123, 456, 789]:
    local_param_dict = {
        'seed': seed,
        'cxpb': 0.8,
        'mutpb': 0.3
    }
    # Run experiment with each seed
```

### When GA Truly Fails

If convergence fails despite these adjustments:

1. **Verify Data Quality**: Check for label noise, class imbalance, or insufficient samples
2. **Check Base Learner Diversity**: Ensure `model_list` contains disparate model types (e.g., tree-based + linear models)
3. **Consider Problem Suitability**: Some problems may not benefit from ensemble optimization via GA
4. **Try Alternative Optimization**: Consider random search or Bayesian optimization for simpler configuration spaces

---

## Configuration Errors

### Problem: Unknown parameter in config.yml

**Solutions:**
1.  **Check Section Names**: Ensure parameters are under the correct section (`global_params`, `ga_params`, `grid_params`).
2.  **Refer to Hyperparameter Reference**: Check {doc}`Hyperparameter_Reference` for valid parameter names and acceptable values.
3.  **Use config.yml.example**: Copy parameters from the example file rather than typing them manually.

### Problem: Grid search takes too long or explores wrong space

**Solutions:**
1.  **Reduce Parameter Combinations**: Remove high-cardinality options from your lists in `config.yml`.
2.  **Check `n_iter` Value**: This controls how many random combinations are sampled, not the total grid size.
3.  **Use Testing Mode**: Set `testing: True` for a smaller default grid.

---

## Common Workflow Issues

### Problem: Results directory is empty or missing expected files

**Solutions:**
1.  **Check Log Output**: Look for error messages before the script exits.
2.  **Verify Write Permissions**: Ensure your user has write access to `base_project_dir`.
3.  **Check Experiment Completion**: The experiment must complete all generations and final evaluation to save key artifacts.

### Problem: Plots are missing or cannot be generated

**Solutions:**
1.  **Ensure Evaluation Step**: Run with `--evaluate` flag if using command line.
2.  **Check Backend**: For headless servers, set matplotlib backend: `matplotlib.use('Agg')`.

---

## Getting Help

If the above solutions do not resolve your issue:

1.  Consult the {doc}`Home` for project overview links
2.  Review code comments and docstrings in `ml_grid/pipeline/main_ga.py`
3.  Check the repository's issue tracker for similar problems
4.  Provide detailed error messages including Python version, OS, and installed packages when seeking help
