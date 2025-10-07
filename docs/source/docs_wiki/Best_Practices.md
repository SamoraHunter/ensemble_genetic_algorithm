# Best Practices and Tips

This guide provides a collection of best practices, tips, and common workflows to help you use the **Ensemble Genetic Algorithm** project more effectively and efficiently.

---

## 1. Starting a New Experiment: Start Small

When starting with a new dataset or a new research question, avoid running a large-scale experiment immediately.

-   **Use the Test Grid**: In your experiment script (e.g., `example_usage.ipynb`), set `testing=True` when calling `ml_grid.pipeline.data.pipe`. This uses a much smaller, predefined hyperparameter grid (`nb_params`, `pop_params`, `g_params`) that runs very quickly, allowing you to verify that your entire pipeline works without errors.
    ```yaml
    global_params:
      testing: True
    ```
-   **Limit Iterations**: In `config.yml`, set `n_iter` to a low value (e.g., 3-5) for initial test runs. This is enough to ensure that the loop executes, data is processed, and results are saved correctly.

-   **Sample Your Data**: If your dataset is very large, consider creating a smaller, representative sample (e.g., 10-20% of the original data) for initial exploration. This will dramatically speed up iteration cycles.

---

## 2. Tuning the Genetic Algorithm

After initial tests, use the results to guide the tuning of the GA itself.

-   **Check Convergence**: Examine the `plot_all_convergence` output.
    -   If the fitness curves are still trending upwards at the final generation, you should increase the values in `g_params` (number of generations) in `grid_param_space_ga.py`.
    -   If the curves flatten out very early, you might be able to reduce `g_params` in your `config.yml` to save computation time in future runs.

-   **Balance Population vs. Runtime**: The `pop_params` (population size) is a trade-off. A larger population explores the search space more thoroughly in each generation but increases runtime. Start with the defaults and only increase if you suspect the GA is failing to find good solutions due to a lack of diversity.

-   **Stick to Default Evolutionary Rates**: The crossover (`cxpb`) and mutation (`mutpb`) probabilities usually work well with default values (e.g., `cxpb` around 0.8, `mutpb` around 0.2). Only tune these if you observe specific issues like premature convergence (too little mutation) or chaotic search (too much mutation).

---

## 3. Managing Runtimes and Resources

Large-scale experiments can be computationally expensive. Here’s how to manage them:

-   **Use Model Caching Wisely**:
    -   `store_base_learners=True`: Set this in the grid for an initial, comprehensive run. It will save every trained base learner to disk, which can consume gigabytes of space.
    -   `use_stored_base_learners=True`: In subsequent runs, set this to `True` in your `config.yml`. The experiment will load and reuse the cached models instead of retraining them, which can reduce runtime by over 90% if you are only changing GA parameters like `population_size` or `ensemble_weighting_method`.

-   **Be Mindful of Weighting Methods**: The `ensemble_weighting_method` has a major impact on runtime.
    -   `'unweighted'` is extremely fast.
    -   `'de'` (Differential Evolution) and `'ann'` (Artificial Neural Network) are significantly more computationally expensive. Consider using them only in your final, most promising experiment configurations.

---

## 4. Iterating on Results for Better Models

The goal of the first experiment is not to find the perfect model, but to learn about the problem space.

-   **Identify Key Hyperparameters**: Use the `plot_combined_anova_feature_importances` and `plot_parameter_distributions` plots. These will show you which hyperparameters have the biggest impact on performance.

-   **Refine Your Search Grid**: Based on the insights above, go back to your `config.yml` and narrow the search space. For example, if a `resample` value of `None` consistently performs poorly, remove it from the list in `grid_params`. If a `pop_params` value of 128 always beats 64, you can focus on higher values in `ga_params`.

-   **Prune Your Model List**: Check the `plot_base_learner_feature_importance` plot. If certain models almost never appear in the top-performing ensembles, you can remove them from the `model_list` in your `config.yml` to focus the search on more promising algorithms.

---

## 5. Final Validation is Crucial

-   Always perform the final evaluation step using the `EnsembleEvaluator` on a hold-out test set.
-   The performance scores from this final step are the most realistic and unbiased measure of your model's ability to generalize to new data. The validation scores reported during the GA run can be optimistically biased.