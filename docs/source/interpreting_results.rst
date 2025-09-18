Interpreting Experiment Results
===============================

After running an experiment using the ``example_usage.ipynb`` notebook, a suite of visualizations is generated in your results directory. These plots are created by the ``GA_results_explorer`` class and are designed to help you understand the experiment's outcomes. This guide explains how to interpret the most important plots.

Understanding Hyperparameter Impact
-----------------------------------

These plots help you identify which configuration settings had the most significant effect on model performance.

``plot_combined_anova_feature_importances``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is one of the most important plots for refining future experiments. It uses an ANOVA F-test to rank every hyperparameter (from data processing, feature selection, and the GA itself) by its impact on the outcome metric (e.g., `auc`).

*   **What to look for**: Look for parameters at the top of the chart. These are the settings that you should focus on tuning. Parameters at the bottom have little to no statistical impact on the results and can often be set to a fixed value in future runs.

``plot_parameter_distributions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This plot shows a series of boxplots or violin plots, one for each hyperparameter. It visualizes the distribution of performance scores for each value of a given parameter.

*   **What to look for**: Identify parameter values that consistently lead to higher scores. For example, if `resample='oversample'` consistently yields a higher median AUC than `'undersample'`, you might want to focus on oversampling in your next experiment.

Analyzing Base Learner Performance
----------------------------------

These plots reveal which models were most effective and how they interacted within ensembles.

``plot_base_learner_feature_importance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This plot shows how often each base learner (e.g., `XGBoost`, `LogisticRegression`) was included in the final, best-performing ensembles.

*   **What to look for**: Models with high importance are valuable contributors. If a model consistently has very low or zero importance, you might consider removing it from your `modelFuncList` to simplify the search space for the GA.

``plot_feature_cooccurrence``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This heatmap shows which pairs of base learners tend to appear together in high-performing ensembles.

*   **What to look for**: Bright squares indicate a strong positive correlation. This suggests that the two models are complementary and work well together. This can help you understand the types of model diversity that are beneficial for your specific problem.

Assessing Genetic Algorithm Behavior
------------------------------------

These plots help you diagnose the health and efficiency of the evolutionary search process.

``plot_all_convergence``
~~~~~~~~~~~~~~~~~~~~~~~~

This plot shows the fitness (performance) of the best individual over generations for every single grid search run.

*   **What to look for**:
    *   **Healthy Convergence**: The curve should rise steadily and then plateau. This indicates the GA has found a good solution and is no longer making significant improvements.
    -   **Insufficient Generations**: If the curve is still rising steeply at the end, it means the GA was stopped prematurely. You should increase the `n_generations` parameter.
    -   **Premature Convergence**: If the curve flattens out almost immediately, the GA may be getting stuck in a local optimum. You might need to increase the `mutation_rate` or population size.

``plot_performance_vs_size``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This scatter plot shows the relationship between the number of base learners in an ensemble (its size) and its performance.

*   **What to look for**: Often, performance will increase with ensemble size up to a point and then plateau or even decrease. This helps you identify an optimal range for the number of base learners, balancing performance with computational complexity.

Final Validation
----------------

While the plots provide deep insights into the search process, the most critical result is the final evaluation on the hold-out test set. At the end of the `example_usage.ipynb` notebook, the ``EnsembleEvaluator`` is used to produce two dataframes:

-   **Results on TEST SET**: Performance on the data split used for GA fitness evaluation.
-   **Results on VALIDATION (HOLD-OUT) SET**: Performance on a completely unseen dataset that was set aside at the very beginning.

The scores on the **VALIDATION (HOLD-OUT) SET** represent the most honest and unbiased measure of your final model's generalization ability. Always use this as the primary metric for reporting final performance.