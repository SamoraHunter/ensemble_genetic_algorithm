Hyperparameter Reference
========================

This page provides a detailed reference for all the hyperparameters that can be configured in the ``ml_grid/util/grid_param_space_ga.py`` file. These parameters control everything from data preprocessing to the behavior of the genetic algorithm.

Data Preprocessing and Feature Engineering
------------------------------------------

These parameters control how the raw data is transformed before being used by the models.

-   ``resample``: The resampling method to handle imbalanced datasets.
    -   **Values**: ``'undersample'``, ``'oversample'``, ``None``
-   ``scale``: Whether to apply feature scaling (e.g., StandardScaler).
    -   **Values**: ``True``, ``False``
-   ``percent_missing``: The maximum percentage of missing values allowed in a column before it is dropped.
    -   **Values**: e.g., ``99.9``, ``99.8``, ``99.7``
-   ``corr``: The correlation threshold for removing highly correlated features.
    -   **Values**: e.g., ``0.9``, ``0.99``
-   ``feature_selection_method``: The statistical method used for feature selection.
    -   **Values**: ``'anova'``, ``'markov_blanket'`` (if implemented)

Genetic Algorithm Parameters
----------------------------

These parameters define the behavior of the evolutionary search process.

-   ``cxpb``: The crossover probability for mating two individuals.
    -   **Values**: e.g., ``0.5``, ``0.75``, ``0.25``
-   ``mutpb``: The mutation probability for an individual.
    -   **Values**: e.g., ``0.2``, ``0.4``, ``0.8``
-   ``indpb``: The probability for each attribute of an individual to be mutated.
    -   **Values**: e.g., ``0.025``, ``0.05``, ``0.075``
-   ``t_size``: The tournament size for parent selection.
    -   **Values**: e.g., ``3``, ``6``, ``9``
-   ``nb_params``: The number of base learners in an individual ensemble.
    -   **Values**: e.g., ``[4, 8, 16, 32]``
-   ``pop_params``: The population size (number of ensembles) in each generation.
    -   **Values**: e.g., ``[32, 64]``
-   ``g_params``: The number of generations the algorithm will run.
    -   **Values**: e.g., ``[128]``

Ensemble and Model Caching
--------------------------

These parameters control how ensembles are constructed and how models are managed.

-   ``weighted``: The method used to weigh the predictions of base learners in an ensemble.
    -   **Values**: ``'ann'`` (Artificial Neural Network), ``'de'`` (Differential Evolution), ``'unweighted'``
-   ``use_stored_base_learners``: If ``True``, the pipeline will load pre-trained models from disk instead of retraining them.
    -   **Values**: ``True``, ``False``
-   ``store_base_learners``: If ``True``, all trained base learners will be saved to disk.
    -   **Values**: ``True``, ``False``
-   ``div_p``: A parameter for diversity-weighted scoring. A value greater than 0 penalizes ensembles with low diversity.
    -   **Values**: e.g., ``0``

Data Subset Selection
---------------------

The ``data`` dictionary allows for the inclusion or exclusion of specific feature groups. This is useful for exploring the impact of different data types.

-   ``age``: Include age-related features.
-   ``sex``: Include sex-related features.
-   ``bmi``: Include BMI-related features.
-   ``ethnicity``: Include ethnicity-related features.
-   ``bloods``: Include blood test result features.
-   ``diagnostic_order``: Include diagnostic order features.
-   ``drug_order``: Include drug order features.
-   ``annotation_n``: Include annotation features.
-   ``meta_sp_annotation_n``: Include meta-annotation features.
-   ``annotation_mrc_n``: Include MRC annotation features.
-   ``meta_sp_annotation_mrc_n``: Include meta MRC annotation features.
-   ``core_02``: Include core_02 features.
-   ``bed``: Include bed-related features.
-   ``news``: Include NEWS score features.

Each of these can be set to ``[True, False]`` in the grid to test runs with and without that feature group.