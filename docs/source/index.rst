.. ga-project documentation master file, created by
   sphinx-quickstart on Wed Sep 17 16:51:36 2025.

##########################
Ensemble Genetic Algorithm
##########################

Welcome to the documentation for ``ga-project``!

``ga-project`` is a Python library that leverages genetic algorithms to construct
powerful machine learning model ensembles. It automates the challenging process of
selecting the best combination of models and their weights to maximize predictive
performance. This tool is for data scientists and machine learning engineers looking
to improve their model accuracy with sophisticated ensembling techniques.

.. note::

   This project is under active development and the API may change.

Installation
============

You can install ``ga-project`` from PyPI using pip:

.. code-block:: bash

   pip install ga-project

Example Workflow
================

The primary way to use this library is by configuring and running an experiment pipeline, as demonstrated in `notebooks/example_usage.ipynb`. This automates grid searching, model training, and evaluation.

Here is a simplified example of the core logic:

.. code-block:: python

   import ml_grid
   from ml_grid.pipeline import main_ga
   from ml_grid.model_classes_ga import (
       logisticRegressionModelGenerator,
       randomForestModelGenerator,
       XGBoostModelGenerator,
   )

   # 1. Define experiment parameters
   input_csv_path = "synthetic_data_for_testing.csv"
   base_project_dir = "HFE_GA_experiments/my_first_run/"

   # 2. Define the pool of base models for the Genetic Algorithm
   model_list = [
       logisticRegressionModelGenerator,
       randomForestModelGenerator,
       XGBoostModelGenerator,
   ]

   # 3. Set hyperparameters for this specific run
   # In a full run, this is typically iterated from a grid search
   hyperparameters = {
       'population_size': 50,
       'n_generations': 20,
       'mutation_rate': 0.2,
       'crossover_rate': 0.8,
   }

   # 4. Configure and run the experiment pipeline
   ml_grid_object = ml_grid.pipeline.data.pipe(
       input_csv_path=input_csv_path,
       base_project_dir=base_project_dir,
       local_param_dict=hyperparameters,
       config_dict={"modelFuncList": model_list},
   )

   # 5. Execute the Genetic Algorithm
   main_ga.run(ml_grid_object, local_param_dict=hyperparameters).execute()

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   Home
   docs_wiki/Usage
   docs_wiki/Project-Dataset-Requirements

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   data_preparation
   architecture
   project_structure
   usage
   configuration_guide
   example_notebook
   interpreting_results
   best_practices
   evaluating_models
   docs_wiki/Diagrams
   diagrams
   hyperparameter_reference
   troubleshooting
   conclusion

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   adding_new_learner
   docs_wiki/Genetic_Algorithm_Deep_Dive
   ga_deep_dive

.. toctree::
   :maxdepth: 2
   :caption: Project Information

   docs_wiki/Contributing
   contributing
   docs_wiki/FAQ
   docs_wiki/License
   license

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules