# Home

Welcome to the **Ensemble Genetic Algorithm** project wiki!

This project provides a robust framework for evolving machine learning ensembles for binary classification problems using genetic algorithms. It's designed to be extensible, highly configurable, and flexible, allowing users to customize base learners, genetic algorithm hyperparameters, and feature space exploration.

## Key Features

-   **Genetic Algorithm Optimization**: Utilizes a genetic algorithm to search for optimal ensembles of machine learning classifiers.
-   **Extensible Base Learners**: Easily integrate various machine learning algorithms as base learners.
-   **Configurable Hyperparameters**: Fine-tune genetic algorithm parameters (e.g., population size, generations) and base learner configurations.
-   **Feature Space Exploration**: Supports grid search over feature spaces and feature transformations.
-   **GPU Acceleration**: Optional GPU support for PyTorch-based models.
-   **Comprehensive Setup**: Includes a `setup.sh` script for automated environment creation and dependency installation.

## Getting Started

To get started with the project, please refer to the {doc}`Installation`.

## Documentation

Explore the following wiki pages for detailed information:

-   **Getting Started**
    -   {doc}`Installation`: How to set up your development environment.
    -   {doc}`Usage`: How to run experiments using `main.py` and `config.yml`.
    -   {doc}`Data_Preparation_Guide`: The required format for your input data.
-   **Core Concepts**
    -   {doc}`Architectural_Overview`: A high-level look at the project's components.
    -   {doc}`Genetic_Algorithm_Deep_Dive`: An explanation of the evolutionary process.
    -   {doc}`Configuration_Guide`: A detailed guide to the `config.yml` file.
    -   {doc}`Hyperparameter_Reference`: A reference for all configurable parameters.
-   **Guides & Tutorials**
    -   {doc}`Interpreting_Results`: How to understand the plots and outputs.
    -   {doc}`Evaluating_Final_Models`: How to validate your final models on unseen data.
    -   {doc}`Adding_a_New_Base_Learner`: How to extend the project with new models.
    -   {doc}`Best_Practices`: Tips for running experiments effectively.
-   **Reference**
    -   {doc}`Troubleshooting`: Solutions for common errors.
    -   {doc}`Project-Structure`: An overview of the repository's file layout.

## License

This project is licensed under the MIT License.

---