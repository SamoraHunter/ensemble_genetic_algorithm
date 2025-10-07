# Diagrams

This section contains visual representations of the genetic algorithm implementation and model architecture within the **Ensemble Genetic Algorithm** project.

## Data Pipeline and Genetic Algorithm

### `main.py` Command-Line Workflow

!main.py Workflow

-   **Source**: assets/main_py_workflow.mmd
-   **Description**: Illustrates the end-to-end workflow when running an experiment from the command line using `main.py`, including the main grid search loop and optional evaluation and plotting steps.

### GA Example Usage, Data Grid and GA Grid Permutations, System Flow (`example_usage.ipynb`)

!GA System Flow

-   **Source**: assets/example_usage_permutations.mmd
-   **Description**: Illustrates the genetic algorithm search over grid parameters, as demonstrated in the example usage notebook.

### GA Data Flow

!GA Data Flow

-   **Source**: assets/ga_data_diagram.mmd
-   **Description**: Illustrates the flow of data through the genetic algorithm pipeline, from input to ensemble generation.

### Model Class Structure

!Model Class Structure

-   **Source**: assets/model_classes.mmd
-   **Description**: Shows the inheritance hierarchy and relationships between the different model classes used in the project.

## Genetic Algorithm Components

### Weighting System

!GA Weighting System

-   **Source**: assets/ga_weighting.mmd
-   **Description**: Demonstrates the weighting mechanism applied to individual base learners within an ensemble.

### Parameter Space Grid

!Grid Parameter Space

-   **Source**: assets/grid_param_space_ga.mmd
-   **Description**: Visualizes how the genetic algorithm explores the parameter space, including feature and hyperparameter grids.

## Model Generation Workflows

### SVC Model Generation

!SVC Model Generation

-   **Source**: assets/svc_model_gen.mmd
-   **Description**: Flow diagram detailing the process for generating Support Vector Classifier (SVC) models as base learners.

### PyTorch Model Generation

!PyTorch Model Generation

-   **Source**: assets/torch_model_gen.mmd
-   **Description**: Flow diagram illustrating the generation process for PyTorch neural network models, including aspects of neural architecture search.

### XGBoost Model Generation

!XGBoost Model Generation

-   **Source**: assets/xgb_model_gen.mmd
-   **Description**: Flow diagram outlining the generation process for XGBoost models.

## Diagram Format

All diagrams are available in both Mermaid source format (`.mmd`) and rendered formats (`.png`/`.svg`). The Mermaid source files can be edited and re-rendered as needed for documentation updates, ensuring the diagrams remain current with the project's development.