# Diagrams

This section contains visual representations of the genetic algorithm implementation and model architecture within the **Ensemble Genetic Algorithm** project.

## Data Pipeline and Genetic Algorithm

### GA Example Usage, Data Grid and GA Grid Permutations, System Flow (`example_usage.ipynb`)

!GA Data Diagram

-   **Source**: assets/example_usage_permutations.mmd
-   **Description**: Illustrates the genetic algorithm search over grid parameters, as demonstrated in the example usage notebook.

### GA Data Flow

!GA Data Diagram

-   **Source**: assets/ga_data_diagram.mmd
-   **Description**: Illustrates the flow of data through the genetic algorithm pipeline, from input to ensemble generation.

### Model Class Structure

!Model Classes

-   **Source**: assets/model_classes.mmd
-   **Description**: Shows the inheritance hierarchy and relationships between the different model classes used in the project.

## Genetic Algorithm Components

### Weighting System

!GA Weighting

-   **Source**: assets/ga_weighting.mmd
-   **Description**: Demonstrates the weighting mechanism applied to individual base learners within an ensemble.

### Parameter Space Grid

!Grid Parameter Space GA

-   **Source**: assets/grid_param_space_ga.mmd
-   **Description**: Visualizes how the genetic algorithm explores the parameter space, including feature and hyperparameter grids.

## Model Generation Workflows

### SVC Model Generation

<img src="https://github.com/SamoraHunter/ensemble_genetic_algorithm/raw/main/assets/svc_model_gen.svg" width="100" />

-   **Source**: assets/svc_model_gen.mmd
-   **Description**: Flow diagram detailing the process for generating Support Vector Classifier (SVC) models as base learners.

### PyTorch Model Generation

<img src="https://github.com/SamoraHunter/ensemble_genetic_algorithm/raw/main/assets/torch_model_gen.svg" width="150" />

-   **Source**: assets/torch_model_gen.mmd
-   **Description**: Flow diagram illustrating the generation process for PyTorch neural network models, including aspects of neural architecture search.

### XGBoost Model Generation

<img src="https://github.com/SamoraHunter/ensemble_genetic_algorithm/raw/main/assets/xgb_model_gen.svg" width="200" />

-   **Source**: assets/xgb_model_gen.mmd
-   **Description**: Flow diagram outlining the generation process for XGBoost models.

## Diagram Format

All diagrams are available in both Mermaid source format (`.mmd`) and rendered formats (`.png`/`.svg`). The Mermaid source files can be edited and re-rendered as needed for documentation updates, ensuring the diagrams remain current with the project's development.