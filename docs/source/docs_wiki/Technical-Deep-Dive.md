# Technical Deep Dive

This section provides an in-depth description of the theoretical background, methodology, and implementation details of the **Ensemble Genetic Algorithm** project.

---

## Table of Contents

- [Overview](#overview)
- [Theoretical Background](#theoretical-background)
- [Implementation Details](#implementation-details)
- [Performance Optimization](#performance-optimization)
- [Architecture Diagrams](#architecture-diagrams)

---

## Overview

This genetic algorithm is designed to evolve an optimal ensemble of machine learning classifiers for binary classification tasks. It applies a grid search over the feature space and genetic algorithm hyperparameters to find the best-performing model ensemble.

### Key Design Principles

1. **Modularity**: The framework separates data preprocessing, model generation, and evolutionary operators.
2. **Extensibility**: New base learners and weighting methods can be easily integrated.
3. **Efficiency**: Model caching and early stopping mechanisms reduce computational overhead.
4. **Robustness**: Comprehensive error handling and logging ensure reliable execution.

---

## Theoretical Background

### Machine Learning Optimization Framework

A classification problem can be expressed as learning a function f: X onto y. A binary classification problem is where y is binary, 0 or 1. Several learning algorithms have been developed to this end by researchers in mathematical statistics and more recently machine learning (Bishop and Nasrabadi, 2006) (Haykin, 1998).

Given a set of learning algorithms A and a limited dataset D = {(x1, y1), ..., (xn, yn)}, the objective of model selection is to identify the algorithm A∗ ∈ A that achieves optimal generalization performance. Generalization performance is assessed by splitting D into disjoint training and validation sets D(i)_train and D(i)_valid.

The model selection problem can be formally expressed as:

```
𝐴∗ = arg min (𝐴∈𝒜𝑘) ∑ 𝑖 𝐿(𝐴, 𝐷(𝑖)train, 𝐷(𝑖)valid)
```

where L(A, D(i)_train, D(i)_valid) represents the loss (e.g., misclassification rate) attained by A when trained on D(i)_train and assessed on D(i)_valid.

### Genetic Algorithm Fundamentals

A genetic algorithm is an optimization technique inspired by the process of natural selection. It's used in machine learning to find optimal solutions to complex problems by mimicking the evolutionary process:

1. **Population Initialization**: Starts with a population of potential solutions represented as individuals
2. **Selection**: Individuals are selected based on their fitness (ability to solve the problem)
3. **Crossover**: Selected individuals recombine to create offspring
4. **Mutation**: Offspring undergo random mutations
5. **Replacement**: New generation replaces the old population

The process repeats over multiple generations, tending to improve overall fitness.

---

## Implementation Details

### Core Components Architecture

```python
# Main pipeline flow
data.pipe() -> ml_grid_object -> main_ga.run().execute()
```

#### 1. Data Pipeline (`ml_grid.pipeline.data.pipe`)

Factory function that creates experiment objects for each grid search iteration:

- **Input**: Dataset path, hyperparameter set, global configuration
- **Process**:
  - Loads and validates dataset
  - Splits into train/validation/test sets
  - Applies feature selection and preprocessing
  - Creates transformed datasets for this iteration's configuration
- **Output**: `ml_grid_object` encapsulating all experiment settings

#### 2. Genetic Algorithm Runner (`main_ga.run`)

The primary engine for running the ensemble evolution. Key methods:

| Method | Description |
|--------|-------------|
| `__init__()` | Initializes DEAP toolbox, GA parameters, logging paths |
| `execute()` | Runs evolutionary loop for all GA hyperparameter combinations |
| `evaluate()` | Evaluates fitness of individual ensembles |

**Evolutionary Loop**:
```
Initialize population -> Evaluate fitness ->
Select parents -> Crossover -> Mutation ->
Evaluate offspring -> Replace population
```

#### 3. Ensemble Generation (`ensemble_generator_ga`)

Creates candidate ensembles by:

1. Randomly selecting base learners from the pool
2. Creating instances with randomized hyperparameters
3. Training on training data with selected feature subsets
4. Evaluating on validation data to get fitness score

#### 4. Weighting Methods

Three ensemble weighting strategies are available:

| Method | Description | Complexity | Performance |
|--------|-------------|------------|-------------|
| `unweighted` | Simple average of predictions | O(n) | Fast, good baseline |
| `de` | Differential Evolution finds optimal weights | O(n×iter) | Better accuracy |
| `ann` | Neural network learns non-linear combination | High | Potentially best |

### Key Data Structures

#### ml_grid_object

Container for a single grid search iteration:

- `X_train`, `y_train`: Training data
- `X_val`, `y_val`: Validation data (for GA fitness)
- `X_test`, `y_test`: Test data (for final evaluation)
- `model_class_list`: List of base learner generators
- `local_param_dict`: Current iteration's hyperparameters

#### Individual Representation

Each ensemble is represented as a chromosome:
```python
individual = [
    [algorithm, feature_subset, trained_model],
    [algorithm, feature_subset, trained_model],
    ...
]
```

See the [API Reference](./API-Reference.md) for comprehensive API documentation of all classes and methods.

---

## Performance Optimization

### Model Caching Strategy

**Benefits**:
- Avoids retraining when data preprocessing changes but models remain the same
- Can reduce runtime by >90% for subsequent runs with similar configurations

**Usage**:
1. First run: `store_base_learners=True` in grid_params
2. Subsequent runs: `use_stored_base_learners=True`

### Early Stopping

Implementation stops evolution when MCC score doesn't improve over 5 consecutive generation evaluations.

**Configuration** (in `config.yml`):
```yaml
global_params:
    gen_eval_score_threshold_early_stopping: 5  # Default
```

### Fitness Evaluation

Primary metric: Area Under ROC Curve (AUC)
Secondary metric: Matthews Correlation Coefficient (MCC) for model selection

### GPU Acceleration

PyTorch models automatically use available CUDA devices:

```python
# Check GPU availability
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

---

## Architecture Diagrams

### System Flow Diagram

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│   config.yml    │────>│   global_params     │────>│  Grid Search Loop│
└─────────────────┘     └─────────────────────┘     └────────┬─────────┘
                                                             │
                                             ┌──────────────▼──────────────┐
                                             │   data.pipe()                 │
                                             │   - Load dataset              │
                                             │   - Split data                │
                                             │   - Apply preprocessing       │
                                             └──────────────┬──────────────┘
                                                            │
                                            ┌──────────────▼──────────────┐
                                            │  ml_grid_object               │
                                            │  (per iteration)              │
                                            └──────────────┬──────────────┘
                                                           │
                                          ┌────────────────▼────────────────┐
                                          │   main_ga.run().execute()         │
                                          │   - DEAP initialization           │
                                          │   - Population generation         │
                                          │   - Evolutionary loop             │
                                          │   - Fitness evaluation            │
                                          └────────────────┬────────────────┘
                                                         │
                                      ┌────────────────────▼────────────────────┐
                                      │   Results Saving                          │
                                      │   - final_grid_score_log.csv              │
                                      │   - best_models.pkl                       │
                                      │   - plots/                                │
                                      └───────────────────────────────────────────┘
```

### Genetic Algorithm Pipeline

```
Initialization (Pop size = 96)
    ↓
Evaluation (AUC on validation set)
    ↓
Selection (Tournament, size=3)
    ↓
Crossover (2-point, prob=0.8)
    ↓
Mutation (gene swap, prob=0.2)
    ↓
Replacement (generational)
    ↓
Early Stopping Check? ──No──> Continue? ──Yes──> Next Generation
     ↓ Yes                          ↓ No
Terminate with Best Solution
```

### Weighting Method Comparison

| Approach | Equation | Pros | Cons |
|----------|----------|------|------|
| Unweighted | ŷ = mean([ŷ₁, ..., ŷₙ]) | Simple, fast | Ignores model quality differences |
| Differential Evolution | ŷ = Σ(wᵢ × ŷᵢ), w=DE-optimal | Accuracy improvement | Slower (~10×) |
| ANN Weighting | ŷ = ANN([ŷ₁, ..., ŷₙ]) | Captures non-linear interactions | Most complex, may overfit |

---

## Performance Benchmarks

### Baseline Performance (Typical Dataset)

| Population Size | Generations | Runtime (min) | Best AUC | Models Evaluated |
|-----------------|-------------|---------------|----------|------------------|
| 32 | 50 | ~15 | 0.78 | 1,600 |
| 64 | 100 | ~60 | 0.82 | 6,400 |
| 128 | 128 | ~240 | 0.85 | 16,384 |

### Optimization Effects

- **Model Caching**: Reduces runtime by ~90% for similar config runs
- **Early Stopping**: Saves ~20-40 generations on average
- **GPU Acceleration**: 3-5× speedup for PyTorch models

---

## Troubleshooting Implementation Issues

### Common issues with `main_ga.run`:

1. **Memory Errors**: Reduce population size or disable model caching
2. **Slow Performance**: Use simpler weighting method, reduce generations
3. **Poor Convergence**: Increase mutation rate or population diversity

See {doc}`Troubleshooting` for detailed solutions.

---

## References

0. Agius, R., Brieghel, C., Andersen, M.A., Pearson, A.T., Ledergerber, B., Cozzi-Lepri, A., Louzoun, Y., Andersen, C.L., Bergstedt, J., von Stemann, J.H., Jorgensen, M., Tang, M.E., Fontes, M., Bahlo, J., Herling, C.D., Hallek, M., Lundgren, J., MacPherson, C.R., Larsen, J. and Niemann, C.U. (2020) 'Machine learning can identify newly diagnosed patients with CLL at high risk of infection', *Nature communications*, 11(1), pp. 363-8. doi: 10.1038/s41467-019-14225-8

1. Bishop, C.M. and Nasrabadi, N.M. (2006) *Pattern recognition and machine learning*, Springer.

2. Haykin, S. (1998) *Neural networks: a comprehensive foundation*, Prentice Hall PTR.

3. Komer, B., Bergstra, J., and Eliasmith, C. (2014) *Hyperopt-sklearn: automatic hyperparameter configuration for scikit-learn*, Citeseer Austin, TX, pp. 50.

4. Thornton, C., Hutter, F., Hoos, H.H., and Leyton-Brown, K. (2013) *Auto-WEKA: Combined selection and hyperparameter optimization of classification algorithms*, pp. 847.

5. Nickolls, J., Buck, I., Garland, M., and Skadron, K. (2008) *Scalable parallel programming with cuda: Is cuda the parallel programming model that application developers have been waiting for?*, *Queue*, 6(2), pp. 40-53.

6. Fortin, F., De Rainville, F., Gardner, M.G., Parizeau, M., and Gagné, C. (2012) *DEAP: Evolutionary algorithms made easy*, *The Journal of Machine Learning Research*, 13(1), pp. 2171-2175.

7. Virtanen, P., Gommers, R., Oliphant, T.E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., and Bright, J. (2020) *SciPy 1.0: fundamental algorithms for scientific computing in Python*, *Nature methods*, 17(3), pp. 261-272.
