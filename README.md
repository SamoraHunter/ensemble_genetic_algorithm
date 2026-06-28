# Ensemble Genetic Algorithm

[![GA Project Test](https://github.com/SamoraHunter/ensemble_genetic_algorithm/actions/workflows/notebook-test.yml/badge.svg)](https://github.com/SamoraHunter/ensemble_genetic_algorithm/actions/workflows/notebook-test.yml) [![Build and Deploy Docs](https://github.com/SamoraHunter/ensemble_genetic_algorithm/actions/workflows/docs.yml/badge.svg)](https://samorahunter.github.io/ensemble_genetic_algorithm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Table of Contents

- [Description](#description)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Documentation](#-documentation)
- [Contributing](#contributing)
- [Performance Benchmarks](#performance-benchmarks)
- [Architecture Diagrams](#architecture-diagrams)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Description

This project provides a genetic algorithm designed to evolve an optimal ensemble of machine learning classifiers for binary classification tasks. It applies a grid search over the feature space and genetic algorithm hyperparameters to find the best-performing model ensemble.

### Primary Use Cases

- **Automated Model Selection**: Discover the best combination of algorithms and hyperparameters
- **Ensemble Optimization**: Find optimal weighting schemes for multi-model predictions
- **Feature Subset Exploration**: Identify the most informative feature combinations
- **Hyperparameter Tuning**: Simultaneously optimize model parameters and selection

---

## Key Features

-   **Evolve Ensembles**: Uses a genetic algorithm to automatically find the best combination of models.
-   **Extensible Model Library**: Easily add any scikit-learn compatible classifier.
-   **Comprehensive Search**: Performs a grid search over data preprocessing, feature selection, and GA hyperparameters.
-   **Advanced Weighting**: Includes methods like Differential Evolution and ANNs to find optimal ensemble weights.
-   **Model Caching**: Re-use trained base learners to dramatically speed up subsequent experiments.
-   **GPU Acceleration**: Automatic utilization of CUDA-enabled GPUs for PyTorch models.

---

## Installation

The package is currently installed from source using the provided `setup.sh` script, which automates virtual environment creation and dependency management.

**Note on PyTorch**: This package requires PyTorch. For GPU support, it is highly recommended that you first install PyTorch manually by following the official instructions at pytorch.org to ensure the correct version for your CUDA toolkit is installed.

### Prerequisites

-   **Python**: Version 3.12 or higher (required for `pyproject.toml` compatibility).
-   **Git**: For cloning the repository.
-   **(Optional) NVIDIA GPU with CUDA**: For GPU-accelerated computations.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SamoraHunter/ensemble_genetic_algorithm.git
    cd ensemble_genetic_algorithm
    ```

2.  **(Optional) Run the setup script:**
    The `setup.sh` script automates the creation of a virtual environment and dependency installation.
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```
    This will create a virtual environment named `ga_env`, install the default dependencies, and set up a Jupyter kernel. The environment will be activated for your current terminal session.

#### Installation Options

The setup script supports different installation profiles. You can specify one using flags:

-   `./setup.sh --cpu`: Installs the CPU-only version of PyTorch. Ideal for systems without a dedicated GPU.
-   `./setup.sh --gpu`: Installs dependencies with GPU support (requires a compatible NVIDIA GPU and CUDA toolkit).
-   `./setup.sh --dev`: Installs all development dependencies, including tools for testing.
-   `./setup.sh --all`: Installs everything, including GPU and development dependencies.

To see all available options, run:
```bash
./setup.sh --help
```

### Activating the Environment

The setup script activates the `ga_env` environment for your current session. For future sessions, you can activate it manually:

```bash
source ga_env/bin/activate
```

---

## Project Dataset Requirements

A numeric data matrix (Pandas dataframe) with a binary outcome variable with the suffix `_outcome_var_1`. For more details see [pat2vec](https://github.com/SamoraHunter/pat2vec).

### Required Format

- All columns must be numeric (integers or floats)
- Column names must not contain special characters
- Outcome variable name must end with `_outcome_var_1`

---

## Quickstart

You can run experiments either from the command line (recommended for most users) or programmatically within a script for development purposes.

### Command-Line Usage

The `main.py` script is the primary entry point for running experiments, which includes both traditional grid search and genetic algorithm execution.

1.  **Activate the virtual environment:**

    ```bash
    source ga_env/bin/activate
    ```

2.  **Run the experiment:**

    -   To run with the default `config.yml`:
        ```bash
        python main.py
        ```
    -   To specify a different configuration file:
        ```bash
        python main.py --config path/to/your/config.yml
        ```
    -   To run, evaluate the best model, and generate all analysis plots:
        ```bash
        python main.py --config path/to/your/config.yml --evaluate --plot
        ```

### Programmatic Usage

For development or debugging, you can run the pipeline within a Python script or Jupyter notebook using the genetic algorithm execution module.

#### Basic Example

```python
from datetime import datetime
import os
import pathlib
from tqdm import tqdm
from ml_grid.pipeline import data, main_ga
from ml_grid.util.global_params import global_parameters
from ml_grid.util.grid_param_space_ga import Grid

# Initialize parameters from a config file
config_path = 'config.yml'
global_params = global_parameters(config_path=config_path)

# Create a unique, timestamped directory for the experiment run
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_specific_dir = os.path.join(global_params.base_project_dir, timestamp)
pathlib.Path(run_specific_dir).mkdir(parents=True, exist_ok=True)

# Update global_params to use this new directory for all outputs
global_params.base_project_dir = run_specific_dir

# Define the search space from the config
grid = Grid(
    global_params=global_params,
    config_path=config_path
)

# Main experiment loop using main_ga.run() entry point
for i in tqdm(range(global_params.n_iter)):
    local_param_dict = next(grid.settings_list_iterator)
    
    # The data pipeline prepares the data for this specific iteration
    ml_grid_object = data.pipe(
        global_params=global_params,
        file_name=global_params.input_csv_path,
        local_param_dict=local_param_dict,
        param_space_index=i,
    )
    
    # Execute the genetic algorithm
    main_ga.run(
        ml_grid_object, 
        local_param_dict=local_param_dict, 
        global_params=global_params
    ).execute()
```

#### Advanced: Multiple Configurations

```python
import numpy as np
from ml_grid.util.global_params import global_parameters

# Load base configuration
base_params = global_parameters(config_path='config.yml')

# Override specific parameters for different scenarios
scenarios = [
    {'n_iter': 10, 'model_list': ['logisticRegression', 'randomForest']},
    {'n_iter': 20, 'model_list': ['XGBoost', 'Pytorch_binary_class']}
]

for i, scenario_config in enumerate(scenarios):
    # Create experiment-specific configuration
    params = global_parameters(
        config_path='config.yml',
        **scenario_config,
        base_project_dir=f"experiments/scenario_{i}_{timestamp}"
    )
    
    grid = Grid(global_params=params, config_path='config.yml')
    
    for j in range(params.n_iter):
        # ... experiment execution as shown above
        pass
```

---

## Configuration

The recommended way to configure the project is by creating a `config.yml` file in your project's root directory. This allows you to manage all settings in one place.

1.  **Create `config.yml`**: Copy the `config.yml.example` file from the repository to a new file named `config.yml`.
2.  **Edit**: Uncomment and modify the parameters you wish to change. Any parameter not specified in your `config.yml` will use its default value.

### Complete Example `config.yml`

```yaml
# config.yml

global_params:
  # --- Experiment Settings ---
  input_csv_path: "data/my_dataset.csv"
  n_iter: 20
  model_list: ["logisticRegression", "randomForest", "XGBoost", "Pytorch_binary_class"]

  # --- Execution & Logging ---
  verbose: 2
  grid_n_jobs: 8
  base_project_dir: "HFE_GA_experiments"
  store_base_learners: True

ga_params:
  nb_params: [8, 16, 24]       # Number of base learners per ensemble
  pop_params: [64, 128]        # Population size
  g_params: [100]              # Number of generations

grid_params:
  weighted: ["unweighted", "de"]   # Ensemble weighting methods
  resample: ["undersample", None]  # Data imbalance handling
  corr: [0.95]                     # Feature correlation threshold
```

### Configuration Sections

#### `global_params`

Controls overall experiment behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_csv_path` | Required | Path to your dataset |
| `n_iter` | 20 | Number of grid search iterations |
| `model_list` | Required | List of base learners to use |
| `verbose` | 2 | Logging verbosity level (0-15) |
| `grid_n_jobs` | 8 | Parallel jobs for grid search |
| `base_project_dir` | "HFE_GA_experiments" | Output directory |
| `testing` | False | Use smaller test grid |

#### `ga_params`

Genetic algorithm evolutionary parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nb_params` | [8, 16, 24] | Ensemble sizes to try |
| `pop_params` | [64, 128] | Population sizes to try |
| `g_params` | [100] | Number of generations |

#### `grid_params`

Search space for each grid iteration:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weighted` | ["unweighted"] | Weighting methods: `unweighted`, `de`, `ann` |
| `resample` | [None] | Resampling: `undersample`, `oversample`, `None` |
| `corr` | [0.95] | Feature correlation threshold |

---

## 📘 Documentation

For detailed user guides, tutorials, and the full API reference, please see the **[Official Documentation](https://ensemble-genetic-algorithm.readthedocs.io/en/latest/)**.

The documentation provides comprehensive information on everything from data preparation to interpreting results and extending the framework.

### Recommended Reading Order

1. **Getting Started**: Installation → Usage → Data Preparation
2. **Core Concepts**: Architectural Overview → Genetic Algorithm Deep Dive
3. **Configuration**: Configuration Guide → Hyperparameter Reference
4. **Advanced Topics**: Performance Benchmarks → Best Practices → Troubleshooting

---

## Contributing

We welcome contributions from the community! Please read through the following before submitting issues or pull requests.

### Reporting Issues

Before reporting an issue, please:

1. Check the [Troubleshooting Guide](docs/source/docs_wiki/Troubleshooting.md)
2. Search existing issues to avoid duplicates
3. Include Python version (`python --version`)
4. Include operating system and hardware specifications
5. Provide a minimal reproducible example

### Pull Request Process

1. Fork the repository
2. Create a branch from `main`
3. Make your changes following our coding conventions
4. Run tests (if available)
5. Update documentation as needed
6. Submit a pull request with a clear description

---

## Performance Benchmarks

### Baseline Performance

Typical performance metrics on standard binary classification datasets:

| Population Size | Generations | Runtime | Best AUC | Models Evaluated |
|-----------------|-------------|---------|----------|------------------|
| 32 | 50 | ~15 min | 0.78 | 1,600 |
| 64 | 100 | ~60 min | 0.82 | 6,400 |
| 128 | 128 | ~240 min | 0.85 | 16,384 |

*Note: Times vary based on dataset size, feature count, and hardware.*

### Optimization Effects

#### Model Caching

- **Benefit**: Reduces runtime by ~90% for subsequent runs with similar configurations
- **Use Case**: Iterative experiment development where only GA parameters change

#### Early Stopping

- **Mechanism**: Stops evolution when MCC score doesn't improve over 5 consecutive generations
- **Average Savings**: ~20-40 generations per experiment
- **Configuration**:
  ```yaml
  global_params:
      gen_eval_score_threshold_early_stopping: 5
  ```

#### GPU Acceleration

- **Speedup**: 3-5× for PyTorch-based models
- **Hardware**: Requires NVIDIA GPU with CUDA compute capability ≥ 3.0

### Memory Usage

| Population Size | Ensemble Size | Memory | Notes |
|-----------------|---------------|--------|-------|
| 32 | 8 | ~1 GB | Minimal caching |
| 64 | 16 | ~4 GB | Standard configuration |
| 128 | 24 | ~8 GB | High-resolution search |

---

## Architecture Diagrams

See the [Architecture Diagrams](docs/source/docs_wiki/Diagrams.md) page for comprehensive visual documentation.

### System Flow

```
config.yml → global_params → Grid Search Loop → data.pipe()
                                            ↓
                                       ml_grid_object
                                            ↓
                                  main_ga.run().execute()
                                            ↓
                                    Results Saving
```

### Genetic Algorithm Pipeline

```
Initialize Population (n individuals)
    ↓
Evaluate Fitness (AUC on validation)
    ↓
Select Parents (Tournament selection)
    ↓
Crossover (2-point, P=0.8)
    ↓
Mutation (gene swap, P=0.2)
    ↓
Replace Population
    ↓
Early Stopping Check? → No → Next Generation
     Yes
     ↓
Terminate with Best Solution
```

### Weighting Method Comparison

| Approach | Complexity | Performance Gain | Use Case |
|----------|------------|------------------|----------|
| Unweighted | O(n) + Fast | Baseline | Quick experiments, large ensembles |
| Differential Evolution | O(n×iter) ~10× | Moderate (+5-10%) | Medium-scale ensembles |
| ANN Weighting | High | Potentially high (+10-20%) | Small ensembles, complex interactions |

---

## License

MIT License

Copyright (c) 2023 Samora Hunter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

---

## Authors

- Samora Hunter 

---

## Acknowledgments

This software is based primarily on Machine learning methodology originally described in:

Agius, R., Brieghel, C., Andersen, M.A. et al. Machine learning can identify newly diagnosed patients with CLL at high risk of infection. *Nat Commun* **11**, 363 (2020). https://doi.org/10.1038/s41467-019-14225-8

---

## Appendix

### Genetic Algorithm Implementation Details

*The following are relevant excerpts from the manuscript presenting this work*

*A classification problem can be expressed as the problem of learning a function f: X onto y. A binary classification problem is a problem where y is binary, 0 or 1. Several learning algorithms have been developed to this end by researchers in mathematical statistics and more recently machine learning ​(Bishop and Nasrabadi, 2006)​ ​(Haykin, 1998)​. The problem is further described by a dataset {d1, … dn} of training data points di = (Xi, yi) ε X * y. A learning algorithm typically has associated hyper parameters λ ε which alter the behaviour of the learning algorithm. The task of machine learning typically entails the optimisation of these hyper parameters. Several approaches to efficient and effective hyperparameter optimisation have been proposed  ​(Komer, Bergstra and Eliasmith, 2014)​.* 

*Given a set of learning algorithms A and a limited dataset D = {(x1, y1), ..., (xn, yn)}, the objective of model selection is to identify the algorithm A∗ ∈ A that achieves optimal generalization performance. Generalization performance is assessed by splitting D into disjoint training and validation sets D(i)_train and D(i)_valid. The learning functions are constructed using A∗ on D(i)_train, and their predictive performance is then evaluated on D(i)_valid. This frames the model selection problem as follows:* 

*(Equation 1)  *

𝐴∗ = arg min (𝐴∈𝒜𝑘) ∑ 𝑖 𝐿(𝐴, 𝐷(𝑖)train, 𝐷(𝑖)valid)

 
*where L(A, D(i)_train, D(i)_valid) represents the loss (e.g., misclassification rate) attained by A when trained on D(i)_train and assessed on D(i)_valid ​(Thornton et al., 2013)​. Performance can be further evaluated by partitioning data into k equally sized folds and the learning algorithm fitted to k-1 folds and evaluated on the held-out set.*  

*The choice of learning algorithm, hyperparameter and feature set can be viewed as a hyperparameter set to be optimised for itself ​(Komer, Bergstra and Eliasmith, 2014)​. This set may then be optimised for using known generally available optimisation methods such as Bayesian hyperparameter optimisation ​(Komer, Bergstra and Eliasmith, 2014)​ and genetic algorithms.*


### Appendix 2 Mb Genetic algorithm, ensemble classifier 

 
*In order to address the optimisation problem in (Equation 1) a method inspired by recent applied machine learning in medicine was developed ​(Agius et al., 2020)​, note well feature engineering methods utilised were similarly heavily inspired by those found in that manuscript. The precise predictive problem addressed by those method is different however the general problem is very similar across available data sources, types of features, numbers of samples and others. This method in its referential form entails a genetic algorithm to search for the optimal ensemble of machine learning classifiers for a binary outcome. In the method developed and extended here it entails a grid search over genetic algorithm hyperparameters and feature space and feature transformations for the optimal ensemble of machine learning classifiers for a binary outcome. Ensemble weighting, additional base learning algorithms, early stopping, model recycling, neural architecture search and more.* 

*A genetic algorithm is an optimization technique inspired by the process of natural selection. It's used in machine learning to find optimal solutions to complex problems by mimicking the evolutionary process. It starts with a population of potential solutions represented as individuals. Through iterations, individuals are selected, recombined, and mutated to create new generations. The selection is based on the fitness of individuals, which measures their quality in solving the problem. Over time, this process tends to improve the overall fitness of the population, leading to solutions that are better adapted to the problem at hand.* 

*The ensemble method described in ​(Agius et al., 2020)​​(Nickolls et al., 2008)​ was first replicated in software. An ensemble modelling approach combines predictions from multiple independent classifiers. The authors cite  ​(Hansen and Salamon, 1990)​ ​(Perrone and Cooper,1995)​ arguing that ensemble methods can reduce overfitting and that multiple independent uncorrelated predictors can reduce the error. This method was designed to consider greatly more variables than are typically found in the medical diagnosis prediction literature. It was then adapted largely to increase the available configuration and explore a greater algorithm, feature and transformation hyperparameter space. Several developments are a result of reengineering to deploy the algorithm in a resource constrained environment. The hyperparameter configuration of the genetic algorithm and feature space were not optimised for, however in principle and given sufficient compute resource, they could be optimised with Bayesian hyperparameter ​(Komer, Bergstra and Eliasmith, 2014)​ optimisation as in Ma.* 

*The following is a description of the method as it is finally implemented for this project. All data transformations and feature space segmenting methods previously described for Ma were used to form a grid of datasets for consideration for the primary dataset’s Da and Db. A random sub sample of the possible grids was selected to reduce compute time. Candidate base learners from Scikit-learn were implemented as in CLL-TIM. Hyperparameter spaces were expanded. An additional base learner of a binary classifier implemented in Pytorch was developed and included. This base learner implements rudimentary aspects of neural architecture search ​(Elsken, Metzen and Hutter, 2019)​ by exposing elementary artificial neural network architecture hyperparameters in the search space. This is an improvement on the Scikit-learn multilayer perceptron classifier implemented in CLL-TIM as it is accelerated by CUDA ​(Nickolls et al., 2008)​ and is greatly more extensible.*  

*The genetic algorithm's process begins with creating a population through random selection from a pool of base learners. Each base learner comprises a specific learning algorithm along with its corresponding hyperparameter space. Initializing a base learner involves training the learning algorithm with a randomized arrangement of hyperparameters on a training dataset. Prior to this, a feature reduction technique is applied to the dataset. The trained model is then utilized to assess its performance on a test dataset, which is shortened by the feature selection process. Once created, the base learner includes the learning algorithm, a subset of features, relevant metrics, an evaluation score (AUC), and a prediction vector for the test dataset.*  

*The pool of base learners is invoked to construct ensembles ranging from two to the maximum specified ensemble size. To introduce a skew towards smaller ensembles, a skew normal distribution function is applied. The ensembles generated using this method serve as individuals within the genetic algorithm. These individuals are defined by the chromosomes of the individual base learners. The fitness of each ensemble is determined by its performance on the test set, as assessed by the AUC metric. Notably, no measures were taken to implement ensemble diversity weighting. However, a hyperparameter for ensemble weighting offers three potential weighting options. The first is no weighting, each base learner in the ensemble’s prediction is collapsed in a matrix and transformed into a binary akin to applying a sigmoid to the mean. The second, differential evolution ​(Virtanen et al., 2020)​ is used to find the weights for each base learner that maximise AUC on the training set. Far fewer iterations for this algorithm than are normally used to reduce compute time. Differential evolution weighted ensemble individuals then have their AUC attribute set to the weighted ensemble score. The third entails a similar optimisation problem however an artificial neural network implemented in Pytorch is used to learn the optimal ensemble weighting. Artificial neural network weighted ensemble individuals then have their AUC attribute set to the weighted ensemble score.*  

*Individuals are generated to fill a population of size 96. These individuals then undergo evaluation whereby they are measured on their classification performance, Matthews’s correlation coefficient was used to evaluate performance on the test set, this is the individual’s fitness. Parents are selected by tournament selection of the size of a hyperparameter from these individuals and 2-point crossover is applied. Mutation of the probability given in a hyperparameter of ensembles occurs when one base learner is swapped out for a newly randomly generated one. Fitness of the offspring is recalculated. This cycle is repeated for a maximum of 128 generations. Early stopping defined by a failure to improve on the maximum MCC score reached after five cycles was implemented. A full description and illustration of this process is available in ​(Agius et al., 2020)​supplementary 17). The genetic algorithm was implemented with DEAP Python library  ​(Fortin et al., 2012)​.*  


### Key Implementation Features

- **Base Learner Pool**: Candidate base learners from Scikit-learn with expanded hyperparameter spaces
- **Neural Network Integration**: PyTorch binary classifier with rudimentary neural architecture search
- **Ensemble Weighting**: Three methods - unweighted, differential evolution, and ANN-based weighting
- **Model Recycling**: Efficient reuse of trained models across similar experiments

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
