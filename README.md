# Ensemble Genetic Algorithm

**Description**: Genetic algorithm to evolve a machine learning ensemble for binary classification.

Prerequisites:
Develop a data matrix d for a binary classification problem. See [pat2vec](https://github.com/SamoraHunter/pat2vec).

Applies a grid search over featurespace and genetic algorithm hyperparameters to optimise for an ensemble of machine learning classifiers on a binary classification problem. 

-Extensible,  
  -select base learners for inclusion
-Highly configurable
  -Configure base learners 
-Flexible
  -Framework can be extended for regression or multiclass classification

## Notebook Version

**Version**: 19.4

### Dependencies

- **deap**: 1.3
- **fuzzysearch**: 0.7.3
- **genetic_selection**: 0.5.1
- **graphviz**: 0.20
- **imblearn**: 0.9.0
- **matplotlib**: 3.5.2
- **numpy**: 1.21.6
- **pandas**: 1.3.5
- **pydot**: 1.4.2
- **pydotplus**: NA
- **pylab**: NA
- **scipy**: 1.7.3
- **scoop**: 0.7
- **seaborn**: 0.11.2
- **session_info**: 1.0.0
- **sklearn**: 1.0.2
- **torch**: 1.12.1+cu102
- **torchmetrics**: 0.7.3
- **tqdm**: 4.64.0
- **xgboost**: 1.4.2

### Additional Jupyter Dependencies

- **IPython**: 7.34.0
- **jupyter_client**: 7.3.3
- **jupyter_core**: 4.10.0
- **notebook**: 6.4.12

## Environment Information

- **Python**: 3.7.6 (default, Jan 8, 2020, 19:59:22) [GCC 7.3.0]
- **Operating System**: Linux-5.4.0-125-generic-x86_64-with-debian-buster-sid

## Session Information

- Session information updated at 2023-07-17 16:06

## Usage

To use this project, follow the steps below:

1. **Set Paths for Input Data**: Ensure you have the necessary input data and set the paths accordingly. Refer to the unit test synthetic data for an example of the feature column naming convention.

2. **Configure Feature Space Exploration**:
   - Determine which parameters to include in the feature space exploration by setting the `grid` parameter to `True` or `False`.

3. **Configure Learning Algorithm Inclusion**:
   - Customize the list of learning algorithms you want to include by populating the `modelFuncList` variable.

4. **Configure Genetic Algorithm Hyperparameters**:
   - Adjust the genetic algorithm's hyperparameters as needed:
     - Maximum Individual Size: Set the value for `nb_params`.
     - Population Size: Set the value for `pop_params`.
     - Maximum Generation Number: Set the value for `g_params`.

Example Configuration:

```python
# Example Configuration
grid = {
    # Set feature space exploration parameters
    "parameter1": True,
    "parameter2": False,
    # Add more parameters as needed
}

modelFuncList = [
    # Add learning algorithms to include in the experiment
    "Algorithm1",
    "Algorithm2",
    # Add more algorithms as needed
]

# Genetic algorithm hyperparameters
nb_params = 100  # Maximum individual size
pop_params = 50  # Population size
g_params = 10    # Maximum generation number

```

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


## Contributing

If you want to contribute to this project, please follow our [contributing guidelines](CONTRIBUTING.md).

## Authors

- Samora Hunter 

## Acknowledgments

This software is based primarily on Machine learning methodology originally described in:

Agius, R., Brieghel, C., Andersen, M.A. et al. Machine learning can identify newly diagnosed patients with CLL at high risk of infection. Nat Commun 11, 363 (2020). https://doi.org/10.1038/s41467-019-14225-8

## Requirements.txt

You can install the required packages by running:

```bash
pip install -r requirements.txt

deap==1.3
fuzzysearch==0.7.3
genetic_selection==0.5.1
graphviz==0.20
imblearn==0.9.0
matplotlib==3.5.2
numpy==1.21.6
pandas==1.3.5
pydot==1.4.2
# pydotplus is not available
# pylab is not available
scipy==1.7.3
scoop==0.7
seaborn==0.11.2
session_info==1.0.0
sklearn==1.0.2
torch==1.12.1+cu102
torchmetrics==0.7.3
tqdm==4.64.0
xgboost==1.4.2

