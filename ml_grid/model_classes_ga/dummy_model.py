import random
import numpy as np
from typing import Any, Dict, List, Tuple
from ml_grid.model_classes_ga.perceptron_dummy_model import perceptronModelGen_dummy
from sklearn.linear_model import Perceptron
import logging

logger = logging.getLogger("ensemble_ga")


class DummyModelGenerator:
    """A generator for creating a baseline 'dummy' model result.

    This class serves as a placeholder or baseline for the model generation
    process within the genetic algorithm. Instead of training a new model, it
    leverages a pre-trained simple Perceptron model and generates random
    predictions to establish a performance floor.

    This is useful for debugging the ensemble pipeline and ensuring that evolved
    ensembles perform better than random chance.

    Warning:
        This may throw a column name related error if included in the model
        pipeline. It is not intended to be used in a production setting.

    Attributes:
        fitted_perceptron (Perceptron): A pre-trained scikit-learn Perceptron model.
        dummy_columns (List[str]): A list of feature names corresponding to the
            `fitted_perceptron`.
    """

    def __init__(self, ml_grid_object: Any, local_param_dict: Dict):
        """Initializes the DummyModelGenerator.

        Args:
            ml_grid_object (Any): An object containing project data and configurations.
            local_param_dict (Dict): A dictionary of local parameters for the run.
        """
        (
            _,
            self.fitted_perceptron,
            self.dummy_columns,
            _,
            _,
            _,
        ) = perceptronModelGen_dummy(ml_grid_object, local_param_dict)

    def dummy_model_gen(
        self, ml_grid_object: Any, local_param_dict: Dict
    ) -> Tuple[float, Perceptron, List[str], int, float, np.ndarray]:
        """Generates a tuple representing a dummy model's evaluation.

        This method returns a standard model result tuple but with fixed or random
        values. The model itself is a pre-fitted Perceptron, and the predictions
        are random.

        Args:
            ml_grid_object (Any): An object containing project data (e.g., y_test).
            local_param_dict (Dict): A dictionary of local parameters (unused).

        Returns:
            A tuple containing the following elements:
                - mccscore (float): A fixed MCC score of 0.5.
                - model (Perceptron): The pre-fitted Perceptron model.
                - feature_names (List[str]): The list of feature names for the model.
                - model_train_time (int): A fixed training time of 0.
                - auc_score (float): A random ROC AUC score between 0.5 and 1.0.
                - y_pred (np.ndarray): A vector of random binary predictions.
        """
        y_test = ml_grid_object.y_test
        mccscore = 0.5
        model_train_time = 0
        auc_score = random.uniform(0.5, 1)
        y_pred = np.random.choice(a=[False, True], size=len(y_test))
        return (
            mccscore,
            self.fitted_perceptron,
            self.dummy_columns,
            model_train_time,
            auc_score,
            y_pred,
        )


# # Example usage:
# # Create an instance of DummyModelGenerator
# dummy_generator = DummyModelGenerator()

# # Call the dummy_model_gen method
# result = dummy_generator.dummy_model_gen(ml_grid_object, local_param_dict)  # Make sure ml_grid_object and local_param_dict are defined
# logger.info(result)
