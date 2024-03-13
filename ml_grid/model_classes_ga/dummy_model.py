import random
import numpy as np
from ml_grid.model_classes_ga.perceptron_dummy_model import perceptronModelGen_dummy


class DummyModelGenerator:
    def __init__(self, ml_grid_object, local_param_dict):
        self.fitted_perceptron = perceptronModelGen_dummy(
            ml_grid_object, local_param_dict
        )[1]
        self.dummy_columns = perceptronModelGen_dummy(ml_grid_object, local_param_dict)[
            2
        ]

    def dummy_model_gen(self, ml_grid_object, local_param_dict):
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
# print(result)
