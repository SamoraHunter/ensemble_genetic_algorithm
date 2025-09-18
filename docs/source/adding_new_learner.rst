Adding a New Base Learner
=========================

The **Ensemble Genetic Algorithm** framework is designed to be extensible, allowing you to easily add new ``scikit-learn``-compatible classifiers to the pool of base learners. This guide walks you through the process.

The core idea is to create a "Model Generator" class for your new algorithm. This class is responsible for defining the model's hyperparameters and instantiating it.

Step 1: Create a Model Generator File
-------------------------------------

Create a new Python file inside the ``ml_grid/model_classes_ga/`` directory. It's best to follow the existing naming convention. For example, if you are adding a ``MyNewClassifier``, you could name the file ``myNewClassifier_model.py``.

Step 2: Implement the Model Generator Class
-------------------------------------------

Inside your new file, create a class that will generate your model. This class needs to follow a specific pattern. Here is a template you can adapt for any ``scikit-learn`` classifier.

.. code-block:: python
   :caption: ml_grid/model_classes_ga/myNewClassifier_model.py

   from sklearn.ensemble import AdaBoostClassifier
   from sklearn.tree import DecisionTreeClassifier

   class MyNewClassifierModelGenerator:
       """
       A generator class for creating instances of MyNewClassifier.
       """
       def __init__(self, ml_grid_object, local_param_dict):
           """
           Initializes the model generator.

           Args:
               ml_grid_object: The main ml_grid object with data and paths.
               local_param_dict: A dictionary of hyperparameters for the current run.
           """
           self.ml_grid_object = ml_grid_object
           self.local_param_dict = local_param_dict

       def model_gen(self):
           """
           Generates and returns an instance of the model.

           This method should define the model and its specific hyperparameters.
           """
           # Example: An AdaBoostClassifier with a Decision Tree base estimator
           model = AdaBoostClassifier(
               base_estimator=DecisionTreeClassifier(max_depth=2),
               n_estimators=100,
               learning_rate=0.5,
               random_state=self.ml_grid_object.global_params.seed
           )

           # The method must return a scikit-learn compatible model instance
           return model

Step 3: Add the New Learner to Your Experiment
----------------------------------------------

Now that you have created the generator, you need to tell your experiment script to use it.

1.  **Import the new generator** in your main experiment script (e.g., ``notebooks/example_usage.ipynb``).

2.  **Add the generator class** to the ``modelFuncList``. The genetic algorithm will now be able to select your new model when building ensembles.

That's it! Your new classifier is now part of the evolutionary search space.