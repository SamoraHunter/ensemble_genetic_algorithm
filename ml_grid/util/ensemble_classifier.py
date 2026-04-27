import warnings
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from ml_grid.ga_functions.ga_ann_util import BinaryClassification


class SklearnEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible wrapper for ensembles evolved by the Genetic Algorithm.
    """

    def __init__(self, ensemble_arch, feature_names):
        self.ensemble_arch = ensemble_arch
        self.feature_names = feature_names
        self.fitted_models = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_models = []
        for model_tuple in self.ensemble_arch:
            # Use index-based access to handle variable tuple lengths safely
            # Standard format: (weight, model_object, mask, score, predictions, ...)
            weight = model_tuple[0]
            model = model_tuple[1]
            mask = model_tuple[2]

            # Handle mask (binary/int array or list of names)
            if (
                isinstance(mask, (list, tuple, np.ndarray))
                and len(mask) > 0
                and isinstance(mask[0], str)
            ):
                active_features = mask
            elif len(mask) != len(self.feature_names) or not all(isinstance(x, (int, np.integer)) and x in [0, 1] for x in mask):
                # Assume list of indices if length doesn't match feature space or contains non-binary values
                active_features = [self.feature_names[i] for i in mask]
            else:
                # Translate binary/int mask to names
                active_features = [
                    self.feature_names[i] for i, val in enumerate(mask) if val == 1
                ]

            try:
                if not isinstance(model, BinaryClassification):
                    model.fit(X[active_features], y)
                self.fitted_models.append((model, active_features, weight))
            except Exception as e:
                warnings.warn(
                    f"Base learner {type(model).__name__} failed to fit and will be excluded. "
                    f"Error: {e}"
                )

        if not self.fitted_models:
            raise ValueError("No base learners in the ensemble could be successfully fitted.")

        return self

    def _check_X(self, X):
        all_req_features = set().union(*(set(m[1]) for m in self.fitted_models))
        missing = [f for f in all_req_features if f not in X.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}")

    def predict(self, X):
        self._check_X(X)
        all_preds = []
        weights = []
        for model, features, weight in self.fitted_models:
            if isinstance(model, BinaryClassification):
                data = torch.FloatTensor(X[features].values)
                model.eval()
                with torch.no_grad():
                    p = torch.round(torch.sigmoid(model(data))).numpy().flatten()
                all_preds.append(p)
            else:
                # Ensure base learner predictions are flattened to 1D
                all_preds.append(np.asarray(model.predict(X[features])).ravel())
            weights.append(weight)
        return np.round(np.average(all_preds, axis=0, weights=weights)).astype(int).ravel()

    def predict_proba(self, X):
        self._check_X(X)
        all_probs = []
        weights = []
        for model, features, weight in self.fitted_models:
            if isinstance(model, BinaryClassification):
                data = torch.FloatTensor(X[features].values)
                model.eval()
                with torch.no_grad():
                    # Sigmoid output is the probability for class 1
                    p = torch.sigmoid(model(data)).numpy().flatten()
                all_probs.append(p)
            elif hasattr(model, "predict_proba"):
                # Extract probability for the positive class (class 1)
                p = np.asarray(model.predict_proba(X[features]))
                if p.ndim > 1 and p.shape[1] > 1:
                    p = p[:, 1]
                all_probs.append(p.ravel())
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X[features])
                # Transform decision function to probability and flatten
                probs = (1 / (1 + np.exp(-df))).ravel()
                all_probs.append(probs)
            else:
                # Fallback to labels flattened to 1D
                all_probs.append(np.asarray(model.predict(X[features])).ravel())
            weights.append(weight)

        # Use evolved weights for the final prediction probabilities
        p1 = np.average(all_probs, axis=0, weights=weights).ravel()
        return np.vstack([1 - p1, p1]).T
