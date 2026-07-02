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
        # Store original feature names from the ensemble for proper mask decoding
        # This is needed when X_train has different column order than training data
        self.original_feature_names_used = (
            feature_names.copy()
            if isinstance(feature_names, list)
            else list(feature_names)
        )
        self.fitted_models = []
        self._all_req_features = None

    @property
    def all_req_features(self):
        """Exposes the union of all features required by the base learners."""
        # Safely handle models loaded from disk without the internal attribute
        val = getattr(self, "_all_req_features", None)
        if val is None and self.fitted_models:
            val = sorted(list(set().union(*(set(m[1]) for m in self.fitted_models))))
            self._all_req_features = val
        return val

    @all_req_features.setter
    def all_req_features(self, value):
        self._all_req_features = value

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_models = []
        all_req_features_set = set()
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
                # Mask already contains feature names - filter to valid features only
                active_features = [f for f in mask if f in self.feature_names]
            elif not all(
                isinstance(x, (int, np.integer)) and x in [0, 1] for x in mask
            ):
                # If mask is NOT binary, assume it contains indices
                # Filter to valid indices only
                active_features = [
                    self.feature_names[i]
                    for i in mask
                    if 0 <= i < len(self.feature_names)
                ]
            else:
                # Translate binary/int mask to names
                # Handle both same-length and different-length masks
                if len(mask) == len(self.feature_names):
                    # Exact match: use enumerate with mask positions
                    active_features = [
                        self.feature_names[i] for i, val in enumerate(mask) if val == 1
                    ]
                else:
                    # Length mismatch: mask was created from a different feature set.
                    # The mask is binary (0/1) with original positions. We need to
                    # only select features that exist in self.feature_names.
                    active_features = []
                    for i, val in enumerate(mask):
                        if int(val) == 1 and i < len(self.feature_names):
                            active_features.append(self.feature_names[i])

            # Check if we have any valid features before attempting fit
            if not active_features:
                warnings.warn(
                    f"Base learner {type(model).__name__} has no valid features and will be excluded. "
                    f"Original mask: {mask}, Valid feature_names: {self.feature_names}"
                )
                continue

            try:
                if not isinstance(model, BinaryClassification):
                    # Verify active_features actually exist in X
                    missing_features = set(active_features) - set(X.columns)
                    if missing_features:
                        warnings.warn(
                            f"Base learner {type(model).__name__} has missing features: "
                            f"{missing_features}. Skipping."
                        )
                        continue
                    model.fit(X[active_features], y)
                self.fitted_models.append((model, active_features, weight))
                all_req_features_set.update(active_features)
            except Exception as e:
                warnings.warn(
                    f"Base learner {type(model).__name__} failed to fit and will be excluded. "
                    f"Error: {e}"
                )

        # Check for empty ensemble before trying to build skip report
        original_ensemble_length = len(self.ensemble_arch)
        if original_ensemble_length == 0:
            raise ValueError(
                "No base learners in the ensemble could be successfully fitted. "
                "The ensemble architecture is empty - no models were provided for fitting. "
                "This typically happens when: (1) best_ensemble CSV data was corrupted or missing, "
                "(2) GA experiment failed to complete, or (3) feature masks are invalid. "
                "Check the run logs for errors like 'No features found' or 'outcome_var_X not in columns'."
            )

        if not self.fitted_models:
            # Build a helpful error message with all skipped models
            skipped = []

            # Iterate over original count to capture details even after filtering
            for i in range(original_ensemble_length):
                try:
                    model_tuple = self.ensemble_arch[i]
                    mask = model_tuple[2] if len(model_tuple) > 2 else None
                    skipped.append(f"{type(model_tuple[1]).__name__}: mask={mask}")
                except (IndexError, TypeError):
                    # Handle malformed tuples gracefully
                    skipped.append(f"Model_{i}: malformed tuple")

            raise ValueError(
                f"No base learners in the ensemble could be successfully fitted. "
                f"All {original_ensemble_length} models were skipped due to feature mismatches or fitting errors. "
                f"Skipped models: {'; '.join(skipped) if skipped else 'Details unavailable'}"
            )

        self.all_req_features = sorted(list(all_req_features_set))
        # Set standard sklearn attribute for broader compatibility
        self.feature_names_in_ = np.array(self.all_req_features)

        return self

    def _check_X(self, X):
        req = self.all_req_features
        if req is None:
            return
        missing = [f for f in req if f not in X.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}")

    def predict(self, X):
        self._check_X(X)
        
        # Check if fit was called before predict
        if not self.fitted_models:
            raise ValueError(
                "This ensemble has not been fitted yet. "
                "Call .fit(X_train, y_train) before calling .predict()."
            )
        
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

        # Handle edge case where weights sum to zero or very close to zero (normalize if needed)
        weights = np.array(weights, dtype=float)
        
        weight_sum = np.sum(weights)
        
        # Use equal weights if all weights are zero or empty
        if np.isclose(weight_sum, 0) or len(weights) == 0:
            if len(weights) > 0:
                weights = np.ones(len(weights)) / len(weights)

        result = np.average(all_preds, axis=0, weights=weights)
        return (
            np.round(result).astype(int).ravel()
        )

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
        weights = np.array(weights)
        if np.isclose(np.sum(weights), 0):
            # Use equal weights if all weights are zero
            weights = np.ones(len(weights)) / len(weights)

        p1 = np.average(all_probs, axis=0, weights=weights).ravel()
        return np.vstack([1 - p1, p1]).T
