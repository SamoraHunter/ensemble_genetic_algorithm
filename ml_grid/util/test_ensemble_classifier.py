import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml_grid.util.ensemble_classifier import SklearnEnsembleClassifier


def test_sklearn_ensemble_mask_with_invalid_features():
    """
    Test that ensemble fitting gracefully handles masks with invalid feature names.

    Regression test for issue where GA evolved ensembles could not be post-hoc
    evaluated due to feature name mismatches between training and evaluation phase.
    """
    # Setup mock data with subset of features
    X = pd.DataFrame(
        {
            "feat_1": [1, 2, 3, 4, 5, 6],
            "feat_2": [6, 5, 4, 3, 2, 1],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0])

    # Mask contains feature names that don't exist in current dataset
    invalid_mask_ensemble = [
        (1.0, LogisticRegression(), ["feat_1", "nonexistent_feature"]),  # partial match
    ]

    clf = SklearnEnsembleClassifier(invalid_mask_ensemble, ["feat_1", "feat_2"])
    clf.fit(X, y)

    assert len(clf.fitted_models) == 1
    assert clf.fitted_models[0][1] == ["feat_1"]  # Only valid features used


def test_sklearn_ensemble_mask_with_out_of_bounds_indices():
    """
    Test that ensemble fitting gracefully handles masks with out-of-bounds indices.

    Regression test for issue where GA evolved ensembles could not be post-hoc
    evaluated due to index mismatches between training and evaluation phase.
    """
    X = pd.DataFrame(
        {
            "feat_1": [1, 2, 3, 4, 5, 6],
            "feat_2": [6, 5, 4, 3, 2, 1],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0])

    # Mask contains binary masks and out-of-bounds index
    out_of_bounds_mask_ensemble = [
        (1.0, LogisticRegression(), [1, 0]),  # Binary mask: include feat_1 only
        (
            1.0,
            LogisticRegression(),
            [0, 5],
        ),  # Mixed: index 5 is out of bounds for 2 features
    ]

    clf = SklearnEnsembleClassifier(out_of_bounds_mask_ensemble, ["feat_1", "feat_2"])
    clf.fit(X, y)

    # Both models fitted since they have at least one valid feature/index each
    assert len(clf.fitted_models) == 2
    assert clf.fitted_models[0][1] == ["feat_1"]  # Binary mask [1,0]
    assert clf.fitted_models[1][1] == [
        "feat_1"
    ]  # Index mask [0,5] filters to valid indices


def test_sklearn_ensemble_mask_filters_all_features():
    """
    Test that ensemble fitting gracefully handles masks where ALL features are filtered out.

    This is a regression test for the specific error scenario:
    'ValueError: No base learners in the ensemble could be successfully fitted.'
    """
    X = pd.DataFrame(
        {
            "feat_1": [1, 2, 3, 4, 5, 6],
            "feat_2": [6, 5, 4, 3, 2, 1],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0])

    # All masks filter out all features
    ensemble_arch = [
        (1.0, LogisticRegression(), ["nonexistent"]),
        (1.0, LogisticRegression(), [99]),  # Out of bounds index
        (1.0, LogisticRegression(), [0, 0]),  # Binary mask with no active features
    ]

    clf = SklearnEnsembleClassifier(ensemble_arch, ["feat_1", "feat_2"])

    # Should raise ValueError since all models get excluded
    error_pattern = r"No base learners in the ensemble could be successfully fitted"
    with pytest.raises(ValueError, match=error_pattern):
        clf.fit(X, y)


def test_sklearn_ensemble_classifier_lifecycle():
    """
    Test the fit, predict, and predict_proba lifecycle of SklearnEnsembleClassifier.
    """
    # 1. Setup mock data
    X = pd.DataFrame(
        {
            "feat_1": [1, 2, 3, 4, 5, 6],
            "feat_2": [6, 5, 4, 3, 2, 1],
            "feat_3": [1, 0, 1, 0, 1, 0],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0])

    # 2. Setup mock ensemble architecture (weight, model, mask)
    # Using mixed mask types (names vs binary vs indices) to test robustness
    feature_names = ["feat_1", "feat_2", "feat_3"]

    ensemble_arch = [
        (1.0, LogisticRegression(), ["feat_1", "feat_2"]),  # names
        (1.0, LogisticRegression(), [1, 0, 1]),  # binary mask (feat_1, feat_3)
        (1.0, LogisticRegression(), [1, 2]),  # indices (feat_2, feat_3)
    ]

    # 3. Initialize and fit
    clf = SklearnEnsembleClassifier(ensemble_arch, feature_names)
    clf.fit(X, y)

    # 4. Test predictions
    preds = clf.predict(X)
    assert len(preds) == 6
    assert np.all(np.isin(preds, [0, 1]))

    probs = clf.predict_proba(X)
    assert len(probs) == 6
    assert np.all((probs >= 0) & (probs <= 1))


def test_ensemble_with_invalid_model_strings():
    """
    Test that ensemble fitting handles model strings that can't be evaluated.

    Regression test for issue where best_ensemble CSV contains corrupted
    model_string representations. When ALL models fail, raises ValueError with details.
    """
    # Setup mock data
    X = pd.DataFrame(
        {
            "feat_1": [1, 2, 3, 4, 5, 6],
            "feat_2": [6, 5, 4, 3, 2, 1],
        }
    )
    y = pd.Series([1, 0, 1, 0, 1, 0])

    # All models have invalid strings (not actual model objects)
    ensemble_arch = [
        (1.0, "NonExistentModelClass()", ["feat_1", "feat_2"]),
        (1.0, "AnotherBadModel()", ["feat_1"]),  # This should fail silently
    ]

    clf = SklearnEnsembleClassifier(ensemble_arch, ["feat_1", "feat_2"])

    # Should raise ValueError since all models fail to parse/fit
    with pytest.raises(ValueError, match="No base learners"):
        clf.fit(X, y)
