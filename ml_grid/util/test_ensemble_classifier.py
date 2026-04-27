import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_grid.util.ensemble_classifier import SklearnEnsembleClassifier


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
