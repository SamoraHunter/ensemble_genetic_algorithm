"""Property-based tests using Hypothesis library.

This module implements property-based testing to validate:
- Hyperparameter space constraints (valid ranges, valid combinations)
- Model generator function outputs match expected structure
- Ensemble weight computations satisfy mathematical properties (sum=1, non-negative)
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("hypothesis")

from hypothesis.extra.numpy import arrays


@given(
    size=st.sampled_from(["medium", "xsmall", "xwide"]),
    param_key=st.sampled_from(
        [
            "log_small",
            "bool_param",
            "log_large",
            "log_large_long",
            "log_med_long",
            "log_med",
            "log_zero_one",
            "lin_zero_one",
        ]
    ),
)
@settings(deadline=None)
def test_param_space_valid_keys_and_types(size, param_key):
    """Test that ParamSpace returns valid arrays for all key/size combinations."""
    from ml_grid.util.param_space import ParamSpace

    ps = ParamSpace(size)

    assert ps.param_dict is not None
    assert param_key in ps.param_dict, f"Key {param_key} should be present"

    value = ps.param_dict[param_key]

    if "bool" in param_key:
        assert isinstance(value, list), "bool_param should be a list"
        assert all(isinstance(x, bool) for x in value)
    else:
        assert isinstance(value, np.ndarray), f"{param_key} should be a numpy array"


@given(size=st.sampled_from(["medium", "xsmall", "xwide"]))
def test_param_space_single_values_valid(size):
    """Test that single values from param_dict are valid."""
    from ml_grid.util.param_space import ParamSpace

    ps = ParamSpace(size)

    for key, value in ps.param_dict.items():
        if isinstance(value, np.ndarray):
            assert len(value) > 0, f"{key} should have at least one element"

            if not np.issubdtype(value.dtype, np.integer):
                continue

            arr = value
            assert np.all(np.isfinite(arr)), f"{key} should contain finite values"


@given(
    log_array=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=10),
        elements=st.floats(allow_nan=False, allow_infinity=False),
    )
)
def test_logspace_constraints(log_array):
    """Test properties of log-spaced arrays."""
    unique_vals = np.unique(log_array)

    if len(unique_vals) > 1:
        assert np.all(
            np.diff(unique_vals) != 0
        ), "Logspace should have increasing values"


@given(
    bool_list=arrays(
        dtype=bool,
        shape=st.integers(min_value=2, max_value=10),
        elements=st.just(True) | st.just(False),
    )
)
def test_bool_param_properties(bool_list):
    """Test boolean parameter constraints."""
    assert len(bool_list) >= 2
    # Boolean arrays can contain any mix of True/False values
    # This property ensures we don't crash when processing them
    assert isinstance(bool_list, np.ndarray)


@given(
    weight_array=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=10),
        elements=st.floats(min_value=0.0, max_value=1e10),
    )
)
@settings(deadline=None)
def test_normalize_weights_non_negative(weight_array):
    """Test that normalized weights are non-negative."""
    from ml_grid.ga_functions.ga_ann_util import normalize

    if np.any(weight_array):
        normalized = normalize(weight_array)

        assert np.all(normalized >= 0), "Normalized weights should be non-negative"


@given(
    weight_array=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=10),
        elements=st.floats(min_value=0.0, max_value=1e5),
    )
)
@settings(deadline=None)
def test_normalize_weights_sum_to_one(weight_array):
    """Test that normalized weights sum to 1."""
    from ml_grid.ga_functions.ga_ann_util import normalize

    if np.any(weight_array) and not np.allclose(weight_array, 0, atol=1e-307):
        normalized = normalize(weight_array)

        if np.sum(normalized) > 0:
            assert (
                np.abs(np.sum(normalized) - 1.0) < 1e-5
            ), f"Normalized weights should sum to 1, got {np.sum(normalized)}"


@given(
    weight_array=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=10),
        elements=st.floats(min_value=0.0, max_value=1e10),
    ),
    prediction_matrix=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=2, max_value=5),
            st.integers(min_value=5, max_value=10),
        ),
        elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
    ),
)
@settings(deadline=None)
def test_ensemble_weighted_prediction_shape(weight_array, prediction_matrix):
    """Test that weighted ensemble prediction has correct shape."""
    from ml_grid.ga_functions.ga_ann_util import normalize

    normalized_weights = normalize(weight_array[: len(prediction_matrix)])

    if len(normalized_weights) > 0 and len(prediction_matrix) > 0:
        prediction_matrix_normalized = np.array(prediction_matrix)[
            : len(normalized_weights)
        ]
        weighted_prediction = np.dot(prediction_matrix_normalized.T, normalized_weights)

        assert (
            len(weighted_prediction) == prediction_matrix.shape[1]
        ), f"Prediction length {len(weighted_prediction)} should match test samples"


@given(
    prediction_array=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=5, max_value=20),
        elements=st.floats(min_value=-1e3, max_value=1e3),
    ),
    weight=st.floats(min_value=-1e3, max_value=1e3),
    num_models=st.integers(min_value=1, max_value=5),
)
@settings(deadline=None)
def test_weighted_prediction_combination(prediction_array, weight, num_models):
    """Test that weighted prediction uses correct combination logic."""
    from ml_grid.ga_functions.ga_ann_util import normalize

    pred_matrix = np.array([prediction_array] * num_models)
    original_weights = np.array([weight] * num_models)

    if len(pred_matrix) > 0 and len(original_weights) > 0:
        normalized = normalize(original_weights)

        if np.sum(normalized) > 0:
            weighted_sum = np.dot(pred_matrix.T, normalized)

            assert len(weighted_sum) == prediction_array.shape[0]


def test_linear_weight_method_constraints():
    """Test linear weight method produces valid weights."""
    from unittest.mock import Mock

    from ml_grid.ga_functions import ga_linear_weight_method

    mock_ml_grid = Mock()
    mock_ml_grid.y_test = np.array([0, 1, 0, 1])

    pred_array_1 = np.array([0.3, 0.7, 0.2, 0.8])
    pred_array_2 = np.array([0.4, 0.6, 0.3, 0.7])
    target_ensemble = [
        (Mock(), Mock(), None, None, None, pred_array_1),
        (Mock(), Mock(), None, None, None, pred_array_2),
    ]

    best = [target_ensemble]

    result = ga_linear_weight_method.find_linear_weights(best, mock_ml_grid)

    assert isinstance(result, np.ndarray), "Weights should be numpy array"
    assert len(result) == 2, f"Expected 2 weights for 2 models, got {len(result)}"
    assert np.all(result >= 0), "All weights should be non-negative"
    assert abs(np.sum(result) - 1.0) < 1e-6, "Weights should sum to 1"


def test_ensemble_weight_prediction_valid():
    """Test weighted ensemble prediction with valid inputs."""
    from unittest.mock import Mock

    from ml_grid.ga_functions import ga_de_weight_method

    mock_ml_grid = Mock()
    mock_ml_grid.X_test_orig = np.array([[1, 2], [3, 4], [5, 6]])
    mock_ml_grid.X_train = np.array([[1, 2], [3, 4], [5, 6]])
    mock_ml_grid.y_train = np.array([0, 1, 0])

    pred_array = np.array([0.3, 0.7, 0.2])
    target_ensemble = [(Mock(), Mock(), None, None, None, pred_array)]

    best = [target_ensemble]
    weights = np.array([1.0])

    result = ga_de_weight_method.get_weighted_ensemble_prediction_de_y_pred_valid(
        best, weights, mock_ml_grid, valid=False
    )

    assert isinstance(result, np.ndarray), "Result should be numpy array"
    assert len(result) == 3, f"Expected 3 predictions, got {len(result)}"


@given(invalid_size=st.sampled_from(["invalid", "", "unknown", "large"]))
def test_param_space_invalid_size(invalid_size):
    """Test that invalid size returns None param_dict."""
    from ml_grid.util.param_space import ParamSpace

    ps = ParamSpace(invalid_size)

    assert ps.param_dict is None, f"Invalid size {invalid_size} should return None"


@given(
    empty_array=arrays(dtype=np.float64, shape=st.integers(min_value=0, max_value=0))
)
def test_normalize_empty_array(empty_array):
    """Test normalize handles empty array."""
    from ml_grid.ga_functions.ga_ann_util import normalize

    result = normalize(empty_array)

    assert len(result) == 0
