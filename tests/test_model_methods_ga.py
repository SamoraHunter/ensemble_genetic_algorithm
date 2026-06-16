"""Tests for model_methods_ga module."""

import json
from unittest.mock import MagicMock

import numpy as np


def test_store_model_sklearn(tmp_path):
    """Test store_model with sklearn model type using tolist() workaround."""
    from ml_grid.util import model_methods_ga

    log_folder = tmp_path / "logs"
    log_folder.mkdir()

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.verbose = 0
    mock_ml_grid_object.logging_paths_obj.model_store_path = str(
        tmp_path / "model_store.json"
    )
    mock_ml_grid_object.logging_paths_obj.log_folder_path = str(log_folder)

    model_store_path = mock_ml_grid_object.logging_paths_obj.model_store_path
    with open(model_store_path, "w") as f:
        json.dump({"models": {}}, f)

    local_param_dict = {"scale": False}
    mccscore = 0.85
    model = "test_sklearn_model"
    feature_list = ["feature1", "feature2"]
    model_train_time = 10
    auc_score = 0.9
    y_pred = np.array([0, 1, 0, 1], dtype=float)
    model_type = "sklearn"

    model_methods_ga.store_model(
        mock_ml_grid_object,
        local_param_dict,
        mccscore,
        model,
        feature_list,
        model_train_time,
        auc_score,
        y_pred,
        model_type,
    )

    with open(model_store_path, "r") as f:
        stored_data = json.load(f)

    assert len(stored_data["models"]) == 1
    stored_model = stored_data["models"]["1"]
    assert stored_model["mcc_score"] == mccscore
    assert stored_model["model"] == str(model)
    assert stored_model["feature_list"] == feature_list
    assert stored_model["model_train_time"] == model_train_time
    assert stored_model["auc_score"] == auc_score
    assert stored_model["y_pred"] == list(y_pred)
    assert stored_model["model_type"] == model_type


def test_store_model_with_scale(tmp_path):
    """Test store_model handles scale=True by converting y_pred to float."""
    from ml_grid.util import model_methods_ga

    log_folder = tmp_path / "logs"
    log_folder.mkdir()

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.verbose = 0
    mock_ml_grid_object.logging_paths_obj.model_store_path = str(
        tmp_path / "model_store.json"
    )
    mock_ml_grid_object.logging_paths_obj.log_folder_path = str(log_folder)

    model_store_path = mock_ml_grid_object.logging_paths_obj.model_store_path
    with open(model_store_path, "w") as f:
        json.dump({"models": {}}, f)

    local_param_dict = {"scale": True}
    mccscore = 0.85
    model = "test_model"
    feature_list = ["feature1"]
    model_train_time = 5
    auc_score = 0.9
    y_pred = np.array([0, 1], dtype=float)
    model_type = "sklearn"

    model_methods_ga.store_model(
        mock_ml_grid_object,
        local_param_dict,
        mccscore,
        model,
        feature_list,
        model_train_time,
        auc_score,
        y_pred,
        model_type,
    )

    with open(model_store_path, "r") as f:
        stored_data = json.load(f)

    stored_model = stored_data["models"]["1"]
    assert isinstance(stored_model["y_pred"][0], float)


def test_store_model_verbose_logging(tmp_path, caplog):
    """Test store_model logs at verbose >= 11."""
    from ml_grid.util import model_methods_ga

    log_folder = tmp_path / "logs"
    log_folder.mkdir()

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.verbose = 11
    mock_ml_grid_object.logging_paths_obj.model_store_path = str(
        tmp_path / "model_store.json"
    )
    mock_ml_grid_object.logging_paths_obj.log_folder_path = str(log_folder)

    model_store_path = mock_ml_grid_object.logging_paths_obj.model_store_path
    with open(model_store_path, "w") as f:
        json.dump({"models": {}}, f)

    model_methods_ga.store_model(
        mock_ml_grid_object,
        {"scale": False},
        0.8,
        "model",
        ["feat"],
        10,
        0.9,
        np.array([0, 1], dtype=float),
        "sklearn",
    )

    assert "store_model" in caplog.text


def test_store_model_verbose_path(tmp_path, caplog):
    """Test store_model performs info logging at verbose >= 1."""
    from ml_grid.util import model_methods_ga

    log_folder = tmp_path / "logs"
    log_folder.mkdir()

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.verbose = 1
    mock_ml_grid_object.logging_paths_obj.model_store_path = str(
        tmp_path / "model_store.json"
    )
    mock_ml_grid_object.logging_paths_obj.log_folder_path = str(log_folder)

    model_store_path = mock_ml_grid_object.logging_paths_obj.model_store_path
    with open(model_store_path, "w") as f:
        json.dump({"models": {}}, f)

    model_methods_ga.store_model(
        mock_ml_grid_object,
        {"scale": False},
        0.8,
        "model",
        ["feat"],
        10,
        0.9,
        np.array([0, 1], dtype=float),
        "sklearn",
    )

    log_text = caplog.text
    assert "model_store_path:" in log_text
    assert "log_folder_path:" in log_text


def test_get_stored_model_sklearn(tmp_path):
    """Test get_stored_model with sklearn model type."""
    from ml_grid.util import model_methods_ga

    log_folder = tmp_path / "logs"
    log_folder.mkdir()

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.verbose = 0
    mock_ml_grid_object.logging_paths_obj.model_store_path = str(
        tmp_path / "model_store.json"
    )
    mock_ml_grid_object.logging_paths_obj.log_folder_path = str(log_folder)
    mock_ml_grid_object.config_dict = MagicMock()
    mock_ml_grid_object.config_dict.modelFuncList = lambda x, y: (
        0.5,
        None,
        [],
        0,
        0.5,
        np.array([]),
    )

    model_store_path = mock_ml_grid_object.logging_paths_obj.model_store_path
    with open(model_store_path, "w") as f:
        json.dump(
            {
                "models": {
                    "1": {
                        "mcc_score": 0.85,
                        "model": str({"sklearn_model": True}),
                        "feature_list": ["feature1", "feature2"],
                        "model_train_time": 10,
                        "auc_score": 0.9,
                        "y_pred": [0, 1, 0, 1],
                        "model_type": "sklearn",
                    }
                }
            },
            f,
        )

    model = model_methods_ga.get_stored_model(mock_ml_grid_object)

    assert len(model) == 6
    mccscore, model_obj, feature_list, train_time, auc, y_pred_arr = model
    assert mccscore == 0.85
    assert feature_list == ["feature1", "feature2"]
    assert train_time == 10
    assert auc == 0.9
    assert np.array_equal(y_pred_arr, np.array([0, 1, 0, 1]))


def test_get_stored_model_fallback_on_error(tmp_path):
    """Test get_stored_model falls back to random model on exception."""
    from ml_grid.util import model_methods_ga

    log_folder = tmp_path / "logs"
    log_folder.mkdir()

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.verbose = 0
    mock_ml_grid_object.logging_paths_obj.model_store_path = str(
        tmp_path / "model_store.json"
    )
    mock_ml_grid_object.logging_paths_obj.log_folder_path = str(log_folder)

    fallback_mock = MagicMock()
    fallback_mock.return_value = (
        0.7,
        "fallback_model",
        ["feat"],
        20,
        0.8,
        np.array([1, 0]),
    )
    mock_ml_grid_object.config_dict = MagicMock()
    mock_ml_grid_object.config_dict.modelFuncList = [fallback_mock]

    model_store_path = mock_ml_grid_object.logging_paths_obj.model_store_path
    with open(model_store_path, "w") as f:
        json.dump({"models": {}}, f)

    result = model_methods_ga.get_stored_model(mock_ml_grid_object)

    assert len(result) == 6
    mccscore, model_obj, feature_list, train_time, auc, y_pred_arr = result
    assert mccscore == 0.7


def test_get_stored_model_exception_logs_error(tmp_path, caplog):
    """Test get_stored_model logs error on exception with one element list."""
    import logging

    from ml_grid.util import model_methods_ga

    log_folder = tmp_path / "logs"
    log_folder.mkdir()

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.verbose = 0
    mock_ml_grid_object.logging_paths_obj.model_store_path = str(
        tmp_path / "model_store.json"
    )
    mock_ml_grid_object.logging_paths_obj.log_folder_path = str(log_folder)
    mock_ml_grid_object.config_dict = MagicMock()

    def mock_model_generator(*args, **kwargs):
        return (0.5, None, [], 0, 0.5, np.array([]))

    mock_ml_grid_object.config_dict.modelFuncList = [mock_model_generator]

    model_store_path = mock_ml_grid_object.logging_paths_obj.model_store_path
    with open(model_store_path, "w") as f:
        json.dump(
            {
                "models": {
                    "1": {"model_type": "invalid"}  # Invalid type to trigger exception
                }
            },
            f,
        )

    with caplog.at_level(logging.ERROR):
        result = model_methods_ga.get_stored_model(mock_ml_grid_object)

    assert "Failed inside getting stored model" in caplog.text
    assert len(result) == 6


def test_get_stored_model_returns_random_model_on_key_error(tmp_path):
    """Test get_stored_model falls back when model key is invalid."""
    from ml_grid.util import model_methods_ga

    log_folder = tmp_path / "logs"
    log_folder.mkdir()

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.verbose = 0
    mock_ml_grid_object.logging_paths_obj.model_store_path = str(
        tmp_path / "model_store.json"
    )
    mock_ml_grid_object.logging_paths_obj.log_folder_path = str(log_folder)

    fallback_mock = MagicMock()
    fallback_mock.return_value = (0.9, "fallback", ["x"], 5, 0.95, np.array([1]))
    mock_ml_grid_object.config_dict = MagicMock()
    mock_ml_grid_object.config_dict.modelFuncList = [fallback_mock]

    model_store_path = mock_ml_grid_object.logging_paths_obj.model_store_path
    with open(model_store_path, "w") as f:
        json.dump({"models": {"1": {}}}, f)

    result = model_methods_ga.get_stored_model(mock_ml_grid_object)

    assert len(result) == 6


def test_store_model_torch_y_pred_conversion(tmp_path):
    """Test that torch model conversion works in store_model."""
    from unittest import mock

    from ml_grid.util import model_methods_ga

    log_folder = tmp_path / "logs" / "torch"
    log_folder.mkdir(parents=True)

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.verbose = 0
    mock_ml_grid_object.logging_paths_obj.model_store_path = str(
        tmp_path / "model_store.json"
    )
    mock_ml_grid_object.logging_paths_obj.log_folder_path = str(log_folder.parent)

    model_store_path = mock_ml_grid_object.logging_paths_obj.model_store_path
    with open(model_store_path, "w") as f:
        json.dump({"models": {}}, f)

    local_param_dict = {"scale": False}
    mccscore = 0.92
    feature_list = ["feature1"]
    model_train_time = 15
    auc_score = 0.95
    y_pred_orig = np.array([0, 1])
    model_type = "torch"

    # Mock torch to avoid pickling issues
    with mock.patch("ml_grid.util.model_methods_ga.torch") as mock_torch:
        mock_model = MagicMock()
        mock_torch.save.return_value = None

        def mock_time_ns():
            return 1234567890

        with mock.patch("time.time_ns", side_effect=mock_time_ns):
            model_methods_ga.store_model(
                mock_ml_grid_object,
                local_param_dict,
                mccscore,
                mock_model,
                feature_list,
                model_train_time,
                auc_score,
                y_pred_orig,
                model_type,
            )

        # Verify torch.save was called with correct args
        assert mock_torch.save.called

    # Verify the timestamp was stored
    with open(model_store_path, "r") as f:
        stored_data = json.load(f)

    stored_model = stored_data["models"]["1"]
    assert stored_model["model_type"] == model_type
    assert stored_model["model"] == 1234567890


def test_get_stored_model_empty_store_fallback(tmp_path):
    """Test get_stored_model with empty models dict falls back."""
    from ml_grid.util import model_methods_ga

    log_folder = tmp_path / "logs"
    log_folder.mkdir()

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.verbose = 0
    mock_ml_grid_object.logging_paths_obj.model_store_path = str(
        tmp_path / "model_store.json"
    )
    mock_ml_grid_object.logging_paths_obj.log_folder_path = str(log_folder)

    fallback_mock = MagicMock()
    fallback_mock.return_value = (0.9, "fallback", ["x"], 5, 0.95, np.array([1]))
    mock_ml_grid_object.config_dict = MagicMock()
    mock_ml_grid_object.config_dict.modelFuncList = [fallback_mock]

    model_store_path = mock_ml_grid_object.logging_paths_obj.model_store_path
    with open(model_store_path, "w") as f:
        json.dump({"models": {}}, f)

    result = model_methods_ga.get_stored_model(mock_ml_grid_object)

    assert len(result) == 6
    assert result[0] == 0.9


def test_store_model_exception_torch_empty_cache(tmp_path, caplog):
    """Test store_model exception handling for torch.cuda.empty_cache."""

    from unittest import mock

    from ml_grid.util import model_methods_ga

    log_folder = tmp_path / "logs"
    log_folder.mkdir()

    mock_ml_grid_object = MagicMock()
    mock_ml_grid_object.verbose = 0
    mock_ml_grid_object.logging_paths_obj.model_store_path = str(
        tmp_path / "model_store.json"
    )
    mock_ml_grid_object.logging_paths_obj.log_folder_path = str(log_folder)

    model_store_path = mock_ml_grid_object.logging_paths_obj.model_store_path
    with open(model_store_path, "w") as f:
        json.dump({"models": {}}, f)

    # Mock torch to raise an exception
    with mock.patch("ml_grid.util.model_methods_ga.torch") as mock_torch:
        mock_torch.cuda.empty_cache.side_effect = RuntimeError("CUDA error")

        model_methods_ga.store_model(
            mock_ml_grid_object,
            {"scale": False},
            0.85,
            "test_model",
            ["feat"],
            10,
            0.9,
            np.array([0, 1], dtype=float),
            "sklearn",
        )

    assert "Failed to torch empty cache" in caplog.text
