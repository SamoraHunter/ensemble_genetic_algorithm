import pytest
from ml_grid.util.global_params import global_parameters
from ml_grid.util.grid_param_space_ga import Grid
from ml_grid.util import config


@pytest.fixture(autouse=True)
def reset_config_flag():
    """Resets the internal flag in the config module before each test."""
    # This ensures that tests for logging messages run predictably.
    config._config_message_printed = False
    yield
    config._config_message_printed = False

def test_global_params_defaults():
    """Tests that global_parameters initializes with hardcoded defaults when no config is found."""
    params = global_parameters(config_path="non_existent_file.yml")
    assert params.verbose == 3
    assert params.grid_n_jobs == 4
    assert params.store_base_learners is False


def test_global_params_from_config_file(tmp_path):
    """Tests that global_parameters correctly loads values from a config file."""
    config_content = """
global_params:
  verbose: 5
  grid_n_jobs: 16
"""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)

    params = global_parameters(config_path=str(config_file))

    assert params.verbose == 5
    assert params.grid_n_jobs == 16
    # This parameter was not in the config, so it should retain its default value
    assert params.store_base_learners is False


def test_global_params_kwargs_override(tmp_path):
    """Tests that keyword arguments override both defaults and config file values."""
    config_content = """
global_params:
  verbose: 5
  grid_n_jobs: 16
"""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)

    # Override 'verbose' from config and 'store_base_learners' from default
    params = global_parameters(
        config_path=str(config_file), verbose=9, store_base_learners=False
    )

    assert params.verbose == 9  # Overridden by kwarg
    assert params.grid_n_jobs == 16  # From config file
    assert params.store_base_learners is False  # Overridden by kwarg


def test_grid_defaults():
    """Tests that the Grid class initializes with its default parameters."""
    grid = Grid(config_path="non_existent_file.yml")
    # Test a default ga_param
    assert grid.pop_params == [32, 64, 128]
    # Test a default grid_param
    assert grid.grid["weighted"] == ["ann", "de", "unweighted"]
    assert grid.grid["corr"] == [0.9, 0.99]


def test_grid_from_config_file(tmp_path):
    """Tests that the Grid class loads ga_params and grid_params from a config file."""
    config_content = """
ga_params:
  pop_params: [50, 100]

grid_params:
  weighted: ["unweighted"]
  corr: [0.95]
  data:
    - bloods: [True] # Override a nested value
"""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)

    grid = Grid(config_path=str(config_file))

    # Check ga_params override
    assert grid.pop_params == [50, 100]
    assert grid.g_params == [128]  # Should remain default

    # Check grid_params override (simple list)
    assert grid.grid["weighted"] == ["unweighted"]
    assert grid.grid["corr"] == [0.95]

    # Check that the recursive merge for the 'data' dictionary worked
    assert grid.grid["data"][0]["bloods"] == [True]  # Overridden
    assert grid.grid["data"][0]["age"] == [True]  # Should remain from default


def test_grid_test_grid_flag_override(tmp_path):
    """Tests that test_grid=True overrides config and defaults for GA params."""
    config_content = """
ga_params:
  pop_params: [50, 100]
  g_params: [200]
"""
    config_file = tmp_path / "config.yml"
    config_file.write_text(config_content)

    # The test_grid=True flag should have the final say on these specific ga_params
    grid = Grid(config_path=str(config_file), test_grid=True)

    assert grid.pop_params == [8]
    assert grid.g_params == [4]
    assert grid.nb_params == [4, 8]


def test_load_config_not_found(caplog):
    """Tests that load_config returns an empty dict and prints info if file not found."""
    config_data = config.load_config("non_existent_file.yml")
    assert config_data == {}
    assert "No 'config.yml' found" in caplog.text


def test_load_config_invalid_yaml(tmp_path, caplog):
    """Tests that load_config handles invalid YAML gracefully by returning an empty dict."""
    config_file = tmp_path / "invalid.yml"
    config_file.write_text("key: value: nested_value")  # Invalid YAML syntax

    config_data = config.load_config(str(config_file))

    assert config_data == {}
    assert "Could not parse YAML file" in caplog.text
    assert "invalid.yml" in caplog.text