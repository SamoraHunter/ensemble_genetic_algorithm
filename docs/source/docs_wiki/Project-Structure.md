# Project Structure

This page provides an overview of the key files and directories within the **Ensemble Genetic Algorithm** project.

## Root Directory

-   `README.md`: The main project README, providing a high-level overview, installation instructions, and usage examples.
-   `main.py`: The primary command-line entry point for running experiments.
-   `config.yml.example`: An example configuration file that users can copy to `config.yml` to customize their experiments.
-   `CONTRIBUTING.md`: Guidelines for contributing to the project.
-   `LICENSE`: The project's license (MIT License).
-   `pyproject.toml`: Defines the project's build system, metadata, and dependencies. This is the central file for dependency management.
-   `setup.sh`: A comprehensive shell script to automate the setup of the development environment, including virtual environment creation and dependency installation.
-   `setup-hooks.sh`: (If present) A script for setting up Git hooks for development workflows.
-   `notebooks/`: Contains example Jupyter notebooks, such as `example_usage.ipynb`.
-   `assets/`: Stores diagrams and other static assets used in the documentation.

## `pyproject.toml`

This file is crucial for modern Python project management. It defines:

-   **Build System**: Specifies `setuptools` for building the package.
-   **Project Metadata**: Includes the project name (`ensemble-genetic-algorithm`), version, description, authors, license, and Python requirements (`>=3.10`).
-   **Dependencies**: Lists all core libraries required for the project to run (e.g., `deap`, `scikit-learn`, `torch`, `pandas`).
-   **Optional Dependencies**: Defines groups of additional dependencies for specific purposes:
    -   `dev`: For development, testing (`pytest`, `pre-commit`), and documentation (`sphinx`).
    -   `gpu`: For GPU-specific libraries (currently empty, but intended for future GPU-specific dependencies).
    -   `all`: Combines `dev` and `gpu` dependencies.
-   **Package Structure**: Explicitly tells `setuptools` that the source code for the main package is located in the `ml_grid` directory.

## `setup.sh`

This shell script streamlines the environment setup process. It handles:

-   **Python Version Check**: Ensures Python 3.10+ is installed.
-   **Virtual Environment Creation**: Creates a `ga_env` virtual environment.
-   **Dependency Installation**: Installs project dependencies using `pip install .` or `pip install -e .[option]` based on command-line arguments (`--cpu`, `--gpu`, `--dev`, `--all`).
-   **Jupyter Kernel Setup**: Registers the virtual environment as a Jupyter kernel.
-   **Verification**: Performs basic checks to confirm successful installation of key libraries.

## `ml_grid/`

This directory is expected to contain the main Python source code for the `ensemble-genetic-algorithm` package, as indicated by `tool.setuptools.packages = ["ml_grid"]` in `pyproject.toml`.