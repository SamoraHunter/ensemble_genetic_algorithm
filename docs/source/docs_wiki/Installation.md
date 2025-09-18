# Installation Guide

This guide provides detailed instructions on how to set up the **Ensemble Genetic Algorithm** project. You can choose between a manual installation or using the provided setup script.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

-   **Python**: Version 3.10 or higher.
-   **Git**: For cloning the repository.
-   **(Optional) NVIDIA GPU with CUDA**: If you plan to use GPU-accelerated computations, ensure you have a compatible NVIDIA GPU and the CUDA Toolkit installed.

## Manual Installation

If you prefer to set up the environment manually, follow these steps. This project uses `pyproject.toml` to manage its dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SamoraHunter/ensemble_genetic_algorithm.git
    cd ensemble_genetic_algorithm
    ```

2.  **Create and activate a virtual environment:**
    It is highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Install the project and its core dependencies using pip. The dependencies are defined in `pyproject.toml`.
    ```bash
    pip install .
    ```
    To install with optional dependencies for development (e.g., testing tools, documentation tools), use:
    ```bash
    pip install .[dev]
    ```
    If you need GPU support and have the necessary CUDA setup, you would typically install `torch` with CUDA support manually or via the `setup.sh --gpu` option. The `pyproject.toml` currently lists `torch==2.0.1` as a core dependency, which might default to a CPU version depending on your `pip` configuration.

## Using the Setup Script (Alternative)

The project includes a comprehensive `setup.sh` script that automates the creation of a dedicated Python virtual environment and installs all necessary dependencies, including specific PyTorch versions for CPU or GPU.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SamoraHunter/ensemble_genetic_algorithm.git
    cd ensemble_genetic_algorithm
    ```

2.  **Run the setup script:**
    Make the script executable and then run it.
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```
    This will create a virtual environment named `ga_env`, install the default dependencies from `pyproject.toml`, and set up a Jupyter kernel. The environment will be activated for your current terminal session.

### Installation Options with `setup.sh`

The `setup.sh` script supports different installation profiles. You can specify one using command-line flags:

-   `./setup.sh --cpu`: Installs the CPU-only version of PyTorch. Ideal for systems without a dedicated GPU.
-   `./setup.sh --gpu`: Installs dependencies with GPU support (requires a compatible NVIDIA GPU and CUDA toolkit). This option will attempt to install the CUDA-enabled PyTorch.
-   `./setup.sh --dev`: Installs all development dependencies, including tools for testing, linting, and documentation.
-   `./setup.sh --all`: Installs everything, combining GPU and development dependencies.
-   `./setup.sh --force`: Forces the recreation of the virtual environment if it already exists.

To see all available options, run:
```bash
./setup.sh --help
```

## Activating the Environment

The `setup.sh` script activates the `ga_env` environment for your current terminal session. For future sessions, or if you installed manually into `.venv`, you can activate it manually:

```bash
source ga_env/bin/activate # If using setup.sh
source .venv/bin/activate   # If using manual installation with .venv
```