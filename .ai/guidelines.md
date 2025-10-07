## Development Guidelines

### Coding Style & Naming Conventions
- **Style**: Follow PEP 8 guidelines. Use 4-space indentation.
- **Naming**: Use `snake_case` for files, functions, and variables (e.g., `main_ga.py`, `evaluate_ensemble`). Use `PascalCase` for classes (e.g., `EnsembleEvaluator`, `GlobalParameters`).
- **Types & Docstrings**: Use type hints for all public APIs. Write clear **Google-style** docstrings for all public classes and functions, as this is what the Sphinx documentation build process expects.
- **Formatting**: The project uses `pre-commit` for formatting. Keep diffs small and consistent with the surrounding code.

### Testing Guidelines
- **Framework**: The project uses `pytest`.
- **Location**: Tests are located in the `tests/` directory.
- **Execution**: Run tests from the root directory with the command: `pytest`.
- **Focus**: Prefer small, deterministic unit tests, especially for utility functions, data processing steps, and configuration management.

### Commit & Pull Request Guidelines
- **Commits**: Use Conventional Commit style (e.g., `feat:`, `fix:`, `docs:`). Write clear, scoped messages.
- **Pull Requests**: Include a summary of changes, the rationale, and link any relevant issues. For changes affecting the workflow, provide example commands (e.g., `python main.py --config ...`).
- **Documentation**: If you change behavior, add a feature, or modify a parameter, update the relevant files in the `docs/source/docs_wiki/` directory.
- **Building Docs**: To build the documentation locally, navigate to the `docs/` directory and run `make html`. This uses Sphinx to generate the HTML site.

### Security & Configuration Tips
- **Large Files**: Do not commit datasets, trained models, or large log files. The `HFE_GA_experiments/` directory is ignored by default. Use external storage for large artifacts.
- **Credentials**: Keep any API keys, tokens, or other secrets out of the repository. Use environment variables or a local, untracked configuration file if needed.
- **CUDA/GPU**: For GPU support, it is recommended to install PyTorch manually following the official instructions at pytorch.org to match your system's CUDA version. The `setup.sh` script also provides a `--gpu` option.