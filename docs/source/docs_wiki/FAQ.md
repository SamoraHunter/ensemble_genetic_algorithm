# FAQ / User Guide

This section addresses common questions and provides guidance for using the **Ensemble Genetic Algorithm** project.

---

### How do I run and convert the example usage notebook using the command line only (for example on a HPC...)?

To execute and convert the example Jupyter notebook, use the following command from the **root** of the repository:

```bash
jupyter nbconvert --to notebook --execute notebooks/example_usage.ipynb --output notebooks/executed_example_usage.ipynb
```

This command will:

-   Run the notebook `example_usage.ipynb` using the current Python environment.
-   Save the executed version as `executed_example_usage.ipynb` in the same `notebooks/` directory.
-   Preserve the interactive IPython functionality (e.g., display, widgets, etc.) during execution.

📌 **Note**: Make sure the `ga_env` (or `.venv`) environment is activated before running this command:

```bash
source ga_env/bin/activate # Or .venv/bin/activate if installed manually
```
This ensures all required dependencies are available for successful execution.