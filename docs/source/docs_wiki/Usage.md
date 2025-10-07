# Usage Guide

This guide explains the primary ways to run experiments using the **Ensemble Genetic Algorithm** project.

---

## Recommended Workflow: Command-Line with `config.yml`

The most straightforward and recommended way to run an experiment is from your terminal using the `main.py` script and a `config.yml` file. This approach keeps your configuration separate from the code and is ideal for most use cases.

1.  **Prepare Your Data**: Ensure your input CSV meets the requirements outlined in the {doc}`Data_Preparation_Guide`.

2.  **Create a Configuration File**: Copy the `config.yml.example` file in the project root to a new file named `config.yml`.

3.  **Edit `config.yml`**: Open your `config.yml` and customize the experiment. At a minimum, you should set:
    -   `global_params.input_csv_path`: Path to your dataset.
    -   `global_params.n_iter`: The number of grid search iterations.
    -   `global_params.model_list`: The base learners to use.
    -   `ga_params` and `grid_params` to define your search space.

    Here is a minimal example to get you started:
    ```yaml
    # In your new config.yml
    global_params:
      input_csv_path: "path/to/your/data.csv"
      n_iter: 10 # Start with a small number of iterations
      model_list: ["logisticRegression", "randomForest", "XGBoost"]

    ga_params:
      pop_params: [64] # Use a single population size to start
    ```

    See the {doc}`Configuration_Guide` for a full list of options.

4.  **Activate Your Environment**:
    ```bash
    source ga_env/bin/activate
    ```

5.  **Run the Experiment**:
    -   To run with the default `config.yml`:
        ```bash
        python main.py
        ```
    -   To specify a different configuration file:
        ```bash
        python main.py --config path/to/your/config.yml
        ```
    -   To automatically evaluate the best model and generate all analysis plots after the run:
        ```bash
        python main.py --config path/to/your/config.yml --evaluate --plot
        ```

The following diagram illustrates this workflow:

!main.py Workflow

---

## Alternative: Running the Example Notebook

For development, debugging, or a more interactive, step-by-step walkthrough, you can use the `example_usage.ipynb` notebook. See the {doc}`Example_Usage_Notebook` guide for a detailed breakdown of its contents.

To execute the notebook from the command line (useful for HPC environments), use the following command from the **root** of the repository:

```bash
jupyter nbconvert --to notebook --execute notebooks/example_usage.ipynb --output notebooks/executed_example_usage.ipynb
```

This command will:

-   Run the notebook `example_usage.ipynb` using the current Python environment.
-   Save the executed version as `executed_example_usage.ipynb` in the same `notebooks/` directory.
-   Preserve interactive IPython functionality (e.g., display, widgets) during execution.

📌 **Note**: Make sure the `ga_env` (or `.venv`) environment is activated before running this command:

```bash
source ga_env/bin/activate # Or .venv/bin/activate if installed manually
```
This ensures all required dependencies are available for successful execution.