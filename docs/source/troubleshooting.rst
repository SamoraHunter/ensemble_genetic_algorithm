Troubleshooting Guide
=====================

This guide provides solutions to common errors and issues you might encounter while using the project.

Environment and Setup Errors
----------------------------

**Problem: ``ModuleNotFoundError: No module named 'ml_grid'``**

*   **Cause**: The Python environment is not set up correctly, or the project was not installed in editable mode.
*   **Solution**:
    1.  Ensure you have activated the correct virtual environment: ``source ga_env/bin/activate``.
    2.  From the project's root directory, install the package in editable mode: ``pip install -e .[dev]``. This links the installation to your source code.

**Problem: PyTorch models are not using the GPU.**

*   **Cause**: PyTorch was not installed with CUDA support, or your system's NVIDIA drivers are not configured correctly.
*   **Solution**:
    1.  Activate your virtual environment.
    2.  Run ``python -c "import torch; print(torch.cuda.is_available())"``.
    3.  If it returns `False`, you need to reinstall PyTorch with CUDA support. Visit the `PyTorch website <https://pytorch.org/get-started/locally/>`_ to get the correct installation command for your system's CUDA version.
    4.  Ensure the ``CUDA_VISIBLE_DEVICES`` environment variable is not set to ``"-1"`` in your script, as this explicitly disables the GPU.

Data-Related Errors
-------------------

**Problem: ``FileNotFoundError: [Errno 2] No such file or directory: 'my_data.csv'``**

*   **Cause**: The path provided in the ``input_csv_path`` variable is incorrect.
*   **Solution**:
    1.  Use an absolute path to your data file (e.g., ``/home/user/data/my_data.csv``).
    2.  If using a relative path, make sure it is correct relative to the directory where you are running the script (usually the project root).

**Problem: ``KeyError: '..._outcome_var_1'`` or "Outcome variable not found"**

*   **Cause**: The outcome variable column in your input CSV is not named correctly.
*   **Solution**:
    1.  The pipeline requires the outcome column's name to end with the suffix ``_outcome_var_1``.
    2.  Rename the column in your CSV file. For example, a column named `target` should be renamed to `target_outcome_var_1`.
    3.  Refer to the :doc:`data_preparation` guide for more details.

Performance Issues
------------------

**Problem: Experiments are running very slowly.**

*   **Cause**: This is often due to a large dataset, a large hyperparameter grid, or computationally expensive settings.
*   **Solution**:
    1.  **Start Small**: For initial tests, set ``testing=True`` in the ``ml_grid.pipeline.data.pipe`` call and use a low ``n_iter`` value (e.g., 3).
    2.  **Use Model Caching**: For a long initial run, set ``store_base_learners=True`` in the grid. For subsequent runs, set ``use_stored_base_learners=True`` to avoid retraining models.
    3.  **Check Weighting Method**: The ``weighted`` hyperparameter has a huge impact on runtime. ``'unweighted'`` is fastest. ``'de'`` and ``'ann'`` are very slow and should be used sparingly.
    4.  **Sample Your Data**: Use the ``column_sample_n`` parameter to run experiments on a random subset of features, or create a smaller sample of your CSV file for initial exploration.