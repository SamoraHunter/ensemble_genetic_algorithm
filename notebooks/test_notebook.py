import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path # Import Path for robust path handling

def test_notebook():
    """
    Reads, executes, and checks a notebook for errors.
    """
    notebook_path = Path(__file__).parent / "example_usage.ipynb"
    notebook_dir = notebook_path.parent

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)


    # This tells the executor to run the notebook from within its own directory.
    ep = ExecutePreprocessor(
        timeout=5400,
        kernel_name="ga_env",
        cwd=notebook_dir  
    )

    try:
        # Execute the notebook. 
        ep.preprocess(nb, {'metadata': {'path': str(notebook_dir)}})
    except Exception as e:
        # If the notebook fails, this will raise a more informative error.
        pytest.fail(f"Error executing notebook {notebook_path.name}: \n{e}")