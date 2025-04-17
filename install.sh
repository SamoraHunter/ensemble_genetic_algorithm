#!/bin/bash

# Function to print error message and exit
print_error_and_exit() {
    echo "$1"
    echo "Please try deleting the existing 'ga_env' directory and run the script again."
    exit 1
}

# Determine which Python command to use
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    # Check if python is actually Python 3
    PY_VERSION=$(python --version 2>&1)
    if [[ $PY_VERSION == *"Python 3"* ]]; then
        PYTHON_CMD="python"
    else
        print_error_and_exit "Python 3 is required but only Python 2 was found. Please install Python 3."
    fi
else
    print_error_and_exit "Python 3 is not found. Please make sure Python 3 is installed."
fi

echo "Using Python command: $PYTHON_CMD"

# Check if python3-venv package is installed (if using apt-based distro)
if command -v dpkg &> /dev/null; then
    if ! dpkg -s python3-venv &> /dev/null; then
        echo "Warning: The 'python3-venv' package might not be installed."
        echo "If you encounter errors, please install it using: sudo apt-get install python3-venv"
    fi
fi

# Check if virtual environment exists
if [ ! -d "ga_env" ]; then
    # Create virtual environment
    $PYTHON_CMD -m venv ga_env || print_error_and_exit "Failed to create virtual environment"
    echo "Virtual environment created successfully."
fi

# Activate virtual environment
source ga_env/bin/activate || print_error_and_exit "Failed to activate virtual environment"
echo "Virtual environment activated."

# Upgrade pip
python -m pip install --upgrade pip

# Install ipykernel
pip install ipykernel

# Add kernel spec
python -m ipykernel install --user --name=ga_env

# Install requirements from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Note: requirements.txt not found. Skipping package installation."
fi

echo "Setup completed successfully!"

# Deactivate virtual environment
deactivate