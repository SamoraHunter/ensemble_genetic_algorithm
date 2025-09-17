#!/bin/bash

# Function to print error message and exit
print_error_and_exit() {
    echo "ERROR: $1"
    echo "Please try deleting the existing 'ga_env' directory and run the script again."
    exit 1
}

# Function to print info message
print_info() {
    echo "INFO: $1"
}

# Function to print warning message
print_warning() {
    echo "WARNING: $1"
}

# Parse command line arguments
INSTALL_TYPE="default"
FORCE_RECREATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            INSTALL_TYPE="cpu"
            shift
            ;;
        --gpu)
            INSTALL_TYPE="gpu"
            shift
            ;;
        --dev)
            INSTALL_TYPE="dev"
            shift
            ;;
        --all)
            INSTALL_TYPE="all"
            shift
            ;;
        --force)
            FORCE_RECREATE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --cpu     Install CPU-only version (good for CI/testing)"
            echo "  --gpu     Install with GPU support"
            echo "  --dev     Install with development dependencies"
            echo "  --all     Install all dependencies (dev + gpu)"
            echo "  --force   Force recreation of virtual environment"
            echo "  --help    Show this help message"
            exit 0
            ;;
        *)
            print_error_and_exit "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

print_info "Installing with profile: $INSTALL_TYPE"

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

print_info "Using Python command: $PYTHON_CMD"

# Check Python version
PY_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_info "Python version: $PY_VERSION"

if [[ "$PY_VERSION" < "3.10" ]]; then
    print_error_and_exit "Python 3.10 or higher is required. Found: $PY_VERSION"
fi

# Check if python3-venv package is installed (if using apt-based distro)
if command -v dpkg &> /dev/null; then
    if ! dpkg -s python3-venv &> /dev/null; then
        print_warning "The 'python3-venv' package might not be installed."
        print_warning "If you encounter errors, please install it using: sudo apt-get install python3-venv"
    fi
fi

# Check if we need to recreate virtual environment
if [ "$FORCE_RECREATE" = true ] && [ -d "ga_env" ]; then
    print_info "Force recreation requested. Removing existing virtual environment..."
    rm -rf ga_env
fi

# Check if virtual environment exists
if [ ! -d "ga_env" ]; then
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv ga_env || print_error_and_exit "Failed to create virtual environment"
    print_info "Virtual environment created successfully."
else
    print_info "Virtual environment already exists."
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source ga_env/bin/activate || print_error_and_exit "Failed to activate virtual environment"
print_info "Virtual environment activated."

# Upgrade pip and install build tools
print_info "Upgrading pip and installing build tools..."
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    print_error_and_exit "pyproject.toml not found. Please make sure it exists in the current directory."
fi

# Install the package based on the selected type
print_info "Installing package dependencies..."
case $INSTALL_TYPE in
    "cpu")
        print_info "Installing CPU-only version..."
        # Install CPU-only PyTorch first
        pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
        # Install the package without GPU dependencies
        pip install -e ".[dev]"
        ;;
    "gpu")
        print_info "Installing with GPU support..."
        pip install -e ".[gpu]"
        ;;
    "dev")
        print_info "Installing with development dependencies..."
        pip install -e ".[dev]"
        ;;
    "all")
        print_info "Installing all dependencies..."
        pip install -e ".[all]"
        ;;
    *)
        print_info "Installing default dependencies..."
        pip install -e .
        ;;
esac

# Set up git hooks if the script exists and this is a git repository
if [ -d ".git" ] && [ -f "setup-hooks.sh" ]; then
    print_info "Setting up git hooks for development..."
    ./setup-hooks.sh
fi

# Install ipykernel and register the environment
print_info "Setting up Jupyter kernel..."
pip install ipykernel
python -m ipykernel install --user --name=ga_env --display-name="GA Project Environment"

# Verify installation
print_info "Verifying installation..."
python -c "
import sys
print(f'Python version: {sys.version}')
print(f'Python executable: {sys.executable}')

# Test key imports
try:
    import numpy
    print(f'NumPy version: {numpy.__version__}')
except ImportError:
    print('WARNING: NumPy not found')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('WARNING: PyTorch not found')

try:
    import pandas
    print(f'Pandas version: {pandas.__version__}')
except ImportError:
    print('WARNING: Pandas not found')
"

print_info "Setup completed successfully!"
print_info "To activate the environment in the future, run: source ga_env/bin/activate"

# Show next steps
echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "=== Next Steps for All Users ==="
echo "1. The virtual environment 'ga_env' is active for this terminal session."
echo "2. For future sessions, activate it with: source ga_env/bin/activate"
echo "3. To start Jupyter, run: jupyter lab"
echo "   - In your notebook, select the 'GA Project Environment' kernel."
echo ""
echo "=== For Developers/Contributors ==="
echo "  - Git hooks have been set up automatically."
echo "  - To run tests, use: pytest"
echo ""
echo "=== Re-running Setup ==="
echo "  You can re-run this script with different options (e.g., ./setup.sh --gpu --force)."
echo "  Use ./setup.sh --help to see all options."

# Keep environment activated for the user
echo ""
print_info "Virtual environment remains activated for this session."