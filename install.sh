#!/bin/bash

# Check if virtual environment exists
if [ ! -d "ga_env" ]; then
    # Create virtual environment
    python3 -m venv ga_env
fi

# Activate virtual environment
source ga_env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install ipykernel
pip install ipykernel

# Add kernel spec
python -m ipykernel install --user --name=ga_env

# Install requirements from requirements.txt
while read -r requirement; do
    pip install "$requirement"
    if [ $? -ne 0 ]; then
        echo "Failed to install $requirement" >> installation_log.txt
    else
        echo "Successfully installed $requirement"
    fi
done < requirements.txt

# Deactivate virtual environment
deactivate
