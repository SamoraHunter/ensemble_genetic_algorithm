@echo off

rem Set paths
set "venv_dir=ga_env"
set "requirements_file=requirements.txt"
set "log_file=installation_log.txt"

rem Check if virtual environment exists
if not exist "%venv_dir%" (
    rem Create virtual environment
    python -m venv "%venv_dir%"
)

rem Activate virtual environment
echo Activating virtual environment...
call "%venv_dir%\Scripts\activate"
if errorlevel 1 (
    echo Failed to activate virtual environment
    exit /b 1
)

rem Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo Virtual environment is not activated
    exit /b 1
)

rem Perform a test to ensure it's activated before installing any packages
python -c "import sys; print('Python version:', sys.version)"

rem Upgrade pip
python -m pip install --upgrade pip

rem Install ipykernel
pip install ipykernel

rem Add kernel spec
python -m ipykernel install --user --name=%venv_dir%

rem Install requirements from requirements.txt
for /f "delims=" %%i in (%requirements_file%) do (
    pip install %%i
    if not !errorlevel!==0 (
        echo Failed to install %%i >> %log_file%
    ) else (
        echo Successfully installed %%i
    )
)

echo.

rem Deactivate virtual environment
call "%venv_dir%\Scripts\deactivate.bat"
