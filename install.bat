@echo off

rem Check if virtual environment exists
if not exist ga_env (
    rem Create virtual environment
    python -m venv ga_env
)

rem Activate virtual environment
call ga_env\Scripts\activate

rem Upgrade pip
python -m pip install --upgrade pip

rem Install ipykernel
pip install ipykernel

rem Add kernel spec
python -m ipykernel install --user --name=ga_env

rem Install requirements from requirements.txt
for /f "delims=" %%i in (requirements.txt) do (
    pip install %%i
    if errorlevel 1 (
        echo Failed to install %%i >> installation_log.txt
    ) else (
        echo Successfully installed %%i
    )
)


rem Deactivate virtual environment
deactivate
