@echo off
REM Cross-platform conda Python wrapper for VS Code debugger.
REM Uses CONDA_ENV_NAME env var (set in .env) to select the conda environment.
REM Falls back to CONDA_DEFAULT_ENV (set by conda activation), then to
REM parsing the env name from the currently selected interpreter's prefix path.

if defined CONDA_ENV_NAME (
    set _ENV_NAME=%CONDA_ENV_NAME%
    goto run
)

if defined CONDA_DEFAULT_ENV (
    set _ENV_NAME=%CONDA_DEFAULT_ENV%
    goto run
)

REM Parse env name from interpreter prefix: ...\envs\<name>\...
for /f "tokens=*" %%i in ('python -c "import sys,os; p=sys.prefix.replace(chr(92),'/'); parts=p.split('/'); idx=next((i for i,x in enumerate(parts) if x=='envs'),-1); print(parts[idx+1] if idx>=0 else '')" 2^>nul') do set _ENV_NAME=%%i

if "%_ENV_NAME%"=="" (
    echo ERROR: Could not determine conda environment name. 1>&2
    echo Set CONDA_ENV_NAME in .env or activate a conda environment. 1>&2
    exit /b 1
)

:run
conda run -n %_ENV_NAME% python %*
