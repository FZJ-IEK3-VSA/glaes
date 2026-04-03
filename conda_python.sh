#!/bin/bash
# Cross-platform conda Python wrapper for VS Code debugger.
# Uses CONDA_ENV_NAME env var (set in .env) to select the conda environment.
# Falls back to CONDA_DEFAULT_ENV (set by conda activation), then to
# parsing sys.prefix from the currently selected interpreter.

if [ -n "$CONDA_ENV_NAME" ]; then
    ENV_NAME="$CONDA_ENV_NAME"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    ENV_NAME="$CONDA_DEFAULT_ENV"
else
    # Parse env name from the interpreter's prefix path (.../envs/<name>/...)
    ENV_NAME=$(python3 -c "import sys, os; parts=sys.prefix.replace('\\\\','/').split('/'); idx=next((i for i,p in enumerate(parts) if p=='envs'), -1); print(parts[idx+1] if idx>=0 else '')" 2>/dev/null)
fi

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: Could not determine conda environment name." >&2
    echo "Set CONDA_ENV_NAME in .env or activate a conda environment." >&2
    exit 1
fi

exec conda run -n "$ENV_NAME" python "$@"
