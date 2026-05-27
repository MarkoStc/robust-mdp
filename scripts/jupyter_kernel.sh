#!/usr/bin/env bash
# Launcher for the ipykernel inside the uenv pytorch image + qa-gym venv.
# Used as the "argv" of a kernelspec so VS Code / JupyterLab can start the kernel
# regardless of whether the user has uenv loaded in their shell.
set -euo pipefail
exec uenv run pytorch/v2.9.1:v2 --view=default -- bash -c \
    'source ~/qa-gym/.venv/bin/activate && exec python -m ipykernel_launcher "$@"' \
    -- "$@"
