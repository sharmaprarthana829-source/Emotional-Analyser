#!/usr/bin/env bash
set -euo pipefail

# Optional: install OS packages only when apt-get is available and permitted.
if command -v apt-get >/dev/null 2>&1; then
  echo "apt-get found. Attempting to install system dependencies..."
  if apt-get update && apt-get install -y libgl1 libglib2.0-0; then
    echo "System dependencies installed successfully."
  else
    echo "Skipping apt packages (not permitted in this environment)."
  fi
else
  echo "apt-get not available. Skipping system package install."
fi

pip install --upgrade pip
pip install -r requirements.txt
