#!/usr/bin/env bash

SHARE=false

while getopts ":s" opt; do
  case ${opt} in
    s ) # process option s
      SHARE=true
      ;;
    \? ) echo "Usage: cmd [-s]"
      ;;
  esac
done

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Setting up for macOS..."
    
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export MPS_DEVICE=mps
    
    python3.10 -m venv illusion
    source illusion/bin/activate
    pip install --upgrade pip
    pip install --upgrade setuptools wheel
    pip install -r requirements.txt
    pip uninstall -y torch
    pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
else
    echo "Setting up for Linux..."
    
    python3.10 -m venv illusion
    source illusion/bin/activate
    pip install --upgrade pip
    pip install --upgrade setuptools wheel
    pip install -r requirements.txt
fi

if $SHARE; then
    echo "Running with share option..."
    python app.py --share
else
    echo "Running without share option..."
    python app.py
fi
