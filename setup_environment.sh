#!/bin/bash

# Create Python virtual environment
d:/sw_install/python_312/./python.exe -m venv venv

# Activate virtual environment
source venv/Scripts/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo
echo "Virtual environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo 