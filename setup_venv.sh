#!/bin/bash

# Source the .bash_profile to load functions
source ~/.bash_profile

echo "Checking for existing virtual environment..."
if [ -d ".venv" ]; then
    echo "A virtual environment already exists in the .venv directory."
    echo "If you want to recreate it, please delete the .venv directory first."
    exit 1
fi

echo "Creating virtual environment..."
/d/sw_install/python_312/python.exe -m venv .venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Error code: $?"
    exit 1
fi

echo "Virtual environment created successfully."
echo "Activating virtual environment..."
source .venv/Scripts/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment. Error code: $?"
    exit 1
fi

echo "Virtual environment activated. You can now install dependencies."

echo "Installing dependencies..."
./.venv/Scripts/python.exe -m pip install -r requirements_dev.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Error code: $?"
    exit 1
fi

echo "Upgrading pip and setuptools..."
./.venv/Scripts/python.exe -m pip install --upgrade pip setuptools
if [ $? -ne 0 ]; then
    echo "Warning: Failed to upgrade pip and setuptools. This is not critical."
    echo "You may want to manually upgrade pip and setuptools later using:"
    echo "./.venv/Scripts/python.exe -m pip install --upgrade pip setuptools"
fi

echo "Virtual environment setup complete!"
echo "To activate the virtual environment, run: source .venv/Scripts/activate"