#!/bin/bash

# Checking if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "Error: requirements.txt not found! Please ensure it exists in the current directory."
    exit 1
fi

# Defining the environment name
env_name="Chest_Disease"

# Creating the conda environment
conda create --name "$env_name" --yes python=3.11
if [ $? -ne 0 ]; then
    echo "Error: Failed to create the conda environment."
    exit 1
fi

echo "Conda environment '$env_name' created successfully."

# Activating the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$env_name"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate the conda environment."
    exit 1
fi

echo "Conda environment '$env_name' activated."

# Installing dependencies from requirements.txt
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies from requirements.txt."
    exit 1
fi

echo "Dependencies installed successfully."
echo "Your conda environment '$env_name' is ready and activated."
