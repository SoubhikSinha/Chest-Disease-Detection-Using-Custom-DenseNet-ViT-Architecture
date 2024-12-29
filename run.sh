#!/bin/bash

# Defining the environment name
env_name="Chest_Disease"

# Activating the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$env_name"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate the conda environment '$env_name'."
    exit 1
fi

echo "Conda environment '$env_name' activated."

# Running the Python script to launch Gradio
python final_project_Gradio_App.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to run the Python script."
    exit 1
fi

echo "Gradio application is running."