#!/bin/bash

# Display the current working directory
echo "Current directory: $(pwd)"

# Install all dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Run the FastAPI backend server
echo "Starting the backend server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000 
 
 
 