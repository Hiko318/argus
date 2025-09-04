#!/bin/bash

# Foresight Installation Script
# This script sets up the Foresight environment and dependencies

set -e  # Exit on any error

echo "Starting Foresight installation..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root for security reasons."
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
echo "Checking system requirements..."

# Check Python
if ! command_exists python3; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check pip
if ! command_exists pip3; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

# Check Node.js (for frontend)
if ! command_exists node; then
    echo "Warning: Node.js not found. Frontend components may not work."
fi

# Check npm
if ! command_exists npm; then
    echo "Warning: npm not found. Frontend dependencies cannot be installed."
fi

# Create virtual environment
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Installing basic dependencies..."
    pip install fastapi uvicorn opencv-python numpy
fi

# Install frontend dependencies
if command_exists npm && [ -f "src/frontend/package.json" ]; then
    echo "Installing frontend dependencies..."
    cd src/frontend
    npm install
    cd ../..
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p data
mkdir -p temp

# Set up Git LFS if not already configured
if command_exists git; then
    echo "Configuring Git LFS..."
    git lfs install
fi

# Make scripts executable
echo "Setting script permissions..."
chmod +x scripts/*.sh

echo "Installation completed successfully!"
echo ""
echo "To start the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start the backend: uvicorn src.backend.main:app --reload"
echo "3. Start the frontend (if applicable): cd src/frontend && npm start"
echo ""
echo "For more information, see the README.md file."