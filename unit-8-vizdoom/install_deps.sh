#!/bin/bash

# ViZDoom Dependencies Installation Script
# Based on: https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#-linux

set -e  # Exit on any error

echo "🎮 Installing ViZDoom dependencies..."

# Check if running as root for system packages
if [[ $EUID -eq 0 ]]; then
    APT_CMD="apt-get"
else
    APT_CMD="sudo apt-get"
    echo "Note: Will use sudo for system package installation"
fi

# Update package list
echo "📦 Updating package list..."
$APT_CMD update

# Install system dependencies for ViZDoom
echo "🔧 Installing system dependencies..."
$APT_CMD install -y \
    build-essential \
    zlib1g-dev \
    libsdl2-dev \
    libjpeg-dev \
    nasm \
    tar \
    libbz2-dev \
    libgtk2.0-dev \
    cmake \
    git \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    timidity \
    libwildmidi-dev \
    unzip \
    ffmpeg

# Boost libraries
echo "🚀 Installing Boost libraries..."
$APT_CMD install -y libboost-all-dev

# Lua binding dependencies
echo "🌙 Installing Lua dependencies..."
$APT_CMD install -y liblua5.1-dev

# Additional dependencies for headless environments
echo "🖥️  Installing display dependencies for headless environments..."
$APT_CMD install -y xvfb x11-utils

# # Python environment setup
# echo "🐍 Setting up Python environment..."

# # Check if we're in the unit-8-vizdoom directory
# if [[ ! -f "pyproject.toml" ]]; then
#     echo "⚠️  pyproject.toml not found. Creating basic configuration..."
#     cat > pyproject.toml << EOF
# [project]
# name = "vizdoom-unit-8"
# version = "0.1.0"
# description = "ViZDoom experiments for Unit 8"
# requires-python = ">=3.8"
# dependencies = [
#     "vizdoom>=1.2.0",
#     "gymnasium>=1.0.0",
#     "numpy>=1.21.0",
#     "torch>=1.12.0",
#     "stable-baselines3>=2.0.0",
#     "matplotlib>=3.5.0",
#     "opencv-python>=4.5.0",
#     "pillow>=8.0.0",
#     "tqdm>=4.60.0",
# ]
# EOF
# fi

# # Create virtual environment and install Python packages
# if command -v uv &> /dev/null; then
#     echo "📦 Using uv for Python package management..."
#     uv sync
# else
#     echo "📦 uv not found, using pip..."
#     python3 -m venv .venv
#     source .venv/bin/activate
#     pip install --upgrade pip
#     pip install vizdoom gymnasium numpy torch stable-baselines3 matplotlib opencv-python pillow tqdm
# fi

# # Test ViZDoom installation
# echo "🧪 Testing ViZDoom installation..."
# if command -v uv &> /dev/null; then
#     uv run python -c "import vizdoom; print('✅ ViZDoom imported successfully')"
# else
#     source .venv/bin/activate
#     python -c "import vizdoom; print('✅ ViZDoom imported successfully')"
# fi

# echo "🎉 ViZDoom dependencies installed successfully!"
# echo ""
# echo "To activate the environment:"
# if command -v uv &> /dev/null; then
#     echo "  No activation needed with uv - just use 'uv run python your_script.py'"
# else
#     echo "  source .venv/bin/activate"
# fi
# echo ""
# echo "For headless environments, prefix commands with:"
# echo "  xvfb-run -a your_command"