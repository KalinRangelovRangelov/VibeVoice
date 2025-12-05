#!/bin/bash
# VibeVoice TTS Web App - Setup Script
# This script sets up the environment and installs all dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║             VibeVoice TTS Web App - Setup                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "→ Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "✗ Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION detected"

# Clone VibeVoice repository
echo ""
echo "→ Setting up VibeVoice repository..."
cd "$PROJECT_DIR"

if [ -d "vibevoice-repo" ]; then
    echo "  Repository already exists, pulling latest..."
    cd vibevoice-repo && git pull && cd ..
else
    echo "  Cloning Microsoft VibeVoice repository..."
    git clone https://github.com/microsoft/VibeVoice.git vibevoice-repo
fi
echo "✓ VibeVoice repository ready"

# Create virtual environment
echo ""
echo "→ Creating virtual environment..."
cd "$PROJECT_DIR"

if [ -d "venv" ]; then
    echo "  Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "→ Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "→ Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"

# Install VibeVoice package
echo ""
echo "→ Installing VibeVoice package..."
cd "$PROJECT_DIR/vibevoice-repo"
pip install -e . -q
cd "$PROJECT_DIR"
echo "✓ VibeVoice package installed"

# Install web app dependencies
echo ""
echo "→ Installing web app dependencies..."
pip install -r backend/requirements.txt -q
echo "✓ Dependencies installed"

# Download a sample voice prompt
echo ""
echo "→ Setting up voice prompts..."
mkdir -p "$PROJECT_DIR/backend/voices"
if [ ! -f "$PROJECT_DIR/backend/voices/default.pt" ]; then
    echo "  Voice prompts will be downloaded on first run"
fi
echo "✓ Voice directory ready"

# Check for Apple Silicon
echo ""
echo "→ Checking system architecture..."
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "✓ Apple Silicon detected - MPS acceleration will be available"
else
    echo "  x86_64 architecture detected"
fi

# Create outputs directory
mkdir -p "$PROJECT_DIR/backend/outputs"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Run ./scripts/start.sh to start the server"
echo "  2. Open http://localhost:8000 in your browser"
echo ""
echo "Note: The first run will download the VibeVoice model (~1GB)"
echo "      This may take a few minutes depending on your connection."
echo ""
