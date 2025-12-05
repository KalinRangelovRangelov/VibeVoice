#!/bin/bash
# VibeVoice TTS Web App - Start Script
# Starts the FastAPI server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.server.pid"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║             VibeVoice TTS Web App - Starting                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✗ Server is already running (PID: $PID)"
        echo "  Use ./scripts/stop.sh to stop it first"
        exit 1
    else
        rm "$PID_FILE"
    fi
fi

# Check if virtual environment exists
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "✗ Virtual environment not found"
    echo "  Run ./scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "→ Activating virtual environment..."
source "$PROJECT_DIR/venv/bin/activate"
echo "✓ Virtual environment activated"

# Create outputs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/backend/outputs"

# Start the server
echo ""
echo "→ Starting VibeVoice server..."
echo "  The server will be available at: http://localhost:8000"
echo ""
echo "  Note: First startup downloads the model (~1GB) and may take a few minutes"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_DIR/backend"

# Check if running in foreground or background mode
if [ "$1" = "--background" ] || [ "$1" = "-b" ]; then
    echo "Starting in background mode..."
    nohup python -m uvicorn app:app --host 0.0.0.0 --port 8000 > "$PROJECT_DIR/server.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo ""
    echo "✓ Server started in background (PID: $(cat $PID_FILE))"
    echo "  View logs: tail -f $PROJECT_DIR/server.log"
    echo "  Stop server: ./scripts/stop.sh"
else
    # Foreground mode - trap signals for clean shutdown
    trap 'echo ""; echo "Shutting down..."; exit 0' SIGINT SIGTERM

    python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
fi
