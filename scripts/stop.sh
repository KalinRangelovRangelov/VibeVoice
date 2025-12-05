#!/bin/bash
# VibeVoice TTS Web App - Stop Script
# Stops the FastAPI server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.server.pid"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║             VibeVoice TTS Web App - Stopping                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to kill process on port 8000
kill_port_8000() {
    echo "→ Looking for processes on port 8000..."
    PIDS=$(lsof -ti:8000 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "  Found processes: $PIDS"
        for PID in $PIDS; do
            echo "  Killing process $PID..."
            kill -15 "$PID" 2>/dev/null || true
        done
        sleep 1

        # Force kill if still running
        PIDS=$(lsof -ti:8000 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            for PID in $PIDS; do
                echo "  Force killing process $PID..."
                kill -9 "$PID" 2>/dev/null || true
            done
        fi
        echo "✓ Processes killed"
        return 0
    fi
    return 1
}

# Check PID file first
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "→ Found PID file with PID: $PID"

    if ps -p "$PID" > /dev/null 2>&1; then
        echo "  Stopping server process..."
        kill -15 "$PID" 2>/dev/null || true
        sleep 2

        if ps -p "$PID" > /dev/null 2>&1; then
            echo "  Force stopping..."
            kill -9 "$PID" 2>/dev/null || true
        fi
        echo "✓ Server stopped"
    else
        echo "  Process not running"
    fi

    rm -f "$PID_FILE"
fi

# Also kill any lingering processes on port 8000
kill_port_8000 || echo "  No processes found on port 8000"

# Clean up old audio files (optional)
OUTPUTS_DIR="$PROJECT_DIR/backend/outputs"
if [ -d "$OUTPUTS_DIR" ]; then
    FILE_COUNT=$(find "$OUTPUTS_DIR" -name "*.wav" -type f | wc -l)
    if [ "$FILE_COUNT" -gt 0 ]; then
        echo ""
        read -p "→ Found $FILE_COUNT audio files. Delete them? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$OUTPUTS_DIR"/*.wav
            echo "✓ Audio files deleted"
        fi
    fi
fi

echo ""
echo "✓ Server stopped successfully"
