#!/bin/bash
# VibeVoice TTS Web App - Status Script
# Shows the current status of the server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_DIR/.server.pid"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║             VibeVoice TTS Web App - Status                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if server is running
RUNNING=false
PID=""

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        RUNNING=true
    fi
fi

# Also check port 8000
PORT_PIDS=$(lsof -ti:8000 2>/dev/null || true)
if [ -n "$PORT_PIDS" ]; then
    RUNNING=true
    PID="$PORT_PIDS"
fi

echo "Server Status:"
if [ "$RUNNING" = true ]; then
    echo "  ● Running (PID: $PID)"
    echo ""
    echo "  URL: http://localhost:8000"
    echo ""

    # Try to get API status
    API_STATUS=$(curl -s http://localhost:8000/api/status 2>/dev/null || echo '{"status":"unavailable"}')
    echo "  API Status: $API_STATUS"
else
    echo "  ○ Stopped"
fi

echo ""
echo "Virtual Environment:"
if [ -d "$PROJECT_DIR/venv" ]; then
    echo "  ✓ Installed"
else
    echo "  ✗ Not installed (run ./scripts/setup.sh)"
fi

echo ""
echo "Output Files:"
OUTPUTS_DIR="$PROJECT_DIR/backend/outputs"
if [ -d "$OUTPUTS_DIR" ]; then
    FILE_COUNT=$(find "$OUTPUTS_DIR" -name "*.wav" -type f 2>/dev/null | wc -l | tr -d ' ')
    TOTAL_SIZE=$(du -sh "$OUTPUTS_DIR" 2>/dev/null | cut -f1)
    echo "  Files: $FILE_COUNT"
    echo "  Size: $TOTAL_SIZE"
else
    echo "  Directory not created yet"
fi

echo ""
