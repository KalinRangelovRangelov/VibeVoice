# VibeVoice TTS Web App

A modern web interface for Microsoft's VibeVoice text-to-speech model, optimized for Apple Silicon.

> **Generated with [Claude Opus 4.5](https://www.anthropic.com/claude)** - 

## Demo

https://github.com/user-attachments/assets/be9fd52d-fabf-43ae-b413-65f529330093


## Requirements

- Python 3.9+
- macOS with Apple Silicon (M1/M2/M3) or NVIDIA GPU
- ~4GB RAM for the 0.5B model

## Quick Start

```bash
# 1. Setup (one-time)
./scripts/setup.sh

# 2. Start the server
./scripts/start.sh

# 3. Open in browser
open http://localhost:8000
```

## Scripts

| Script | Description |
|--------|-------------|
| `./scripts/setup.sh` | Install dependencies and create virtual environment |
| `./scripts/start.sh` | Start the server (foreground mode) |
| `./scripts/start.sh -b` | Start the server in background |
| `./scripts/stop.sh` | Stop the server |
| `./scripts/status.sh` | Check server status |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/status` | GET | Server status |
| `/api/synthesize` | POST | Generate speech from text |
| `/api/audio/{id}` | GET | Download generated audio |

## Architecture

```
VibeVoice/
├── backend/
│   ├── app.py           # FastAPI server
│   ├── requirements.txt # Python dependencies
│   └── outputs/         # Generated audio files
├── frontend/
│   ├── index.html       # Main page
│   └── static/
│       ├── css/         # Styles
│       └── js/          # JavaScript
└── scripts/
    ├── setup.sh         # Installation script
    ├── start.sh         # Start server
    ├── stop.sh          # Stop server
    └── status.sh        # Check status
```

## Notes

- First startup downloads the model (~1GB) from Hugging Face
- Supports English and Chinese text
- Uses MPS acceleration on Apple Silicon
