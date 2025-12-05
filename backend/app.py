"""
VibeVoice TTS Web API
FastAPI backend for text-to-speech synthesis using Microsoft's VibeVoice model
"""

import os
import sys
import uuid
import copy
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

# Add vibevoice repo to path
VIBEVOICE_REPO = Path(__file__).parent.parent / "vibevoice-repo"
if VIBEVOICE_REPO.exists():
    sys.path.insert(0, str(VIBEVOICE_REPO))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Global model instance
model = None
processor = None
device = None
dtype = None
model_loaded = False
voice_prompts = {}  # Dict of voice_name -> voice_prompt
loading_status = "initializing"  # Track loading progress

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

VOICES_DIR = Path(__file__).parent / "voices"
VOICES_DIR.mkdir(exist_ok=True)

MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"

# Available voices
AVAILABLE_VOICES = {
    "emma": {"file": "en-Emma_woman.pt", "name": "Emma", "gender": "Woman", "accent": "English"},
    "grace": {"file": "en-Grace_woman.pt", "name": "Grace", "gender": "Woman", "accent": "English"},
    "carter": {"file": "en-Carter_man.pt", "name": "Carter", "gender": "Man", "accent": "English"},
    "davis": {"file": "en-Davis_man.pt", "name": "Davis", "gender": "Man", "accent": "English"},
    "frank": {"file": "en-Frank_man.pt", "name": "Frank", "gender": "Man", "accent": "English"},
    "mike": {"file": "en-Mike_man.pt", "name": "Mike", "gender": "Man", "accent": "English"},
    "samuel": {"file": "in-Samuel_man.pt", "name": "Samuel", "gender": "Man", "accent": "Indian"},
}
DEFAULT_VOICE = "emma"


# Request/Response models
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: str = Field(default=DEFAULT_VOICE, description="Voice to use")

class TTSResponse(BaseModel):
    audio_url: str
    audio_id: str
    duration: float
    message: str

class StatusResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    loading_message: str

class VoiceInfo(BaseModel):
    id: str
    name: str
    gender: str
    accent: str


def get_device():
    """Detect the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download_voice_prompt(voice_id: str = DEFAULT_VOICE):
    """Download a voice prompt from GitHub repo"""
    global voice_prompts

    if voice_id not in AVAILABLE_VOICES:
        print(f"[VibeVoice] Unknown voice: {voice_id}")
        return None

    voice_info = AVAILABLE_VOICES[voice_id]
    voice_filename = voice_info["file"]
    voice_path = VOICES_DIR / voice_filename

    # Check if already loaded in memory
    if voice_id in voice_prompts:
        return voice_prompts[voice_id]

    # Check if cached on disk
    if voice_path.exists():
        print(f"[VibeVoice] Loading cached voice: {voice_info['name']}...")
        voice_prompts[voice_id] = torch.load(voice_path, map_location="cpu", weights_only=False)
        return voice_prompts[voice_id]

    print(f"[VibeVoice] Downloading voice: {voice_info['name']}...")
    try:
        import urllib.request

        # Download from the GitHub repo directly
        voice_url = f"https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/{voice_filename}"

        print(f"[VibeVoice] Fetching from {voice_url}")
        urllib.request.urlretrieve(voice_url, str(voice_path))

        voice_prompts[voice_id] = torch.load(voice_path, map_location="cpu", weights_only=False)
        print(f"[VibeVoice] Voice '{voice_info['name']}' ready")
        return voice_prompts[voice_id]

    except Exception as e:
        print(f"[VibeVoice] Could not download voice: {e}")
        # Try to load from local vibevoice-repo if available
        local_voice = VIBEVOICE_REPO / "demo" / "voices" / "streaming_model" / voice_filename
        if local_voice.exists():
            print(f"[VibeVoice] Using local voice from repo...")
            voice_prompts[voice_id] = torch.load(local_voice, map_location="cpu", weights_only=False)
            return voice_prompts[voice_id]
        else:
            print(f"[VibeVoice] ERROR: Voice '{voice_id}' not available.")
            return None


def load_model():
    """Load the VibeVoice model"""
    global model, processor, device, dtype, model_loaded, voice_prompt, loading_status

    device = get_device()
    loading_status = f"Initializing on {device}..."
    print(f"[VibeVoice] {loading_status}")

    try:
        # Import VibeVoice classes
        loading_status = "Loading VibeVoice modules..."
        print(f"[VibeVoice] {loading_status}")

        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor
        )

        # Device-specific configuration
        if device == "mps":
            dtype = torch.float32
            attn_impl = "sdpa"
            device_map = None  # Will move manually
        elif device == "cuda":
            dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
            device_map = "cuda"
        else:
            dtype = torch.float32
            attn_impl = "sdpa"
            device_map = "cpu"

        loading_status = "Downloading processor (first run only)..."
        print(f"[VibeVoice] {loading_status}")
        processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_ID)

        loading_status = "Downloading model weights (~1GB, first run only)..."
        print(f"[VibeVoice] {loading_status}")
        try:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            print(f"[VibeVoice] Flash attention failed ({e}), falling back to SDPA...")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map=device_map if device != "cuda" else None,
                attn_implementation="sdpa",
            )

        # Move model to device for MPS
        loading_status = f"Loading model to {device.upper()}..."
        print(f"[VibeVoice] {loading_status}")
        if device == "mps":
            model = model.to("mps")

        # Ensure return_dict=True for proper output format
        if hasattr(model, 'config'):
            model.config.return_dict = True
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            if hasattr(model.model.language_model, 'config'):
                model.model.language_model.config.return_dict = True

        model.eval()

        # Download default voice prompt
        loading_status = "Downloading default voice..."
        print(f"[VibeVoice] {loading_status}")
        download_voice_prompt(DEFAULT_VOICE)

        model_loaded = True
        loading_status = "Ready"
        print(f"[VibeVoice] Model loaded successfully on {device}")

    except ImportError as e:
        loading_status = f"Error: {e}"
        print(f"[VibeVoice] Import error: {e}")
        print("[VibeVoice] Make sure you ran ./scripts/setup.sh to install the VibeVoice package")
        raise
    except Exception as e:
        loading_status = f"Error: {e}"
        print(f"[VibeVoice] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_model_background():
    """Load model in background thread"""
    import threading
    thread = threading.Thread(target=load_model, daemon=True)
    thread.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Load model in background so server can respond immediately
    load_model_background()
    yield


# App configuration
app = FastAPI(
    title="VibeVoice TTS",
    description="Text-to-Speech API powered by Microsoft VibeVoice",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the frontend"""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    return FileResponse(frontend_path)


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get API status"""
    return StatusResponse(
        status="ready" if model_loaded else "loading",
        device=device or "unknown",
        model_loaded=model_loaded,
        loading_message=loading_status
    )


@app.get("/api/voices", response_model=list[VoiceInfo])
async def get_voices():
    """Get available voices"""
    return [
        VoiceInfo(id=vid, name=v["name"], gender=v["gender"], accent=v["accent"])
        for vid, v in AVAILABLE_VOICES.items()
    ]


@app.post("/api/synthesize", response_model=TTSResponse)
async def synthesize(request: TTSRequest):
    """Synthesize speech from text"""
    global model, processor, device, dtype, voice_prompts

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Get the selected voice (download if not cached)
    voice_id = request.voice if request.voice in AVAILABLE_VOICES else DEFAULT_VOICE
    voice_prompt = download_voice_prompt(voice_id)

    if voice_prompt is None:
        raise HTTPException(
            status_code=503,
            detail=f"Voice '{voice_id}' not available. Please try another voice."
        )

    try:
        import scipy.io.wavfile as wav

        def move_to_device(obj, target_device, target_dtype):
            """Recursively move tensors in nested dict/list to device, preserving object types"""
            from transformers.modeling_outputs import ModelOutput
            from transformers.cache_utils import DynamicCache

            if torch.is_tensor(obj):
                if obj.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    return obj.to(target_device, dtype=target_dtype)
                else:
                    return obj.to(target_device)
            elif isinstance(obj, DynamicCache):
                # Handle DynamicCache by moving all tensors in key_cache and value_cache
                new_cache = DynamicCache()
                new_cache.key_cache = [
                    move_to_device(k, target_device, target_dtype) for k in obj.key_cache
                ]
                new_cache.value_cache = [
                    move_to_device(v, target_device, target_dtype) for v in obj.value_cache
                ]
                # Copy other attributes
                if hasattr(obj, '_seen_tokens'):
                    new_cache._seen_tokens = obj._seen_tokens
                return new_cache
            elif isinstance(obj, ModelOutput):
                # Preserve ModelOutput subclass type (like BaseModelOutputWithPast)
                new_dict = {k: move_to_device(v, target_device, target_dtype) for k, v in obj.items()}
                return obj.__class__(**new_dict)
            elif isinstance(obj, dict):
                return {k: move_to_device(v, target_device, target_dtype) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                moved = [move_to_device(v, target_device, target_dtype) for v in obj]
                return type(obj)(moved) if isinstance(obj, tuple) else moved
            return obj

        # Prepare cached prompt (voice conditioning) - required for VibeVoice
        # Must move all tensors to the same device as the model
        cached_prompt = copy.deepcopy(voice_prompt)
        cached_prompt = move_to_device(cached_prompt, device, dtype)

        # Process input text (matching demo code exactly)
        inputs = processor.process_input_with_cached_prompt(
            text=request.text,
            cached_prompt=cached_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move inputs to device
        inputs = move_to_device(inputs, device, dtype)

        # Also move the prefilled outputs for generation
        prefilled = move_to_device(copy.deepcopy(voice_prompt), device, dtype) if voice_prompt else None

        # Generate audio (matching demo code exactly)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.5,
                tokenizer=processor.tokenizer,
                do_sample=False,
                verbose=False,
                all_prefilled_outputs=prefilled,
            )

        # Extract audio from outputs (VibeVoice uses speech_outputs)
        if hasattr(outputs, 'speech_outputs') and len(outputs.speech_outputs) > 0:
            audio = outputs.speech_outputs[0]
        elif hasattr(outputs, 'audio'):
            audio = outputs.audio
        elif hasattr(outputs, 'waveform'):
            audio = outputs.waveform
        elif isinstance(outputs, dict):
            audio = outputs.get('speech_outputs', outputs.get('audio', outputs.get('waveform', None)))
            if isinstance(audio, list) and len(audio) > 0:
                audio = audio[0]
        elif isinstance(outputs, (list, tuple)):
            audio = outputs[0]
        else:
            audio = outputs

        # Convert to numpy
        if torch.is_tensor(audio):
            audio = audio.cpu().float().numpy()

        audio = np.array(audio).squeeze()

        # VibeVoice uses 24kHz sample rate
        sample_rate = 24000

        # Normalize and convert to int16
        audio_max = np.abs(audio).max()
        if audio_max > 0:
            if audio_max > 1.0:
                audio = audio / audio_max
            audio_normalized = (audio * 32767).astype(np.int16)
        else:
            audio_normalized = np.zeros_like(audio, dtype=np.int16)

        # Save to file
        audio_id = str(uuid.uuid4())
        audio_path = OUTPUT_DIR / f"{audio_id}.wav"
        wav.write(str(audio_path), sample_rate, audio_normalized)

        duration = len(audio) / sample_rate

        return TTSResponse(
            audio_url=f"/api/audio/{audio_id}.wav",
            audio_id=audio_id,
            duration=round(duration, 2),
            message="Speech synthesized successfully"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio files"""
    audio_path = OUTPUT_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/wav")


@app.delete("/api/audio/{filename}")
async def delete_audio(filename: str):
    """Delete an audio file"""
    audio_path = OUTPUT_DIR / filename
    if audio_path.exists():
        audio_path.unlink()
    return {"message": "Deleted"}


# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent.parent / "frontend" / "static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
