from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import io
import numpy as np
import torch
import torchaudio
import whisperx
import argparse

app = FastAPI()

# ----- Parse CLI args -----
parser = argparse.ArgumentParser(description="WhisperX FastAPI Server")
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
parser.add_argument('--port', type=int, default=8011, help='Port number')
parser.add_argument('--model', type=str, default='turbo', help='WhisperX model: turbo | small | medium | large-v3')
parser.add_argument('--language', type=str, default='vi', help='Language hint for faster transcription')
args, _ = parser.parse_known_args()

# ----- Device & Compute Optimization -----
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

# Enable TensorFloat-32 for extra GPU speed
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load model globally (only once)
try:
    model = whisperx.load_model(args.model, device, compute_type=compute_type)
except Exception as e:
    raise RuntimeError(f"Failed to load WhisperX model '{args.model}': {e}")

# Optimize CPU threads to prevent oversubscription
try:
    torch.set_num_threads(1)
except Exception:
    pass

TARGET_SR = 16000

def _read_audio_bytes_to_f32_mono(wav_bytes: bytes) -> np.ndarray:
    """Convert uploaded audio bytes -> float32 mono 16k array."""
    bio = io.BytesIO(wav_bytes)
    waveform, sr = torchaudio.load(bio)  # Handles WAV, MP3, FLAC
    waveform = waveform.to(torch.float32)

    # Convert stereo -> mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    return waveform.squeeze(0).cpu().numpy()

def _transcribe_array(x: np.ndarray, lang_hint: str | None) -> dict:
    """Run WhisperX transcription."""
    with torch.inference_mode():
        return model.transcribe(x, batch_size=16, language=lang_hint)

def _extract_text(result: dict) -> str:
    if "text" in result and result["text"]:
        return result["text"].strip()
    segments = result.get("segments") or []
    return " ".join(seg.get("text", "") for seg in segments).strip()

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(None),
    audio: UploadFile = File(None),
    language: str = Form(default=args.language),
):
    uf = file or audio
    if uf is None:
        raise HTTPException(422, detail="Missing file: upload as 'file' or 'audio'")

    data = await uf.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty")

    try:
        arr = _read_audio_bytes_to_f32_mono(data)
        if arr.size == 0:
            raise ValueError("Decoded waveform is empty")

        result = _transcribe_array(arr, lang_hint=language if language else None)
        text = _extract_text(result)

        return JSONResponse({
            "text": text,
            "language": result.get("language"),
            "num_segments": len(result.get("segments", [])),
            "duration_sec": round(len(arr) / TARGET_SR, 2)
        })

    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,           # Keep 1 worker for GPU model sharing
        loop="uvloop",       # Faster event loop
        http="httptools"     # Faster HTTP parser
    )
