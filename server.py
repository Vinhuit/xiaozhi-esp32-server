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


# ----- Parse command-line arguments -----
parser = argparse.ArgumentParser(description="WhisperX FastAPI server")
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
parser.add_argument('--port', type=int, default=8022, help='Port to bind')
parser.add_argument('--model', type=str, default='turbo', help='Model name: turbo | small | medium | large-v3')
parser.add_argument('--language', type=str, default='vi', help='Language hint for transcription')
args, _ = parser.parse_known_args()

# ----- Load model once (GPU if available) -----
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
model_name = args.model
model = whisperx.load_model(model_name, device, compute_type=compute_type)

# Optional: reduce CPU thread contention a bit
try:
    torch.set_num_threads(1)
except Exception:
    pass

TARGET_SR = 16000

def _read_audio_bytes_to_f32_mono(wav_bytes: bytes) -> np.ndarray:
    """Read bytes -> float32 mono @16k, all in memory, no temp files."""
    bio = io.BytesIO(wav_bytes)
    # torchaudio handles WAV/PCM reliably; returns Tensor [channels, samples]
    wav, sr = torchaudio.load(bio)  # float32 in [-1, 1] if WAV float, else int -> converted later
    # Convert to float32
    if wav.dtype != torch.float32:
        wav = wav.to(torch.float32) / 32768.0  # from int16 to float32 range
    # Mono
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    # Flatten to 1D numpy
    return wav.squeeze(0).cpu().numpy()

def _transcribe_array(x: np.ndarray, lang_hint: str | None) -> dict:
    # Give language hint to avoid detection latency when you know it (e.g., "vi")
    with torch.inference_mode():
        return model.transcribe(x, batch_size=16, language=lang_hint)

def _best_text(result: dict) -> str:
    if "text" in result and result["text"]:
        return result["text"].strip()
    segs = result.get("segments") or []
    return " ".join(s.get("text", "") for s in segs).strip()

def _extract_upload(file: UploadFile | None, audio: UploadFile | None) -> UploadFile:
    uf = file or audio
    if uf is None:
        raise HTTPException(422, detail="missing file; send multipart with field 'file' or 'audio'")
    return uf

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(None),
    audio: UploadFile = File(None),
    language: str = Form("vi"),   # hint for latency; change default if needed
):
    uf = _extract_upload(file, audio)
    data = await uf.read()
    if not data:
        raise HTTPException(400, "empty audio")

    try:
        arr = _read_audio_bytes_to_f32_mono(data)
        if arr.size == 0:
            raise ValueError("decoded empty waveform")
        result = _transcribe_array(arr, lang_hint=language if language else None)
        text = _best_text(result)
        return JSONResponse({"text": text, "language": result.get("language"), "num_segments": len(result.get("segments", []))})
    except Exception as e:
        raise HTTPException(500, f"transcription failed: {e}")

if __name__ == "__main__":
    # Single worker per GPU to avoid memory duplication; use uvloop/httptools for speed
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        loop="uvloop",
        http="httptools",
    )
