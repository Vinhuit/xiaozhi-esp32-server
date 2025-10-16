import os
import struct
import time
import uuid
import aiohttp
import opuslib_next

from typing import Optional, Tuple, List, Dict, Any

from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    """
    WhisperX (remote) ASR provider designed to work cleanly with ASRProviderBase.

    Key points:
    - Do NOT override receive_audio; let ASRProviderBase manage buffering/VAD/stop.
    - All work happens in speech_to_text(), which the base calls once per utterance.
    - Creates a per-call aiohttp.ClientSession (bound to the loop that calls it) to
      avoid "Future attached to a different loop" errors when the base runs in threads.

    Remote endpoint (FastAPI style):
      POST http://asr-server:8022/transcriber
      multipart/form-data with one of:
        - field 'file' = audio.wav   (FastAPI default)
        - field 'audio' = audio.wav  (fallback)

    Config (optional):
      - remote_url: str              default "http://asr-server:8022/transcriber"
      - request_timeout_s: int       default 20
      - save_tmp_wav: bool           default True (mirrors sherpa logs)
    """

    def __init__(self, config, delete_audio_file):
        super().__init__()
        self.interface_type = InterfaceType.STREAM

        # Audio decode params
        self.sample_rate = 16000
        self.channels = 1
        self.bits_per_sample = 16
        self.decoder = opuslib_next.Decoder(self.sample_rate, self.channels)

        # Remote server
        self.remote_url = config.get("remote_url", "http://asr-server:8022/transcribe")
        self.request_timeout_s = int(config.get("request_timeout_s", 20))

        # Temp file behavior (for sherpa-like logs)
        self.save_tmp_wav = bool(config.get("save_tmp_wav", True))
        self.delete_audio_file = bool(delete_audio_file)

    async def open_audio_channels(self, conn):
        # Use the base threading/queue pipeline
        await super().open_audio_channels(conn)

    # IMPORTANT: do NOT override receive_audio — use ASRProviderBase.receive_audio

    async def speech_to_text(
        self, opus_data: List[bytes] | bytes | bytearray, session_id: str, audio_format: str = "opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Called by ASRProviderBase.handle_voice_stop() once per utterance.

        Steps:
          1) Decode Opus frames -> PCM16 mono@16k.
          2) Early-gate noise/too-short utterances.
          3) Wrap to WAV (optionally save tmp for logs).
          4) POST to remote WhisperX server (field 'file', fallback 'audio').
          5) Parse robustly; drop empty/noise results so base won't trigger LLM/TTS.
        """
        t0 = time.perf_counter()

        # ---- 1) Decode Opus -> PCM16
        pcm = bytearray()
        if isinstance(opus_data, (list, tuple)):
            for pkt in opus_data:
                if not pkt:
                    continue
                try:
                    # 960 samples ≈ 20ms at 16k mono (common frame size in your pipeline)
                    pcm.extend(self.decoder.decode(pkt, 960))
                except Exception:
                    # tolerate corrupt frames
                    pass
        elif isinstance(opus_data, (bytes, bytearray)):
            fmt = (audio_format or "").lower()
            if fmt in ("pcm", "pcm16", "wav"):
                # already PCM16 (or WAV bytes) — base normally doesn't send this, but support it
                pcm = bytearray(opus_data)
            else:
                # best-effort: treat entire buffer as one packet
                try:
                    pcm.extend(self.decoder.decode(bytes(opus_data), 960))
                except Exception:
                    pass
        else:
            logger.bind(tag=TAG).warning("Unsupported opus_data type; returning empty result")
            return "", None

        # ---- 2) Early noise/too-short gating
        dur_sec = self._pcm_duration_sec(bytes(pcm))
        if dur_sec < 0.35:  # tweak between 0.30–0.50s to taste
            logger.bind(tag=TAG).debug(f"Ignored short utterance ({dur_sec*1000:.0f} ms)")
            return "", None

        # ---- 3) Wrap PCM -> WAV (+ optional tmp save for sherpa-like logs)
        wav_bytes = self._pcm16_to_wav(bytes(pcm))
        tmp_path = None
        if self.save_tmp_wav:
            try:
                os.makedirs("tmp", exist_ok=True)
                tmp_path = f"tmp/asr_base_{session_id}_{uuid.uuid4()}.wav"
                t_save0 = time.perf_counter()
                with open(tmp_path, "wb") as f:
                    f.write(wav_bytes)
                logger.bind(tag=TAG).debug(
                    f"Saved audio to {tmp_path} in {time.perf_counter() - t_save0:.2f}s"
                )
            except Exception as e:
                logger.bind(tag=TAG).warning(f"Failed to save tmp wav: {e}")

        # ---- 4) POST to remote (create session bound to THIS loop)
        payload: Dict[str, Any] = {"text": "", "language": None, "num_segments": None}
        try:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = await self._post_wav_with_fallback(wav_bytes, session)
        except Exception as e:
            logger.bind(tag=TAG).error(f"Final remote transcription failed: {e}")
            payload = {"text": "", "language": None, "num_segments": None}

        text = (payload.get("text") or "").strip()

        # ---- 5) Log timing + cleanup tmp
        logger.bind(tag=TAG).debug(
            f"ASR decoded in {time.perf_counter() - t0:.2f}s:  "
            f"{ {'text': text, 'language': payload.get('language'), 'num_segments': payload.get('num_segments')} }"
        )
        if self.save_tmp_wav and tmp_path and self.delete_audio_file:
            try:
                os.remove(tmp_path)
                logger.bind(tag=TAG).debug(f"Deleted temp file: {tmp_path}")
            except Exception:
                pass

        # ---- 6) Final gating: drop empties/noise/zero segments so base won't trigger LLM/TTS
        if (not text) or self._is_trivial_text(text) or payload.get("num_segments") == 0:
            logger.bind(tag=TAG).debug("Ignored empty/noise transcript; returning empty so base won’t trigger LLM/TTS.")
            return "", None

        return text, None

    async def close(self):
        # Nothing persistent to close; base will handle its cleanup
        await super().close()

    def stop_ws_connection(self):
        # compatibility no-op (base calls this after ASR)
        pass

    # ---------------------------- Helpers ----------------------------

    def _pcm16_to_wav(self, pcm_bytes: bytes) -> bytes:
        """Wrap raw PCM16 (mono @ sample_rate) into a WAV container."""
        byte_rate = self.sample_rate * self.channels * (self.bits_per_sample // 8)
        block_align = self.channels * (self.bits_per_sample // 8)
        data_size = len(pcm_bytes)
        riff_size = 36 + data_size

        header = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
        fmt_chunk = (
            b"fmt " +
            struct.pack(
                "<IHHIIHH",
                16,                       # Subchunk1Size (PCM)
                1,                        # AudioFormat (PCM)
                self.channels,
                self.sample_rate,
                byte_rate,
                block_align,
                self.bits_per_sample,
            )
        )
        data_chunk = b"data" + struct.pack("<I", data_size)
        return header + fmt_chunk + data_chunk + pcm_bytes

    async def _post_wav_with_fallback(
        self, wav_bytes: bytes, session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """
        POST WAV to remote_url using the provided session (bound to current loop).
        Returns a dict: {"text": str, "language": str|None, "num_segments": int|None}
        """
        async def post(field_name: str) -> tuple[int, Dict[str, Any]]:
            form = aiohttp.FormData()
            form.add_field(field_name, wav_bytes, filename="audio.wav", content_type="audio/wav")
            # Optional metadata (server may ignore)
            form.add_field("sample_rate", str(self.sample_rate))
            form.add_field("channels", str(self.channels))
            form.add_field("bits_per_sample", str(self.bits_per_sample))

            async with session.post(self.remote_url, data=form) as resp:
                if resp.status == 200:
                    # Try JSON first (don’t call resp.text() before this)
                    try:
                        data = await resp.json(content_type=None)
                        text = (data.get("text") or data.get("transcript") or "").strip()
                        lang = data.get("language")
                        num_segments = data.get("num_segments")
                        if not text and "segments" in data:
                            segs = data.get("segments") or []
                            text = " ".join(s.get("text", "") for s in segs).strip()
                            num_segments = len(segs)
                        return 200, {"text": text, "language": lang, "num_segments": num_segments}
                    except Exception:
                        body = (await resp.text()).strip()
                        return 200, {"text": body, "language": None, "num_segments": None}
                else:
                    body = (await resp.text()).strip()
                    return resp.status, {"error": body}

        status, payload = await post("file")
        if status == 200:
            return payload

        # Fallback if API expects "audio"
        body = payload.get("error", "")
        if status == 422 and ("file" in body.lower() or "field required" in body.lower()):
            status2, payload2 = await post("audio")
            if status2 == 200:
                return payload2
            raise RuntimeError(f"HTTP {status2}: {payload2.get('error','')}")

        raise RuntimeError(f"HTTP {status}: {payload.get('error','')}")

    def _pcm_duration_sec(self, pcm_bytes: bytes) -> float:
        """Duration of PCM16 mono bytes."""
        # 2 bytes per sample at 16-bit mono
        return len(pcm_bytes) / (self.sample_rate * 2)

    def _is_trivial_text(self, text: str) -> bool:
        """Heuristic to filter fillers / very short noise."""
        if not text:
            return True
        t = text.strip().lower()
        if len(t) < 2:
            return True
        # Common fillers (extend as needed)
        fillers = {"ờ", "ừ", "ừm", "uh", "um", "hả", "hở", "à", "ạ", "vâng"}
        return t in fillers
