import os
import time
import wave
import io
import sys
import uuid
import psutil
import shutil
import numpy as np
from typing import Optional, Tuple, List
from config.logger import setup_logging
from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType

from faster_whisper import WhisperModel

TAG = __name__
logger = setup_logging()

MAX_RETRIES = 2
RETRY_DELAY = 1  # seconds
MIN_MEMORY_BYTES = 2 * 1024 * 1024 * 1024  # 2GB


# Capture stdout during model loading
class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        output = self._output.getvalue()
        self._output.close()
        if output:
            logger.bind(tag=TAG).info(output.strip())


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        self.interface_type = InterfaceType.LOCAL
        self.delete_audio_file = delete_audio_file
        self.model_dir = config.get("model_dir", "large-v3")
        self.output_dir = config.get("output_dir", "/tmp/asr")
        os.makedirs(self.output_dir, exist_ok=True)

        # Memory check
        total_mem = psutil.virtual_memory().total
        if total_mem < MIN_MEMORY_BYTES:
            logger.bind(tag=TAG).error(f"Insufficient memory (<2GB). Available: {total_mem / (1024*1024):.2f} MB")

        # Load Whisper model
        try:
            with CaptureOutput():
                self.model = WhisperModel(
                    self.model_dir,
                    device="cuda",
                    compute_type="int8",
                    download_root=self.model_dir if os.path.isdir(self.model_dir) else None,
                )
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise

    def read_wave(self, wave_filename: str) -> Tuple[np.ndarray, int]:
        with wave.open(wave_filename, "rb") as f:
            assert f.getnchannels() == 1
            assert f.getsampwidth() == 2
            sample_rate = f.getframerate()
            frames = f.readframes(f.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, sample_rate

    def get_audio_duration(self, wave_filename: str) -> float:
        with wave.open(wave_filename, "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        file_path = None
        retry_count = 0

        while retry_count < MAX_RETRIES:
            try:
                # Decode audio
                if audio_format == "pcm":
                    pcm_data = opus_data
                else:
                    pcm_data = self.decode_opus(opus_data)

                combined_pcm_data = b"".join(pcm_data)

                # Check disk space
                if not self.delete_audio_file:
                    free_space = shutil.disk_usage(self.output_dir).free
                    if free_space < len(combined_pcm_data) * 2:
                        raise OSError("Insufficient disk space")

                # Save to file
                file_path = self.save_audio_to_file(pcm_data, session_id)
                logger.bind(tag=TAG).debug(f"Saved audio to {file_path}")

                # Check audio duration
                duration = self.get_audio_duration(file_path)
                if duration < 0.5:
                    logger.bind(tag=TAG).warning(f"Audio too short ({duration:.2f}s), skipping transcription.")
                    return "", file_path

                # Transcription
                start = time.time()
                segments, info = self.model.transcribe(
                    file_path,
                    language="vi",
                    beam_size=5,
                    compute_type="float16",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=200)
                )
                segments_list = list(segments)
                if not segments_list:
                    logger.bind(tag=TAG).warning("No segments returned from Whisper.")
                    return "", file_path

                text = "".join(segment.text for segment in segments_list).strip()
                if not text:
                    logger.bind(tag=TAG).warning("Whisper transcribed empty text.")
                    return "", file_path

                logger.bind(tag=TAG).debug(f"Transcribed in {time.time() - start:.2f}s | Text: {text}")
                return text, file_path

            except OSError as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    logger.bind(tag=TAG).error(f"Transcription failed after {retry_count} retries: {e}", exc_info=True)
                    return "", file_path
                logger.bind(tag=TAG).warning(f"Retrying Whisper ASR ({retry_count}/{MAX_RETRIES}) due to error: {e}")
                time.sleep(RETRY_DELAY)

            except Exception as e:
                logger.bind(tag=TAG).error(f"Whisper ASR failed: {e}", exc_info=True)
                return "", file_path

            finally:
                if self.delete_audio_file and file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.bind(tag=TAG).debug(f"Deleted audio file: {file_path}")
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"Failed to delete audio file: {file_path} | Error: {e}")