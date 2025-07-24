import time
import wave
import os
import sys
import io
from typing import Optional, Tuple, List
import numpy as np
import sherpa_onnx
from config.logger import setup_logging
from core.providers.asr.dto.dto import InterfaceType
from core.providers.asr.base import ASRProviderBase

TAG = __name__
logger = setup_logging()


class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self.output = self._output.getvalue()
        self._output.close()
        if self.output:
            logger.bind(tag=TAG).info(self.output.strip())


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        self.interface_type = InterfaceType.LOCAL
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")
        self.delete_audio_file = delete_audio_file

        os.makedirs(self.output_dir, exist_ok=True)

        tokens_path = os.path.join(self.model_dir, "tokens.txt")
        if not os.path.isfile(tokens_path):
            raise FileNotFoundError(f"Missing tokens.txt at {tokens_path}")

        self.tokens_path = tokens_path

        def find_model_file(substring: str):
            for f in os.listdir(self.model_dir):
                if substring in f.lower() and f.lower().endswith(".onnx"):
                    return os.path.join(self.model_dir, f)
            return None

        encoder_path = find_model_file("encoder")
        decoder_path = find_model_file("decoder")
        joiner_path = find_model_file("joiner")

        # Fallback to single model file if no 3-part model found
        single_model_path = None
        if not (encoder_path and decoder_path and joiner_path):
            for f in os.listdir(self.model_dir):
                if f.lower().endswith(".onnx"):
                    path = os.path.join(self.model_dir, f)
                    if not any(x in f.lower() for x in ["encoder", "decoder", "joiner"]):
                        single_model_path = path
                        break

        with CaptureOutput():
            if encoder_path and decoder_path and joiner_path:
                logger.bind(tag=TAG).info("Using 3-part Transducer model")
                self.model = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=encoder_path,
                    decoder=decoder_path,
                    joiner=joiner_path,
                    tokens=self.tokens_path,
                    num_threads=2,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    debug=False
                )
            elif single_model_path:
                logger.bind(tag=TAG).info(f"Using single .onnx model: {os.path.basename(single_model_path)}")
                self.model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=single_model_path,
                    tokens=self.tokens_path,
                    num_threads=2,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    debug=False
                )
            else:
                raise FileNotFoundError("No valid model found in model_dir")

    def read_wave(self, wave_filename: str) -> Tuple[np.ndarray, int]:
        with wave.open(wave_filename) as f:
            assert f.getnchannels() == 1, f.getnchannels()
            assert f.getsampwidth() == 2, f.getsampwidth()
            num_samples = f.getnframes()
            samples = f.readframes(num_samples)
            samples_int16 = np.frombuffer(samples, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768.0
            return samples_float32, f.getframerate()

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        file_path = None
        try:
            start_time = time.time()
            if audio_format == "pcm":
                pcm_data = opus_data
            else:
                pcm_data = self.decode_opus(opus_data)
            file_path = self.save_audio_to_file(pcm_data, session_id)
            logger.bind(tag=TAG).debug(f"Saved audio to {file_path} in {time.time() - start_time:.2f}s")

            start_time = time.time()
            s = self.model.create_stream()
            samples, sample_rate = self.read_wave(file_path)
            s.accept_waveform(sample_rate, samples)
            self.model.decode_stream(s)
            text = s.result.text
            logger.bind(tag=TAG).debug(f"ASR decoded in {time.time() - start_time:.2f}s: {text}")
            return text, file_path

        except Exception as e:
            logger.bind(tag=TAG).error(f"Speech recognition failed: {e}", exc_info=True)
            return "", file_path
        finally:
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.bind(tag=TAG).debug(f"Deleted temp file: {file_path}")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"File deletion failed: {file_path} | Error: {e}")