import asyncio
import websockets
import numpy as np
import wave
import json
import logging
import os

from core.providers.asr.dto.dto import InterfaceType
from core.providers.asr.base import ASRProviderBase

logger = logging.getLogger(__name__)

class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        self.interface_type = InterfaceType.LOCAL
        self.delete_audio_file = delete_audio_file

        self.server_addr = config.get("server_addr", "localhost")
        self.samples_per_message = config.get("samples_per_message", 8000)
        self.seconds_per_message = config.get("seconds_per_message", 0.1)
        self.output_dir = config.get("output_dir", "/tmp")
        os.makedirs(self.output_dir, exist_ok=True)

    def read_wave(self, wave_filename: str) -> np.ndarray:
        with wave.open(wave_filename) as f:
            assert f.getframerate() == 16000, f.getframerate()
            assert f.getnchannels() == 1, f.getnchannels()
            assert f.getsampwidth() == 2, f.getsampwidth()
            num_samples = f.getnframes()
            samples = f.readframes(num_samples)
            samples_int16 = np.frombuffer(samples, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768
            return samples_float32

    async def websocket_asr(self, wav_path: str) -> str:
        data = self.read_wave(wav_path)
        result = ""
        total_bytes = data.nbytes
        sample_rate = 16000

        # Construct header
        header = (
            sample_rate.to_bytes(4, "little", signed=True) +
            total_bytes.to_bytes(4, "little", signed=True)
        )

        async with websockets.connect(f"{self.server_addr}") as websocket:
            # Start result receiver
            async def receive_results(socket):
                last_message = ""
                async for message in socket:
                    if message != "Done!":
                        last_message = message
                        try:
                            logger.info(f"WS Partial: {json.loads(message)}")
                        except Exception:
                            logger.info(f"WS Partial (raw): {message}")
                    else:
                        break
                return last_message

            receive_task = asyncio.create_task(receive_results(websocket))

            # Send header + first chunk
            chunk0 = data[:self.samples_per_message]
            await websocket.send(header + chunk0.tobytes())
            start = self.samples_per_message

            # Send remaining audio
            while start < data.shape[0]:
                end = min(start + self.samples_per_message, data.shape[0])
                await websocket.send(data[start:end].tobytes())
                await asyncio.sleep(self.seconds_per_message)
                start = end

            await websocket.send("Done")
            decoding_results = await receive_task
            logger.info(f"WS Final result: {decoding_results}")

            try:
                return json.loads(decoding_results).get("text", decoding_results)
            except Exception:
                return decoding_results

    async def speech_to_text(self, opus_data, session_id, audio_format="opus"):
        file_path = None
        try:
            wav_path = self.save_audio_to_file(self.decode_opus(opus_data), session_id)
            file_path = wav_path  # for deletion later
            text = await self.websocket_asr(wav_path)
            return text, wav_path
        except Exception as e:
            logger.error(f"Speech recognition failed: {e}", exc_info=True)
            return "", file_path
        finally:
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"File deletion failed: {file_path} | Error: {e}")
