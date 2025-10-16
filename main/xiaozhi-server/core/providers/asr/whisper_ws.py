import asyncio
import numpy as np
import wave
import json
import logging
import os
import aiofiles
import socket
from datetime import datetime

from core.providers.asr.dto.dto import InterfaceType
from core.providers.asr.base import ASRProviderBase

logger = logging.getLogger(__name__)

class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        self.interface_type = InterfaceType.LOCAL
        self.delete_audio_file = delete_audio_file
        self.server_addr = config.get("server_addr", "ws://localhost:43007")
        self.seconds_per_message = float(config.get("seconds_per_message", 0.1))
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

    async def tcp_asr(self, wav_path: str) -> str:
        server_host, server_port = self.server_addr.replace("ws://", "").split(":")
        server_port = int(server_port)
        result_lines = []

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((server_host, server_port))
                logger.info(f"Connected to TCP server at {server_host}:{server_port}")

                with open(wav_path, 'rb') as f:
                    while chunk := f.read(2048):
                        sock.sendall(chunk)

                sock.shutdown(socket.SHUT_WR)

                buffer = b""
                while True:
                    data = sock.recv(1024)
                    if not data:
                        break
                    buffer += data
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        line_str = line.decode("utf-8").strip()
                        if line_str:
                            # Remove timestamp prefixes like '01 02 '
                            parts = line_str.split(maxsplit=2)
                            if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
                                cleaned = parts[2]
                            else:
                                cleaned = line_str
                            logger.info(f"Received: {cleaned}")
                            result_lines.append(cleaned)

                if not result_lines:
                    logger.warning("No transcript lines received. Consider adjusting server to flush incomplete output.")

        except Exception as e:
            logger.error(f"TCP communication failed: {e}", exc_info=True)

        return "\n".join(result_lines)

    async def speech_to_text(self, opus_data, session_id, audio_format="opus"):
        file_path = None
        try:
            wav_path = self.save_audio_to_file(self.decode_opus(opus_data), session_id)
            file_path = wav_path
            text = await self.tcp_asr(wav_path)
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
