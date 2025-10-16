import os
import uuid
import asyncio
from elevenlabs.client import AsyncElevenLabs
from core.utils.util import check_model_key
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.api_key = config.get("api_key")
        self.voice_id = config.get("voice_id", "HQZkBNMmZF5aISnrU842")
        self.model_id = config.get("model_id", "eleven_flash_v2_5")
        self.stability = float(config.get("stability", 0.5))
        self.similarity_boost = float(config.get("similarity_boost", 0.5))
        self.output_dir = config.get("output_dir", "tmp/")
        model_key_msg = check_model_key("TTS", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)

        self.client = AsyncElevenLabs(api_key=self.api_key)

    async def text_to_speak(self, text, output_file):
        stream =  self.client.text_to_speech.stream(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="mp3_44100_32",  # Use 16000 for better compatibility
            voice_settings={
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
            }
        )

        self.audio_file_type = "mp3"

        # Collect audio bytes from async stream 
        audio_bytes = b""
        async for chunk in stream:
            if isinstance(chunk, bytes):
                audio_bytes += chunk
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
            return None
        return audio_bytes
