import os
import uuid
import edge_tts
from datetime import datetime
from langdetect import detect
from core.providers.tts.base import TTSProviderBase


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        # Load default and language-specific voices
        self.default_voice = config.get("private_voice") or config.get("voice")
        self.voice_map = {
            "en": config.get("voice_en", "en-US-JennyNeural"),
            "vi": config.get("voice_vi", "vi-VN-HoaiMyNeural"),
        }

        # Audio format
        self.audio_file_type = config.get("format", "mp3")
        self.speech_rate = config.get("rate", "+10%")  # Default to 1.1x

    def generate_filename(self, extension=".mp3"):
        return os.path.join(
            self.output_file,
            f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}",
        )

    async def text_to_speak(self, text, output_file):
        try:
            # Detect language
            language = detect(text)
            voice = self.voice_map.get(language, self.default_voice)

            # Create Communicate instance with rate
            communicate = edge_tts.Communicate(
                text, voice=voice, rate=self.speech_rate
            )

            if output_file:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "wb") as f:
                    pass

                with open(output_file, "ab") as f:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            f.write(chunk["data"])
            else:
                audio_bytes = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_bytes += chunk["data"]
                return audio_bytes

        except Exception as e:
            error_msg = f"Edge TTS request failed: {e}"
            raise Exception(error_msg)
