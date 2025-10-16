# main/xiaozhi-server/core/providers/tts/coqui_tts.py
import os
import uuid
from datetime import datetime
from core.providers.tts.base import TTSProviderBase
import torch
from TTS.api import TTS

class CoquiTTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.voice = config.get("voice")  # For single speaker / default voice
        self.model_name = config.get("model_name")
        self.audio_file_type = config.get("format", "wav")  # Default to wav

        # Initialize TTS with the specified model
        self.tts = TTS(model_name=self.model_name, progress_bar=False).to(self.device)

    def generate_filename(self, extension=".wav"):
        return os.path.join(
            self.output_file,
            f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}",
        )

    async def text_to_speak(self, text, output_file):
        try:
            if output_file:
                if self.model_name.startswith("tts_models/multilingual"):
                    # Multi-speaker model
                    speaker_wav = self.config.get("speaker_wav")
                    language = self.config.get("language", "vie")  # Default to English
                    self.tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=output_file)
                else:
                    # Single speaker model
                    self.tts.tts_to_file(text=text, file_path=output_file)
            else:
                # Return audio data as bytes
                audio_bytes = self.tts.tts(text=text)
                return audio_bytes
        except Exception as e:
            error_msg = f"Coqui TTS request failed: {e}"
            raise Exception(error_msg)

    def voice_conversion(self, source_wav, target_wav, output_file):
        """Converts the voice in source_wav to the target_wav voice."""
        try:
            # Use the voice conversion model
            tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(self.device)
            tts.voice_conversion_to_file(source_wav=source_wav, target_wav=target_wav, file_path=output_file)
        except Exception as e:
            error_msg = f"Voice conversion failed: {e}"
            raise Exception(error_msg)