import wave
import io
from core.utils.util import check_model_key
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging

from google import genai
from google.genai import types
import base64

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.model = config.get("model", "gemini-2.5-flash-preview-tts")
        self.voice = config.get("voice", "Kore")
        self.output_file = config.get("output_dir", "tmp/")
        self.sample_rate = config.get("sample_rate", 24000)
        self.channels = config.get("channels", 1)
        self.sample_width = config.get("sample_width", 2)
        self.api_key = config.get("api_key")
        model_key_msg = check_model_key("TTS", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)
        self.client = genai.Client(api_key=self.api_key)

    def _get_gemini_tts_pcm(self, text):
        """
        Make Gemini TTS API call and return decoded PCM bytes.
        Raises Exception on error.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"Say: {text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice,
                        )
                    )
                ),
            )
        )
        if not response.candidates or not response.candidates[0].content or not hasattr(response.candidates[0].content, 'parts') or not response.candidates[0].content.parts:
            logger.bind(tag=TAG).error(f"Gemini TTS API did not return audio. Full response: {response}")
            raise Exception("Gemini TTS API did not return audio. Check quota, API key, or input.")
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        audio_bytes = base64.b64decode(audio_data)
        return audio_bytes

    async def text_to_speak(self, text, output_file=None, frame_ms=60, frame_callback=None):
        """
        Generate TTS audio and process PCM in frames. If frame_callback is provided, call it for each frame.
        If output_file is provided, save as WAV. Otherwise, return list of PCM frames.
        """
        try:
            audio_bytes = self._get_gemini_tts_pcm(text)
            # If a frame_callback is provided, process PCM in frames and call the callback
            if frame_callback:
                for frame in self.process_pcm_frames(audio_bytes, frame_ms=frame_ms):
                    frame_callback(frame)
                return None
            # Save as WAV using the correct wave file writer
            if output_file:
                logger.bind(tag=TAG).info(f"PCM length: {len(audio_bytes)}, sample_rate: {self.sample_rate}, channels: {self.channels}, sample_width: {self.sample_width}")
                self.pcm_to_wav(audio_bytes, output_file=output_file)
                return None
            else:
                # Return list of PCM frames for further processing
                return list(self.process_pcm_frames(audio_bytes, frame_ms=frame_ms))
        except Exception as e:
            logger.bind(tag=TAG).error(f"Gemini TTS request failed: {str(e)}")
            raise Exception(f"Gemini TTS request failed: {str(e)}")

    def process_pcm_frames(self, pcm_data, frame_ms=60):
        """
        Process PCM data in frames (default 60ms, 16kHz, 1ch = 1920 bytes per frame).
        Yields each frame for further processing (e.g., Opus encoding or streaming).
        """
        frame_bytes = int(self.sample_rate * self.channels * frame_ms / 1000 * self.sample_width)
        total_len = len(pcm_data)
        for i in range(0, total_len, frame_bytes):
            frame = pcm_data[i : i + frame_bytes]
            if len(frame) < frame_bytes:
                # Pad last frame if needed
                frame = frame + b"\x00" * (frame_bytes - len(frame))
            yield frame

    def to_tts(self, text: str, output_file=None) -> bytes:
        """
        Non-streaming TTS: returns WAV bytes (if output_file is None) or saves to file.
        """
        import time
        start_time = time.time()
        try:
            audio_bytes = self._get_gemini_tts_pcm(text)
            logger.info(f"Gemini TTS request success: {text}, time: {time.time() - start_time}s, PCM length: {len(audio_bytes)}")
            wav_bytes = self.pcm_to_wav(audio_bytes, output_file=output_file)
            if output_file:
                return b""  # Return empty bytes if saved to file
            if not isinstance(wav_bytes, (bytes, bytearray)):
                logger.bind(tag=TAG).error("TTS output is not bytes. Check pcm_to_wav return value.")
                return b""
            return wav_bytes
        except Exception as e:
            logger.bind(tag=TAG).error(f"Gemini TTS request exception: {e}")
            return b""

    def to_tts_single_stream(self, text, is_last=False):
        try:
            max_repeat_time = 5
            text = MarkdownCleaner.clean_markdown(text)
            while max_repeat_time > 0:
                try:
                    # Get PCM frames from Gemini
                    pcm_frames = asyncio.run(self.text_to_speak(text))
                    if not pcm_frames:
                        raise Exception("No PCM frames returned from Gemini TTS")
                    # Encode and queue
                    for frame in pcm_frames:
                        opus = self.opus_encoder.encode_pcm_to_opus(frame, end_of_stream=False)
                        if opus:
                            self.tts_audio_queue.put((SentenceType.MIDDLE, opus, None))
                    break
                except Exception as e:
                    logger.bind(tag=TAG).warning(
                        f"Gemini TTS generation failed {6 - max_repeat_time} times: {text}, error: {e}"
                    )
                    max_repeat_time -= 1
            if max_repeat_time > 0:
                logger.bind(tag=TAG).info(
                    f"Gemini TTS generation succeeded: {text}, retries: {5 - max_repeat_time}"
                )
            else:
                logger.bind(tag=TAG).error(
                    f"Gemini TTS generation failed: {text}, please check network or service"
                )
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to generate Gemini TTS file: {e}")
        finally:
            return None

    def pcm_to_wav(self, pcm_bytes, output_file=None):
        """
        Convert PCM bytes to WAV format.
        If output_file is provided, save to file.
        Otherwise, return WAV bytes.
        """
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)
        wav_bytes = buffer.getvalue()
        if output_file:
            with open(output_file, "wb") as f:
                f.write(wav_bytes)
            return None
        if not isinstance(wav_bytes, (bytes, bytearray)):
            logger.bind(tag=TAG).error("pcm_to_wav did not return bytes.")
            return b""
        return wav_bytes
