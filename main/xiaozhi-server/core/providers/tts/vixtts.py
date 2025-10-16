import os
import torch
import torchaudio
import subprocess
from datetime import datetime
from unidecode import unidecode
from underthesea import sent_tokenize
from vinorm import TTSnorm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging
from tqdm import tqdm
TAG = __name__
logger = setup_logging()

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None

class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.model_dir = config.get("model_dir", "models/viXTTS")
        self.reference_audio = config.get("ref_audio","models/viXTTS/vi_sample.wav")  
        self.use_deepfilter = config.get("use_deepfilter", False)
        self.normalize_text = config.get("normalize_text", True)
        self.lang = config.get("lang", "vi")
        self.output_file = config.get("output_dir","models/viXTTS/output")
        self.xtts_model = None
        
        self._load_model()

    def _load_model(self):
        global XTTS_MODEL
        clear_gpu_cache()
        required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
        files_in_dir = os.listdir(self.model_dir)
        if not all(file in files_in_dir for file in required_files):
            raise Exception("Missing required model files in model_dir")
        config_path = os.path.join(self.model_dir, "config.json")
        config = XttsConfig()
        config.load_json(config_path)
        if XTTS_MODEL is None:
            XTTS_MODEL = Xtts.init_from_config(config)
            XTTS_MODEL.load_checkpoint(config, checkpoint_dir=self.model_dir, use_deepspeed=False)
            if torch.cuda.is_available():
                XTTS_MODEL.cuda()
        self.xtts_model = XTTS_MODEL

    def _normalize_text(self, text):
        return (
            TTSnorm(text, unknown=False, lower=False, rule=True)
            .replace("..", ".")
            .replace("!.", "!")
            .replace("?.", "?")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace('"', "")
            .replace("'", "")
            .replace("AI", "Ây Ai")
            .replace("A.I", "Ây Ai")
        )
    
    def calculate_keep_len(self, text, lang):
        """Simple hack for short sentences"""
        if lang in ["ja", "zh-cn"]:
            return -1

        word_count = len(text.split())
        num_punct = text.count(".") + text.count("!") + text.count("?") + text.count(",")

        if word_count < 5:
            return 15000 * word_count + 2000 * num_punct
        elif word_count < 10:
            return 13000 * word_count + 2000 * num_punct
        return -1
    
    async def text_to_speak(self, text, output_file):
        if self.normalize_text and self.lang == "vi":
            text = self._normalize_text(text)

        # fallback output_file
        output_path = output_file or self.generate_filename(extension=".wav")

        speaker_audio_file = self.reference_audio
        speaker_audio_key = speaker_audio_file
        if not hasattr(self, 'conditioning_latents_cache'):
            self.conditioning_latents_cache = {}
        conditioning_latents_cache = self.conditioning_latents_cache
        cache_key = (
            speaker_audio_key,
            XTTS_MODEL.config.gpt_cond_len,
            XTTS_MODEL.config.max_ref_len,
            XTTS_MODEL.config.sound_norm_refs,
        )
        if cache_key in conditioning_latents_cache:
            print("Using conditioning latents cache...")
            gpt_cond_latent, speaker_embedding = conditioning_latents_cache[cache_key]
        else:
            print("Computing conditioning latents...")
            gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
                audio_path=speaker_audio_file,
                gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
                max_ref_length=XTTS_MODEL.config.max_ref_len,
                sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
            )
            conditioning_latents_cache[cache_key] = (gpt_cond_latent, speaker_embedding)

        sentences = sent_tokenize(text)
        wav_chunks = []
        for sentence in tqdm(sentences):
            if not sentence.strip():
                continue
            wav_chunk = self.xtts_model.inference(
                text=sentence,
                language=self.lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.3,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=30,
                top_p=0.85,
                enable_text_splitting=True
                
            )
            keep_len = self.calculate_keep_len(sentence, self.lang)
            wav_chunk["wav"] = wav_chunk["wav"][:keep_len]

            wav_chunks.append(torch.tensor(wav_chunk["wav"]))
         self.audio_file_type = "wav"
        logger.bind(tag=TAG).warning(f"[Output tts] {output_path}")
        out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
        torchaudio.save(output_path, out_wav, 24000)
        print(f"Saved final file to {output_path}")
        if self.output_file:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes
        else:
            with open(output_path, "rb") as f:
                audio_bytes = f.read()
            return audio_bytes
                
        # return "Speech generated !"
