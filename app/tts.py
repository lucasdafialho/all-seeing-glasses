import logging
import tempfile
import os
from typing import Optional
import subprocess
import shutil
from pathlib import Path
import threading
import time
import io

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)


class ImprovedTTS:
    def __init__(self):
        self.gtts_available = GTTS_AVAILABLE
        self.pygame_initialized = False
        self.piper_available = False
        self.last_audio_file = None
        
        self._initialize_pygame()
        self._check_piper()
        
        logger.info("TTS melhorado inicializado")
    
    def _initialize_pygame(self):
        if not self.gtts_available:
            return
            
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.pygame_initialized = True
            logger.info("Pygame mixer inicializado para reprodução de áudio")
        except Exception as e:
            logger.warning(f"Erro ao inicializar pygame: {e}")
            self.pygame_initialized = False
    
    def _check_piper(self):
        if shutil.which("piper"):
            model_path = Path("models/pt_BR-faber-medium.onnx")
            if model_path.exists():
                self.piper_available = True
                logger.info("Piper TTS encontrado com modelo português")
            else:
                logger.info("Piper encontrado, mas modelo PT-BR não disponível")
    
    def _speak_with_gtts(self, text: str) -> bool:
        if not self.gtts_available or not self.pygame_initialized:
            return False
        
        try:
            tts = gTTS(text=text, lang='pt-br', slow=False)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                audio_path = tmp_file.name
            
            tts.save(audio_path)
            
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            if self.last_audio_file and os.path.exists(self.last_audio_file):
                try:
                    os.unlink(self.last_audio_file)
                except:
                    pass
            
            self.last_audio_file = audio_path
            logger.info(f"Áudio reproduzido com Google TTS: {text[:30]}...")
            return True
            
        except Exception as e:
            logger.error(f"Erro com Google TTS: {e}")
            return False
    
    def _speak_with_piper(self, text: str) -> bool:
        if not self.piper_available:
            return False
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                audio_path = tmp_file.name
            
            model_path = Path("models/pt_BR-faber-medium.onnx")
            
            cmd = [
                "piper",
                "--model", str(model_path),
                "--output_file", audio_path
            ]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text, timeout=10)
            
            if process.returncode == 0:
                if self.pygame_initialized:
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                else:
                    subprocess.run(["aplay", audio_path], check=True, timeout=10)
                
                os.unlink(audio_path)
                logger.info(f"Áudio reproduzido com Piper: {text[:30]}...")
                return True
            else:
                logger.error(f"Piper falhou: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Erro com Piper TTS: {e}")
            return False
    
    def _speak_with_espeak_ng(self, text: str) -> bool:
        engines = ["espeak-ng", "espeak"]
        
        for engine in engines:
            if not shutil.which(engine):
                continue
                
            try:
                cmd = [engine, "-v", "pt+f3", "-s", "160", "-a", "100", text]
                result = subprocess.run(cmd, timeout=10, check=True)
                logger.info(f"Áudio reproduzido com {engine}")
                return True
            except Exception as e:
                logger.warning(f"{engine} falhou: {e}")
                continue
        
        return False
    
    def speak_direct(self, text: str) -> bool:
        if not text:
            return False
        
        text = text.strip()
        if not text:
            return False
        
        logger.info(f"Reproduzindo: {text}")
        
        methods = [
            ("Google TTS", self._speak_with_gtts),
            ("Piper TTS", self._speak_with_piper),
            ("espeak-ng", self._speak_with_espeak_ng)
        ]
        
        for method_name, method_func in methods:
            try:
                if method_func(text):
                    return True
            except Exception as e:
                logger.error(f"Erro em {method_name}: {e}")
                continue
        
        logger.error("Todas as opções de TTS falharam")
        return False

    def generate_audio(self, text: str) -> Optional[bytes]:
        if not text:
            return None
        
        text = text.strip()
        if not text:
            return None

        logger.info(f"Gerando áudio para: {text}")

        if self.gtts_available:
            try:
                tts = gTTS(text=text, lang='pt-br', slow=False)
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                audio_fp.seek(0)
                return audio_fp.read()
            except Exception as e:
                logger.error(f"Erro ao gerar áudio com gTTS: {e}")

        if self.piper_available:
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp_file:
                    audio_path = tmp_file.name
                
                model_path = Path("models/pt_BR-faber-medium.onnx")
                cmd = ["piper", "--model", str(model_path), "--output_file", audio_path]
                
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True)
                process.communicate(input=text, timeout=10)

                if process.returncode == 0:
                    with open(audio_path, 'rb') as f:
                        audio_data = f.read()
                    os.unlink(audio_path)
                    return audio_data
                else:
                     logger.error("Piper falhou ao gerar áudio.")

            except Exception as e:
                logger.error(f"Erro ao gerar áudio com Piper: {e}")
        
        logger.error("Todas as opções de geração de áudio falharam.")
        return None
    
    def is_available(self) -> bool:
        return (self.gtts_available or 
                self.piper_available or 
                shutil.which("espeak-ng") or 
                shutil.which("espeak"))
    
    def get_status(self) -> dict:
        return {
            "google_tts": self.gtts_available and self.pygame_initialized,
            "piper_tts": self.piper_available,
            "espeak_ng": shutil.which("espeak-ng") is not None,
            "espeak": shutil.which("espeak") is not None,
            "pygame_mixer": self.pygame_initialized
        }


_improved_tts = None

def get_improved_tts():
    global _improved_tts
    if _improved_tts is None:
        _improved_tts = ImprovedTTS()
    return _improved_tts
