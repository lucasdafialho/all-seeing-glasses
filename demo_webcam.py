#!/usr/bin/env python3

import cv2
import numpy as np
import time
import threading
import argparse
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.detection import get_detector
from app.tts import get_improved_tts
from app.utils import format_detection_description
from app.config import settings

class WebcamDemo:
    def __init__(self, camera_id=0, show_fps=True, enable_tts=True):
        self.camera_id = camera_id
        self.show_fps = show_fps
        self.enable_tts = enable_tts
        
        print("Iniciando Sistema de Óculos Assistivos - Demo Webcam")
        print("=" * 60)
        
        self.detector = None
        self.tts = None
        self.cap = None
        
        self.running = False
        self.last_tts_time = 0
        self.tts_thread = None
        self.current_detections = []
        
        self._init_components()
    
    def _init_components(self):
        """Inicializa detector e TTS"""
        try:
            print("Carregando detector YOLO...")
            self.detector = get_detector()
            print(f"Modelo {settings.yolo_model} carregado com sucesso!")
            
            if self.enable_tts:
                print("Inicializando sistema TTS melhorado...")
                self.tts = get_improved_tts()
                status = self.tts.get_status()
                
                if self.tts.is_available():
                    available_engines = []
                    if status["google_tts"]:
                        available_engines.append("Google TTS (melhor qualidade)")
                    if status["piper_tts"]:
                        available_engines.append("Piper TTS (alta qualidade)")
                    if status["espeak_ng"]:
                        available_engines.append("eSpeak-NG")
                    elif status["espeak"]:
                        available_engines.append("eSpeak")
                    
                    print(f"TTS disponível com: {', '.join(available_engines)}")
                else:
                    print("TTS não disponível, funcionando sem áudio")
                    self.enable_tts = False
            
        except Exception as e:
            print(f"Erro ao inicializar componentes: {e}")
            sys.exit(1)
    
    def _init_camera(self):
        """Inicializa a câmera"""
        try:
            print(f"Conectando à câmera {self.camera_id}...")
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Não foi possível abrir a câmera {self.camera_id}")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"Câmera conectada: {width}x{height} @ {fps}fps")
            
        except Exception as e:
            print(f"Erro ao conectar câmera: {e}")
            sys.exit(1)
    
    def _speak_detections(self, detections):
        """Executa TTS em thread separada"""
        if not self.enable_tts or not self.tts or not detections:
            return
        
        current_time = time.time()
        if current_time - self.last_tts_time < settings.min_detection_interval:
            return
        
        def speak_worker():
            try:
                description = format_detection_description(detections)
                print(f"Falando: {description}")
                self.tts.speak_direct(description)
            except Exception as e:
                print(f"Erro no TTS: {e}")
        
        if self.tts_thread and self.tts_thread.is_alive():
            return
        
        self.tts_thread = threading.Thread(target=speak_worker, daemon=True)
        self.tts_thread.start()
        self.last_tts_time = current_time
    
    def _draw_info_panel(self, frame, detections, fps=0):
        """Desenha painel de informações na tela"""
        height, width = frame.shape[:2]
        
        panel_height = 120
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        title = f"Sistema de Oculos Assistivos - {datetime.now().strftime('%H:%M:%S')}"
        cv2.putText(panel, title, (10, 25), font, 0.7, (255, 255, 255), 2)
        
        if self.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(panel, fps_text, (width - 120, 25), font, 0.6, (0, 255, 0), 2)
        
        status = f"Detectados: {len(detections)} objetos"
        cv2.putText(panel, status, (10, 50), font, 0.6, (255, 255, 255), 2)
        
        model_info = f"Modelo: {settings.yolo_model} | Conf: {settings.confidence_threshold} | IoU: {getattr(settings, 'iou_threshold', 0.8)}"
        cv2.putText(panel, model_info, (10, 75), font, 0.5, (200, 200, 200), 1)
        
        controls = "ESC: Sair | SPACE: TTS On/Off | T: Mostrar/Ocultar deteccoes"
        cv2.putText(panel, controls, (10, 100), font, 0.4, (255, 255, 0), 1)
        
        return np.vstack([panel, frame])
    
    def _draw_detection_list(self, frame, detections):
        """Desenha lista de detecções no lado direito"""
        if not detections:
            return frame
        
        height, width = frame.shape[:2]
        list_width = 300
        
        list_panel = np.zeros((height, list_width, 3), dtype=np.uint8)
        list_panel[:] = (30, 30, 30)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(list_panel, "DETECCOES:", (10, 30), font, 0.6, (255, 255, 255), 2)
        
        y_offset = 60
        for i, det in enumerate(detections[:8]):
            from app.utils import translate_class_name
            class_name = translate_class_name(det['class_name'])
            confidence = det['confidence']
            position = det['position']
            proximity = det['proximity']
            
            text = f"{class_name} ({confidence:.2f})"
            pos_text = f"{position} - {proximity}"
            
            color = (0, 255, 0) if proximity == 'perto' else (0, 165, 255)
            if proximity == 'longe':
                color = (255, 0, 0)
            
            cv2.putText(list_panel, text, (10, y_offset), font, 0.5, color, 1)
            cv2.putText(list_panel, pos_text, (10, y_offset + 20), font, 0.4, (200, 200, 200), 1)
            
            y_offset += 50
        
        return np.hstack([frame, list_panel])
    
    def run(self):
        """Executa a demonstração"""
        self._init_camera()
        
        print("\nDEMONSTRAÇÃO INICIADA")
        print("Controles:")
        print("  ESC - Sair")
        print("  SPACE - Ativar/Desativar TTS")
        print("  T - Mostrar/Ocultar detecções visuais")
        print("  R - Reset do histórico de detecções")
        print("-" * 60)
        
        self.running = True
        show_detections = True
        
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Erro ao ler frame da câmera")
                    break
                
                start_time = time.time()
                
                try:
                    detections = self.detector.detect_objects(
                        frame,
                        confidence_threshold=getattr(settings, 'confidence_threshold', 0.45),
                        iou_threshold=getattr(settings, 'iou_threshold', 0.8)
                    )
                    
                    filtered_detections = self.detector.filter_repeated_detections(
                        detections, "demo_session"
                    )
                    
                    if filtered_detections:
                        self.current_detections = filtered_detections
                        self._speak_detections(filtered_detections)
                        
                        print(f"Novas detecções: {len(filtered_detections)} objetos")
                        for det in filtered_detections[:3]:
                            print(f"   • {det['class_name']} ({det['confidence']:.2f}) - {det['position']}, {det['proximity']}")
                    
                    if show_detections and detections:
                        frame = self.detector.draw_detections(frame, detections)
                    
                except Exception as e:
                    print(f"Erro na detecção: {e}")
                    detections = []
                
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()
                
                frame_with_panel = self._draw_info_panel(frame, detections, current_fps)
                frame_final = self._draw_detection_list(frame_with_panel, detections)
                
                cv2.imshow('Sistema de Oculos Assistivos - Demo', frame_final)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Encerrando demonstração...")
                    break
                elif key == ord(' '):  # SPACE
                    self.enable_tts = not self.enable_tts
                    status = "ativado" if self.enable_tts else "desativado"
                    print(f"TTS {status}")
                elif key == ord('t') or key == ord('T'):
                    show_detections = not show_detections
                    status = "ativadas" if show_detections else "desativadas"
                    print(f"Detecções visuais {status}")
                elif key == ord('r') or key == ord('R'):
                    self.detector.last_detections = {}
                    self.detector.last_detection_time = {}
                    print("Histórico de detecções resetado")
                
                processing_time = time.time() - start_time
                
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        except Exception as e:
            print(f"Erro durante execução: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Limpa recursos"""
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2)
        
        print("Recursos liberados")
        print("Demonstração encerrada!")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstração do Sistema de Óculos Assistivos com Webcam"
    )
    parser.add_argument(
        "--camera", "-c", type=int, default=0,
        help="ID da câmera (padrão: 0)"
    )
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Desativar Text-to-Speech"
    )
    parser.add_argument(
        "--no-fps", action="store_true",
        help="Não mostrar FPS"
    )
    parser.add_argument(
        "--confidence", "-conf", type=float, default=0.35,
        help="Threshold de confiança (padrão: 0.35)"
    )
    
    args = parser.parse_args()
    
    if args.confidence:
        settings.confidence_threshold = args.confidence
    
    demo = WebcamDemo(
        camera_id=args.camera,
        show_fps=not args.no_fps,
        enable_tts=not args.no_tts
    )
    
    try:
        demo.run()
    except Exception as e:
        print(f"Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
