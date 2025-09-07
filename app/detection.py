import numpy as np
import cv2
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from pathlib import Path
import time

from app.config import settings
from app.utils import calculate_position, calculate_proximity, translate_class_name
import cv2.data as cvdata

logger = logging.getLogger(__name__)


class ObjectDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa o detector de objetos com YOLOv8.
        """
        self.model = None
        self.model_path = model_path or os.path.join(settings.models_dir, settings.yolo_model)
        self.last_detections = {}
        self.last_detection_time = {}
        self._load_model()
        self.glasses_cascade = None
        if getattr(settings, 'enable_glasses_detection', True):
            try:
                cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye_tree_eyeglasses.xml')
            except Exception:
                cascade_path = None
            if cascade_path and os.path.exists(cascade_path):
                self.glasses_cascade = cv2.CascadeClassifier(cascade_path)
    
    def _load_model(self):
        """
        Carrega o modelo YOLO.
        """
        try:
            os.makedirs(settings.models_dir, exist_ok=True)
            
            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.info(f"Baixando modelo YOLO: {settings.yolo_model}")
                self.model = YOLO(settings.yolo_model)
                
                if not model_file.parent.exists():
                    model_file.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Modelo baixado e salvo em: {self.model_path}")
            else:
                logger.info(f"Carregando modelo existente: {self.model_path}")
                self.model = YOLO(self.model_path)
            
            logger.info("Modelo YOLO carregado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo YOLO: {e}")
            raise RuntimeError(f"Falha ao carregar modelo YOLO: {str(e)}")
    
    def detect_objects(self, image: np.ndarray, 
                       confidence_threshold: Optional[float] = None,
                       iou_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Detecta objetos na imagem usando YOLO.
        """
        if self.model is None:
            raise RuntimeError("Modelo YOLO não está carregado")
        
        try:
            conf_threshold = confidence_threshold or settings.confidence_threshold
            iou_th = iou_threshold or getattr(settings, 'iou_threshold', 0.7)
            
            results = self.model(image, conf=conf_threshold, iou=iou_th, verbose=False)
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        class_id = int(box.cls)
                        class_name = self.model.names[class_id]
                        
                        confidence = float(box.conf)
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        x1, y1, x2, y2 = bbox
                        
                        height, width = image.shape[:2]
                        position = calculate_position((x1, y1, x2, y2), width)
                        proximity = calculate_proximity((x1, y1, x2, y2), width, height)
                        
                        detection = {
                            "class_name": class_name,
                            "confidence": round(confidence, 3),
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2)
                            },
                            "position": position,
                            "proximity": proximity
                        }
                        
                        detections.append(detection)
            
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

            if self.glasses_cascade is not None:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    eyes = self.glasses_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
                    for (ex, ey, ew, eh) in eyes[:2]:
                        x1, y1, x2, y2 = ex, ey, ex + ew, ey + eh
                        height, width = image.shape[:2]
                        position = calculate_position((x1, y1, x2, y2), width)
                        proximity = calculate_proximity((x1, y1, x2, y2), width, height)
                        detections.insert(0, {
                            "class_name": "glasses",
                            "confidence": 0.6,
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "position": position,
                            "proximity": proximity
                        })
                except Exception:
                    pass
            
            priority_detections = [d for d in detections 
                                  if d['class_name'].lower() in settings.priority_classes]
            other_detections = [d for d in detections 
                               if d['class_name'].lower() not in settings.priority_classes]
            
            final_detections = priority_detections + other_detections
            
            return final_detections[:settings.max_detections_per_frame]
            
        except Exception as e:
            logger.error(f"Erro durante detecção: {e}")
            raise RuntimeError(f"Falha na detecção de objetos: {str(e)}")
    
    def filter_repeated_detections(self, detections: List[Dict[str, Any]], 
                                  session_id: str = "default") -> List[Dict[str, Any]]:
        """
        Filtra detecções repetidas para evitar narração excessiva.
        """
        current_time = time.time()
        
        if session_id not in self.last_detection_time:
            self.last_detection_time[session_id] = 0
            self.last_detections[session_id] = []
        
        time_since_last = current_time - self.last_detection_time[session_id]
        
        if time_since_last < settings.min_detection_interval:
            new_detections = []
            for det in detections:
                is_new = True
                for last_det in self.last_detections[session_id]:
                    if (det['class_name'] == last_det['class_name'] and 
                        det['position'] == last_det['position'] and
                        det['proximity'] == last_det['proximity']):
                        is_new = False
                        break
                
                if is_new:
                    new_detections.append(det)
            
            if new_detections:
                self.last_detections[session_id] = detections
                self.last_detection_time[session_id] = current_time
                return new_detections
            else:
                return []
        else:
            self.last_detections[session_id] = detections
            self.last_detection_time[session_id] = current_time
            return detections
    
    def draw_detections(self, image: np.ndarray, 
                       detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Desenha as detecções na imagem.
        """
        img_copy = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            color = (0, 255, 0) if det['proximity'] == 'perto' else (0, 165, 255)
            if det['proximity'] == 'longe':
                color = (255, 0, 0)
            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            label = f"{translate_class_name(det['class_name'])} ({det['confidence']:.2f})"
            position_label = f"{det['position']} - {det['proximity']}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (w1, h1), _ = cv2.getTextSize(label, font, font_scale, thickness)
            (w2, h2), _ = cv2.getTextSize(position_label, font, font_scale, thickness)
            
            cv2.rectangle(img_copy, (x1, y1 - h1 - h2 - 10), (x1 + max(w1, w2) + 10, y1), color, -1)
            
            cv2.putText(img_copy, label, (x1 + 5, y1 - h2 - 5), 
                       font, font_scale, (255, 255, 255), thickness)
            cv2.putText(img_copy, position_label, (x1 + 5, y1 - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return img_copy
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo carregado.
        """
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": settings.yolo_model,
            "classes": len(self.model.names),
            "target_classes": settings.target_classes,
            "confidence_threshold": settings.confidence_threshold
        }


detector_instance = None

def get_detector() -> ObjectDetector:
    """
    Retorna uma instância singleton do detector.
    """
    global detector_instance
    if detector_instance is None:
        detector_instance = ObjectDetector()
    return detector_instance
