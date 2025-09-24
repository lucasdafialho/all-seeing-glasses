from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    app_name: str = "Sistema de Óculos Assistivos"
    version: str = "1.0.0"
    
    host: str = "0.0.0.0"
    port: int = 8000
    
    yolo_model: str = "yolov8s.pt"
    models_dir: str = "models"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.8
    
    target_classes: List[str] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    priority_classes: List[str] = [
        "person", "dog", "cat", "car", "bicycle", "motorcycle", "traffic light", 
        "cell phone", "laptop", "tv", "bottle", "cup", "chair", "couch"
    ]
    
    tts_engine: str = "pyttsx3"
    tts_rate: int = 140
    tts_volume: float = 1.0
    tts_voice_index: int = 0
    
    min_detection_interval: float = 1.0
    
    max_detections_per_frame: int = 2
    enable_glasses_detection: bool = True
    
    proximity_thresholds: dict = {
        "near": 0.3,
        "medium": 0.15,
        "far": 0.0
    }
    
    position_thresholds: dict = {
        "left": 0.33,
        "center": 0.66,
        "right": 1.0
    }
    
    cors_origins: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
