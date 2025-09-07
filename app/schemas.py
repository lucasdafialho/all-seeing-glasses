from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import base64


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Coordenada X do canto superior esquerdo")
    y1: float = Field(..., description="Coordenada Y do canto superior esquerdo")
    x2: float = Field(..., description="Coordenada X do canto inferior direito")
    y2: float = Field(..., description="Coordenada Y do canto inferior direito")


class Detection(BaseModel):
    class_name: str = Field(..., description="Nome da classe detectada")
    confidence: float = Field(..., ge=0, le=1, description="Confiança da detecção")
    bbox: BoundingBox = Field(..., description="Caixa delimitadora")
    position: str = Field(..., description="Posição relativa (esquerda, centro, direita)")
    proximity: str = Field(..., description="Proximidade estimada (perto, médio, longe)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "class_name": "person",
                "confidence": 0.92,
                "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 300},
                "position": "center",
                "proximity": "near"
            }
        }


class ImageRequest(BaseModel):
    image: str = Field(..., description="Imagem em base64")
    return_audio: bool = Field(False, description="Se deve retornar áudio TTS")
    
    @validator("image")
    def validate_base64(cls, v):
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Imagem base64 inválida")


class DetectionResponse(BaseModel):
    success: bool = Field(..., description="Status da operação")
    detections: List[Detection] = Field(..., description="Lista de detecções")
    description: Optional[str] = Field(None, description="Descrição textual das detecções")
    audio: Optional[str] = Field(None, description="Áudio TTS em base64")
    processing_time: float = Field(..., description="Tempo de processamento em segundos")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "detections": [
                    {
                        "class_name": "person",
                        "confidence": 0.92,
                        "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 300},
                        "position": "center",
                        "proximity": "near"
                    }
                ],
                "description": "Pessoa à frente, perto",
                "processing_time": 0.123,
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Status do serviço")
    version: str = Field(..., description="Versão da API")
    model_loaded: bool = Field(..., description="Se o modelo YOLO está carregado")
    tts_available: bool = Field(..., description="Se o TTS está disponível")
    timestamp: datetime = Field(default_factory=datetime.now)


class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Tipo da mensagem (frame, config, control)")
    data: Dict[str, Any] = Field(..., description="Dados da mensagem")
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    success: bool = Field(False)
    error: str = Field(..., description="Mensagem de erro")
    details: Optional[Dict[str, Any]] = Field(None, description="Detalhes adicionais do erro")
    timestamp: datetime = Field(default_factory=datetime.now)
