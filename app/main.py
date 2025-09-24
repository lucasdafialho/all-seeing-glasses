from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import json
import time
import logging
from typing import Optional
from datetime import datetime
import base64

from app.config import settings
from app.schemas import (
    ImageRequest, DetectionResponse, HealthCheckResponse,
    WebSocketMessage, ErrorResponse, Detection, BoundingBox
)
from app.detection import get_detector
from app.tts import get_improved_tts
from app.utils import (
    base64_to_image, image_to_base64, resize_image_if_needed,
    format_detection_description, audio_to_base64, validate_image_file
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="API para sistema de óculos assistivos com detecção de objetos e TTS"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None
tts = None


@app.on_event("startup")
async def startup_event():
    global detector, tts
    
    logger.info(f"Iniciando {settings.app_name} v{settings.version}")
    
    try:
        detector = get_detector()
        logger.info("Detector de objetos inicializado")
    except Exception as e:
        logger.error(f"Erro ao inicializar detector: {e}")
    
    try:
        tts = get_improved_tts()
        logger.info("Sistema TTS inicializado")
    except Exception as e:
        logger.error(f"Erro ao inicializar TTS: {e}")
    
    logger.info("Aplicação iniciada com sucesso")


@app.get("/", response_model=dict)
async def root():
    return {
        "name": settings.app_name,
        "version": settings.version,
        "endpoints": {
            "health": "/healthz",
            "detect": "/detect",
            "stream": "/stream"
        }
    }


@app.get("/healthz", response_model=HealthCheckResponse)
async def health_check():
    model_loaded = detector is not None and detector.model is not None
    tts_available = tts is not None and tts.is_available()
    
    return HealthCheckResponse(
        status="healthy" if model_loaded else "degraded",
        version=settings.version,
        model_loaded=model_loaded,
        tts_available=tts_available
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(request: ImageRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector não disponível")
    
    try:
        start_time = time.time()
        
        image = base64_to_image(request.image)
        
        image = resize_image_if_needed(image)
        
        detections = detector.detect_objects(image)
        
        detection_objects = []
        for det in detections:
            detection_objects.append(Detection(
                class_name=det['class_name'],
                confidence=det['confidence'],
                bbox=BoundingBox(**det['bbox']),
                position=det['position'],
                proximity=det['proximity']
            ))
        
        description = format_detection_description(detections)
        
        audio_base64 = None
        if request.return_audio and tts and tts.is_available():
            audio_data = tts.generate_audio(description)
            if audio_data:
                audio_base64 = audio_to_base64(audio_data)
        
        processing_time = time.time() - start_time
        
        return DetectionResponse(
            success=True,
            detections=detection_objects,
            description=description,
            audio=audio_base64,
            processing_time=processing_time
        )
        
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro na detecção: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.post("/detect/file", response_model=DetectionResponse)
async def detect_objects_file(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector não disponível")
    
    try:
        contents = await file.read()
        
        if not validate_image_file(contents):
            raise HTTPException(status_code=400, detail="Arquivo de imagem inválido")
        
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        request = ImageRequest(image=image_base64, return_audio=False)
        
        return await detect_objects(request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao processar arquivo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Nova conexão WebSocket. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Conexão WebSocket encerrada. Total: {len(self.active_connections)}")
    
    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)


manager = ConnectionManager()


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await manager.connect(websocket)
    session_id = f"ws_{id(websocket)}"
    
    try:
        logger.info(f"Nova sessão WebSocket iniciada: {session_id}")
        
        await websocket.send_json({
            "type": "connection",
            "data": {
                "status": "connected",
                "session_id": session_id,
                "model": settings.yolo_model,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        config = {
            "return_audio": False,
            "detection_interval": settings.min_detection_interval,
            "confidence_threshold": settings.confidence_threshold
        }
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "config":
                    config.update(message.get("data", {}))
                    await websocket.send_json({
                        "type": "config_updated",
                        "data": config,
                        "timestamp": datetime.now().isoformat()
                    })
                    continue
                
                elif message.get("type") == "frame":
                    frame_data = message.get("data", {})
                    image_base64 = frame_data.get("image")
                    
                    if not image_base64:
                        await websocket.send_json({
                            "type": "error",
                            "data": {"error": "Imagem não fornecida"},
                            "timestamp": datetime.now().isoformat()
                        })
                        continue
                    
                    try:
                        start_time = time.time()
                        
                        image = base64_to_image(image_base64)
                        image = resize_image_if_needed(image, max_size=1280)
                        
                        detections = detector.detect_objects(
                            image, 
                            confidence_threshold=config.get("confidence_threshold")
                        )
                        
                        filtered_detections = detector.filter_repeated_detections(
                            detections, 
                            session_id
                        )
                        
                        response_data = {
                            "detections": detections,
                            "new_detections": filtered_detections,
                            "processing_time": time.time() - start_time
                        }
                        
                        if filtered_detections and config.get("return_audio"):
                            description = format_detection_description(filtered_detections)
                            if tts and tts.is_available():
                                audio_data = tts.generate_audio(description)
                                if audio_data:
                                    response_data["audio"] = audio_to_base64(audio_data)
                                    response_data["description"] = description
                        
                        await websocket.send_json({
                            "type": "detection",
                            "data": response_data,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        logger.error(f"Erro ao processar frame: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "data": {"error": f"Erro ao processar frame: {str(e)}"},
                            "timestamp": datetime.now().isoformat()
                        })
                
                elif message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
            except WebSocketDisconnect:
                logger.info(f"Cliente desconectado: {session_id}")
                break
            except json.JSONDecodeError as e:
                await websocket.send_json({
                    "type": "error",
                    "data": {"error": f"JSON inválido: {str(e)}"},
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Erro no WebSocket: {e}")
                await websocket.send_json({
                    "type": "error",
                    "data": {"error": str(e)},
                    "timestamp": datetime.now().isoformat()
                })
                break
    
    finally:
        manager.disconnect(websocket)
        logger.info(f"Sessão WebSocket encerrada: {session_id}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            details={"status_code": exc.status_code}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Erro não tratado: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Erro interno do servidor",
            details={"type": type(exc).__name__}
        ).dict()
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )
