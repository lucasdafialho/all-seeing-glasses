import base64
import numpy as np
import cv2
from PIL import Image
import io
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Converte uma string base64 para uma imagem numpy array.
    """
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return np.array(img)
    except Exception as e:
        logger.error(f"Erro ao converter base64 para imagem: {e}")
        raise ValueError(f"Falha ao decodificar imagem base64: {str(e)}")


def image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """
    Converte uma imagem numpy array para string base64.
    """
    try:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        img_pil = Image.fromarray(image)
        buffer = io.BytesIO()
        img_pil.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/{format.lower()};base64,{img_str}"
    except Exception as e:
        logger.error(f"Erro ao converter imagem para base64: {e}")
        raise ValueError(f"Falha ao codificar imagem: {str(e)}")


def resize_image_if_needed(image: np.ndarray, max_size: int = 1920) -> np.ndarray:
    """
    Redimensiona a imagem se necessário, mantendo a proporção.
    """
    height, width = image.shape[:2]
    
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"Imagem redimensionada de {width}x{height} para {new_width}x{new_height}")
    
    return image


def calculate_position(bbox: Tuple[float, float, float, float], image_width: int) -> str:
    """
    Calcula a posição relativa do objeto na imagem.
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    relative_x = center_x / image_width
    
    if relative_x < 0.33:
        return "esquerda"
    elif relative_x < 0.66:
        return "centro"
    else:
        return "direita"


def calculate_proximity(bbox: Tuple[float, float, float, float], 
                        image_width: int, image_height: int) -> str:
    """
    Estima a proximidade do objeto com base no tamanho da bbox.
    """
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    bbox_area = bbox_width * bbox_height
    image_area = image_width * image_height
    relative_area = bbox_area / image_area
    
    if relative_area > 0.3:
        return "perto"
    elif relative_area > 0.15:
        return "médio"
    else:
        return "longe"


def translate_class_name(class_name: str) -> str:
    """
    Traduz nomes de classes para português.
    """
    translations = {
        "person": "pessoa",
        "bicycle": "bicicleta", 
        "car": "carro",
        "motorcycle": "moto",
        "airplane": "avião",
        "bus": "ônibus",
        "train": "trem",
        "truck": "caminhão",
        "boat": "barco",
        "traffic light": "semáforo",
        "fire hydrant": "hidrante",
        "stop sign": "placa de pare",
        "parking meter": "parquímetro",
        "bench": "banco",
        "bird": "pássaro",
        "cat": "gato",
        "dog": "cachorro",
        "horse": "cavalo",
        "sheep": "ovelha",
        "cow": "vaca",
        "elephant": "elefante",
        "bear": "urso",
        "zebra": "zebra",
        "giraffe": "girafa",
        "backpack": "mochila",
        "umbrella": "guarda-chuva",
        "handbag": "bolsa",
        "tie": "gravata",
        "suitcase": "mala",
        "frisbee": "frisbee",
        "skis": "esquis",
        "snowboard": "snowboard",
        "sports ball": "bola",
        "kite": "pipa",
        "baseball bat": "taco de baseball",
        "baseball glove": "luva de baseball",
        "skateboard": "skate",
        "surfboard": "prancha de surf",
        "tennis racket": "raquete de tênis",
        "bottle": "garrafa",
        "wine glass": "taça",
        "cup": "xícara",
        "fork": "garfo",
        "knife": "faca",
        "spoon": "colher",
        "bowl": "tigela",
        "banana": "banana",
        "apple": "maçã",
        "sandwich": "sanduíche",
        "orange": "laranja",
        "broccoli": "brócolis",
        "carrot": "cenoura",
        "hot dog": "cachorro-quente",
        "pizza": "pizza",
        "donut": "rosquinha",
        "cake": "bolo",
        "chair": "cadeira",
        "couch": "sofá",
        "potted plant": "planta",
        "bed": "cama",
        "dining table": "mesa",
        "toilet": "vaso sanitário",
        "tv": "televisão",
        "laptop": "notebook",
        "mouse": "mouse",
        "remote": "controle remoto",
        "keyboard": "teclado",
        "cell phone": "celular",
        "microwave": "micro-ondas",
        "oven": "forno",
        "toaster": "torradeira",
        "sink": "pia",
        "refrigerator": "geladeira",
        "book": "livro",
        "clock": "relógio",
        "vase": "vaso",
        "scissors": "tesoura",
        "teddy bear": "ursinho",
        "hair drier": "secador",
        "toothbrush": "escova de dente",
        "glasses": "óculos"
    }
    
    return translations.get(class_name.lower(), class_name)


def format_detection_description(detections: list) -> str:
    """
    Formata as detecções em uma descrição textual.
    """
    if not detections:
        return "Nenhum objeto detectado"
    
    descriptions = []
    
    grouped = {}
    for det in detections[:5]:
        key = f"{det['class_name']}_{det['position']}_{det['proximity']}"
        if key not in grouped:
            grouped[key] = det
    
    for det in grouped.values():
        class_name = translate_class_name(det['class_name'])
        position = det['position']
        proximity = det['proximity']
        
        if position == "centro":
            desc = f"{class_name} à frente, {proximity}"
        else:
            desc = f"{class_name} à {position}, {proximity}"
        
        descriptions.append(desc)
    
    return ". ".join(descriptions) + "."


def audio_to_base64(audio_data: bytes) -> str:
    """
    Converte dados de áudio para base64.
    """
    return base64.b64encode(audio_data).decode('utf-8')


def validate_image_file(file_bytes: bytes) -> bool:
    """
    Valida se os bytes representam uma imagem válida.
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        return True
    except Exception:
        return False
