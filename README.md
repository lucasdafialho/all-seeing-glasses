# All Seeing Glasses

Sistema completo de detecção de objetos em tempo real usando YOLO e Text-to-Speech para auxiliar pessoas com deficiência visual.

## Funcionalidades

- **Detecção de Objetos em Tempo Real**: Utiliza YOLOv8 para detectar pessoas, animais, veículos e objetos diversos
- **Text-to-Speech**: Converte detecções em narração de áudio com alta qualidade
- **API REST**: Endpoints para processamento de imagens
- **WebSocket**: Stream de vídeo em tempo real para integração
- **Demo com Webcam**: Interface completa para demonstrações e testes
- **Funcionamento Offline**: Sistema operacional sem necessidade de conexão com internet

## Pré-requisitos

- Python 3.11 ou superior
- Câmera/webcam (necessária para demonstração)
- Sistema operacional Linux, Windows ou macOS
- Acesso à internet (para instalação de dependências e TTS Google)

## Instalação

### 1. Clone o repositório
```bash
git clone <url-do-repositorio>
cd ic-ocr
```

### 2. Crie um ambiente virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate     # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute a demonstração com webcam
```bash
python demo_webcam.py
```

### 5. Execute a API completa (opcional)
```bash
python -m app.main
# ou
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Demonstração com Webcam

Execute o script de demonstração para testar o sistema com sua câmera:

```bash
python demo_webcam.py
```

### Controles da Demonstração

| Tecla | Função |
|-------|--------|
| ESC | Encerrar aplicação |
| SPACE | Ativar/Desativar TTS (áudio) |
| T | Mostrar/Ocultar detecções visuais |
| R | Reset do histórico de detecções |

### Parâmetros de Linha de Comando

```bash
# Usar câmera específica (padrão: 0)
python demo_webcam.py --camera 1

# Desativar Text-to-Speech
python demo_webcam.py --no-tts

# Ajustar limiar de confiança (padrão: 0.45)
python demo_webcam.py --confidence 0.5

# Visualizar todas as opções disponíveis
python demo_webcam.py --help
```

## API REST

### Iniciar o servidor
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

#### Health Check
```bash
curl http://localhost:8000/healthz
```

#### Detectar objetos (Base64)
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABA...",
    "return_audio": true
  }'
```

#### Detectar objetos (Upload de arquivo)
```bash
curl -X POST "http://localhost:8000/detect/file" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@imagem.jpg"
```

### Exemplo de Resposta da API
```json
{
  "success": true,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.92,
      "bbox": {
        "x1": 100,
        "y1": 100, 
        "x2": 200,
        "y2": 300
      },
      "position": "center",
      "proximity": "near"
    }
  ],
  "description": "Pessoa à frente, perto",
  "audio": "base64_encoded_audio_data",
  "processing_time": 0.123,
  "timestamp": "2024-01-01T12:00:00"
}
```

## WebSocket

Conecte-se ao WebSocket para stream em tempo real:

```javascript
const ws = new WebSocket('ws://localhost:8000/stream');

// Enviar frame de vídeo
ws.send(JSON.stringify({
  type: "frame",
  data: {
    image: "base64_image_data"
  }
}));

// Configurar parâmetros
ws.send(JSON.stringify({
  type: "config",
  data: {
    return_audio: true,
    confidence_threshold: 0.4
  }
}));
```

## Docker

### Build da imagem
```bash
docker build -t assistive-glasses .
```

### Executar container
```bash
docker run -p 8000:8000 assistive-glasses
```

### Docker Compose
```bash
docker-compose up -d
```

## Configurações

### Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto para personalizar as configurações:

```bash
YOLO_MODEL=yolov8s.pt
CONFIDENCE_THRESHOLD=0.45
IOU_THRESHOLD=0.8
MIN_DETECTION_INTERVAL=1.0
MAX_DETECTIONS_PER_FRAME=12
HOST=0.0.0.0
PORT=8000
```

### Classes de Objetos Detectadas
O sistema detecta as 80 classes do dataset COCO, incluindo:

**Pessoas e Animais:**
- person (pessoa), dog (cachorro), cat (gato), bird (pássaro)

**Veículos:**
- car (carro), bicycle (bicicleta), motorcycle (moto), bus (ônibus)

**Objetos do Cotidiano:**
- cell phone (celular), laptop (notebook), tv (televisão), bottle (garrafa)

**Mobiliário:**
- chair (cadeira), couch (sofá), bed (cama), dining table (mesa)

**Sinalização:**
- traffic light (semáforo), stop sign (placa de pare)

## Estrutura do Projeto

```
ic-ocr/
├── app/
│   ├── main.py          # API FastAPI principal
│   ├── config.py        # Configurações
│   ├── detection.py     # Detecção YOLO
│   ├── tts.py          # Text-to-Speech
│   ├── schemas.py      # Modelos Pydantic
│   └── utils.py        # Utilitários
├── models/             # Modelos YOLO (baixados automaticamente)
├── demo_webcam.py      # Script de demonstração
├── requirements.txt    # Dependências Python
├── Dockerfile         # Container Docker
├── docker-compose.yml # Orchestração Docker
└── README.md          # Este arquivo
```

## Solução de Problemas

### Problemas com Câmera
Se a câmera não funcionar, teste diferentes dispositivos:
```bash
python demo_webcam.py --camera 0
python demo_webcam.py --camera 1
python demo_webcam.py --camera 2
```

### Problemas com Text-to-Speech
Para executar sem áudio (apenas detecção visual):
```bash
python demo_webcam.py --no-tts
```

### Performance e Otimização
Para melhorar o desempenho em sistemas mais lentos:
```bash
# Reduzir sensibilidade (menos detecções)
python demo_webcam.py --confidence 0.6

# Verificar disponibilidade de GPU
python -c "import torch; print('GPU disponível:', torch.cuda.is_available())"
```

### Download Manual do Modelo YOLO
Em caso de problemas com download automático:
```python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')  # Download automático do modelo
```

### Dependências de Sistema
Para Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3-dev portaudio19-dev
```

## Desenvolvimento

### Executar em modo desenvolvimento
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Logs detalhados
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Arquitetura do Sistema

### Componentes Principais

1. **Módulo de Detecção (`detection.py`)**
   - Carregamento e inicialização do modelo YOLOv8
   - Processamento de frames de vídeo
   - Filtragem de detecções por confiança e IoU

2. **Módulo de Text-to-Speech (`tts.py`)**  
   - Google TTS (gTTS) para alta qualidade (online)
   - Sistema de fallback automático

3. **API REST (`main.py`)**
   - Endpoints para detecção em imagens
   - WebSocket para streaming em tempo real

4. **Interface de Demonstração (`demo_webcam.py`)**
   - Captura de vídeo em tempo real
   - Exibição de resultados visuais

### Fluxo de Processamento

1. **Captura**: Frame obtido da webcam/câmera
2. **Pré-processamento**: Redimensionamento e normalização
3. **Inferência**: Detecção usando modelo YOLO treinado
4. **Pós-processamento**: Filtragem por confiança e NMS
5. **Interpretação**: Cálculo de posição e proximidade
6. **Saída**: Descrição textual e síntese de fala

## Considerações Técnicas

### Requisitos de Hardware
- **CPU**: Processador dual-core mínimo (quad-core recomendado)
- **RAM**: 4GB mínimo (8GB recomendado)
- **Armazenamento**: 2GB de espaço livre
- **Conectividade**: Internet para TTS Google (opcional)

### Otimizações Implementadas
- Non-Maximum Suppression (NMS) para reduzir detecções duplicadas
- Cache de modelos YOLO para inicialização mais rápida
- Filtro temporal para evitar redundância na narração

## Referências Técnicas

- **YOLO**: [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- **OpenCV**: [Computer Vision Library](https://opencv.org/)
- **FastAPI**: [Modern Web API Framework](https://fastapi.tiangolo.com/)
- **gTTS**: [Google Text-to-Speech Library](https://github.com/pndurette/gTTS)

## Licença

Este projeto foi desenvolvido para fins acadêmicos e está disponível sob licença MIT.

