from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI(title="API de Detecção de Objetos com YOLO")

# Carrega o modelo YOLO pré-treinado
model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "API do YOLO está a funcionar com sucesso!"}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Não foi possível decodificar a imagem."}

    results = model(img)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "box": [round(coord, 2) for coord in xyxy]
            })

    return {
        "total_detections": len(detections),
        "detections": detections
    }