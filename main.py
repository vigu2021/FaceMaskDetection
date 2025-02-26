from fastapi import FastAPI,UploadFile,File,HTTPException
from ultralytics import YOLO
import torch
import os
import logging
from data_processing.resize_images import resize_image_with_annotations
import cv2
import numpy as np
from fastapi.responses import FileResponse


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = "best.pt"

if not MODEL_PATH.endswith(".pt"):
    logger.error(f"❌ File must be a .pt file, MODEL PATH was {MODEL_PATH}")
    raise ValueError(f"❌ File must be a .pt file, MODEL PATH was {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    logger.error("f❌ Model file not found: {MODEL_PATH}")
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

try:
    model = YOLO(MODEL_PATH)
    logger.info(f"✅ YOLO model loaded successfully: {MODEL_PATH}")
except Exception as e:
    logger.critical(f"🚨 Failed to load model: {e}")
    raise SystemExit(e) 

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend.html")



@app.post("/detect_mask")
async def detect_mask(file: UploadFile = File()):
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail="❌ Only JPEG, JPG, or PNG files are allowed.")

    image_bytes = await file.read() #Image -> bytes
    nparr = np.frombuffer(image_bytes, np.uint8) #Bytes -> Numpy arr
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR) #BGR
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) #BGR -> RGB

    results = model(image_rgb)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) #Box pos
            conf = round(box.conf[0].item(), 2) #Confidence
            cls = int(box.cls[0].item()) 
            label = result.names[cls] #Mask or no mask detected

            logging.info(f"Box at {(x1,y1)} {(x2,y2)}")
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox":[x1, y1, x2, y2]
            })

    return {"detections":detections}