from fastapi import FastAPI, UploadFile, File
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import io

app = FastAPI()

MODEL_ID = "microsoft/OmniParser-v2.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(MODEL_ID).to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    text_output = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    return {"prediction": text_output}

