import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
from fastapi.responses import StreamingResponse
from src.functions import combined_loss
from src.inference import predict_img
import warnings

warnings.filterwarnings("ignore")   
model = tf.keras.models.load_model(
    "model/pet_segmentation.keras",
    custom_objects={'combined_loss': combined_loss}
)
app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB").resize((128, 128))
    prediction=predict_img(img,model)        
    buf = BytesIO()
    prediction.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)