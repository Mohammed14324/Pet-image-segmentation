from libraries import *

model = tf.keras.models.load_model(
    "model/pet_segmentation.keras",
    custom_objects={"combined_loss": combined_loss}
)

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    pred_array = np.clip(prediction[0] * 255.0, 0, 255).astype(np.uint8)
    img_array=img_array[0]*255.0
    reuslte_img=img_array.tolist()
    for i in range(128):
        for j in range(128):
            resulte=np.argmax(pred_array[i][j])
            if(resulte!=1):
                for k in range(3):
                    reuslte_img[i][j][k]=1-reuslte_img[i][j][k]
    resulte_img=np.array(reuslte_img).astype(np.uint8)
    pred_img = Image.fromarray(resulte_img)
    
    buf = BytesIO()
    pred_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)