from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from inferenece import inference

from PIL import Image
import io

app = FastAPI()

@app.post("/predict_helmet/")
async def predict_medium(file: UploadFile = File(...)):

    image = Image.open(file.file)

    result = inference(image)
    
    buffer = io.BytesIO()   # Create a buffer to hold the image data

    result.save(buffer, format="JPEG")  # Save the result image to the buffer
    buffer.seek(0) # Reset buffer position to the beginning
    return StreamingResponse(buffer, media_type="image/jpeg")

