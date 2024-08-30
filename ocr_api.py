from texify.inference import batch_inference
from texify.model.model import load_model
from PIL import Image
from texify.model.processor import load_processor
from fastapi import FastAPI, File, UploadFile
import io

model = load_model()
processor = load_processor()

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ocr")
async def ocr_page(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_img = Image.open(io.BytesIO(contents))
        batch_inference(input_img, model, processor, temperature=0)
    except Exception:
        return {"message": "This was an error uploading the file."}
    finally:
        await file.close()
    
    return {"message": f"Successfully upload {file.filename}"}