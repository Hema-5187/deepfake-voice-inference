from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.core.config import UPLOAD_FOLDER
from app.ml.inference import inference_engine

app = FastAPI(
    title="Deepfake Voice Inference API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "service": "Deepfake Voice Inference API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    suffix = Path(file.filename).suffix

    filename = f"{uuid.uuid4()}{suffix}"

    temp_path = UPLOAD_FOLDER / filename

    try:

        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = inference_engine.predict(temp_path)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        if temp_path.exists():
            temp_path.unlink()