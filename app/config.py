from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / ".env")

# ==========================================
# Model
# ==========================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "wav2wavec2 + SVM"
)

MODEL_CACHE = os.getenv(
    "MODEL_CACHE",
    "./model_cache"
)

# ==========================================
# Upload Folder
# ==========================================

UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)