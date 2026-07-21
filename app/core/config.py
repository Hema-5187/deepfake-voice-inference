from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / ".env")

MODEL_FILE = os.getenv("MODEL_FILE","embedding_model.pkl")



DATABASE_URL = os.getenv("DATABASE_URL")

SECRET_KEY = os.getenv("SECRET_KEY")

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 60

MODEL_NAME = os.getenv("MODEL_NAME")

MODEL_CACHE = os.getenv("MODEL_CACHE")

UPLOAD_FOLDER = BASE_DIR / "uploads"

UPLOAD_FOLDER.mkdir(exist_ok=True)