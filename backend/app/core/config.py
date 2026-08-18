import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "EmotionSense AI"
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./emotisense.db")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "ml/models/saved")
    DATASET_DIR: str = os.getenv("DATASET_DIR", "ml/data/ravdess")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

settings = Settings()
