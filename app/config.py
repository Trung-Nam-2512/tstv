from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    MONGO_URI = os.getenv("MONGO_URI")
    ALLOW_ORIGINS = ["*"]
    STATIONS_API_BASE_URL = os.getenv("STATIONS_API_BASE_URL")  # E.g., "https://api.com/v1/stations"
    STATS_API_BASE_URL = os.getenv("STATS_API_BASE_URL")  # E.g., "https://api.com/v1/stations/stats"
    API_KEY = os.getenv("API_KEY")  # For header auth, e.g., "X-API-Key: your_key"
    DEPTH_THRESHOLD = float(os.getenv("DEPTH_THRESHOLD", 2.0))  # Alert if depth > this (m), customizable

config = Config()