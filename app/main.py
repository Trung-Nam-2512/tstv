from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import config
from .routers.data_router import router as data_router
from .routers.stats_router import router as stats_router
from .routers.analysis_router import router as analysis_router
from .routers.external_router import router as external_router
import logging
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Phần mềm phân tích dữ liệu khí tượng thủy văn")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router)
app.include_router(stats_router)
app.include_router(analysis_router)
app.include_router(external_router)

# Mongo client init - chỉ khởi tạo nếu có URI
if config.MONGO_URI:
    try:
        app.state.mongo_client = AsyncIOMotorClient(config.MONGO_URI)
        logging.info("MongoDB connected successfully")
    except Exception as e:
        logging.warning(f"MongoDB connection failed: {e}")
        app.state.mongo_client = None
else:
    logging.info("MongoDB URI not provided, skipping MongoDB connection")
    app.state.mongo_client = None