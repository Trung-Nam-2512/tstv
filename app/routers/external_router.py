from fastapi import APIRouter, Depends
from datetime import datetime, date
from ..services.mongo_service import MongoService
from ..dependencies import get_mongo_service

router = APIRouter(prefix="/external", tags=["external"])

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Backend is running"}

@router.post("/visit")
async def record_visit(mongo_service: MongoService = Depends(get_mongo_service)):
    """Ghi lại lượt truy cập"""
    try:
        success = await mongo_service.record_visit()
        if success:
            return {"status": "success", "message": "Visit recorded"}
        else:
            return {"status": "error", "message": "Failed to record visit"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/stats-visit")
async def get_visit_stats(mongo_service: MongoService = Depends(get_mongo_service)):
    """Lấy thống kê lượt truy cập"""
    try:
        stats = await mongo_service.get_visit_stats()
        return stats
    except Exception as e:
        return {"status": "error", "message": str(e)}
