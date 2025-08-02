from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from ..dependencies import get_data_service
from ..services.data_service import DataService
from ..models.data_models import UploadManualPayload

router = APIRouter(prefix="/data", tags=["data"])

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), data_service: DataService = Depends(get_data_service)):
    """Upload file CSV hoặc Excel"""
    return await data_service.upload_file(file)

@router.post("/upload_manual")
def upload_manual(payload: UploadManualPayload, data_service: DataService = Depends(get_data_service)):
    """Upload dữ liệu thủ công qua JSON"""
    return data_service.upload_manual(payload)

@router.get("/current")
def get_current_data(data_service: DataService = Depends(get_data_service)):
    """Lấy dữ liệu hiện tại"""
    if data_service.data is None:
        raise HTTPException(status_code=404, detail="Chưa có dữ liệu được tải")
    return {
        "data": data_service.data.to_dict(orient="records"),
        "main_column": data_service.main_column,
        "shape": data_service.data.shape
    }

@router.delete("/clear")
def clear_data(data_service: DataService = Depends(get_data_service)):
    """Xóa dữ liệu hiện tại"""
    data_service.data = None
    data_service.main_column = None
    return {"message": "Dữ liệu đã được xóa"}
