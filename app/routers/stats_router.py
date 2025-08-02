from fastapi import APIRouter, Depends, Query
from ..dependencies import get_stats_service
from ..services.stats_service import StatsService

router = APIRouter(prefix="/stats", tags=["stats"])

@router.get("/basic")
def get_basic_stats(agg_func: str = Query('max'), stats_service: StatsService = Depends(get_stats_service)):
    """Lấy thống kê cơ bản"""
    return stats_service.get_basic_stats(agg_func)

@router.get("/monthly")
def get_monthly_stats(stats_service: StatsService = Depends(get_stats_service)):
    """Lấy thống kê theo tháng"""
    return stats_service.get_monthly_stats()

@router.get("/annual")
def get_annual_stats(agg_func: str = Query('max'), stats_service: StatsService = Depends(get_stats_service)):
    """Lấy thống kê theo năm"""
    return stats_service.get_annual_stats(agg_func)
