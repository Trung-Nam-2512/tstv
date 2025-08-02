import pandas as pd
from fastapi import HTTPException
from .data_service import DataService
from typing import Dict, Any
from ..utils.helpers import validate_agg_func

class StatsService:
    def __init__(self, data_service: DataService):
        self.data_service = data_service

    def get_basic_stats(self, agg_func: str = 'max') -> Dict[str, Any]:
        """Lấy thống kê cơ bản với hàm tổng hợp được chỉ định"""
        validate_agg_func(agg_func)
        df = self.data_service.data
        main_column = self.data_service.main_column
        if df is None:
            raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
        
        aggregated = df.groupby('Year')[main_column].agg(agg_func)
        stats = {
            "count": len(aggregated),
            "min": float(aggregated.min()),
            "max": float(aggregated.max()),
            "mean": float(aggregated.mean()),
            "std": float(aggregated.std()),
            "median": float(aggregated.median())
        }
        return stats

    def get_monthly_stats(self) -> Dict[str, Any]:
        """Lấy thống kê theo tháng"""
        df = self.data_service.data
        main_column = self.data_service.main_column
        if df is None:
            raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
        
        if 'Month' not in df.columns:
            raise HTTPException(status_code=400, detail="Dữ liệu không có cột Month")
        
        monthly_stats = df.groupby('Month')[main_column].agg(['min', 'max', 'mean', 'std']).reset_index()
        return monthly_stats.to_dict(orient="records")

    def get_annual_stats(self, agg_func: str = 'max') -> Dict[str, Any]:
        """Lấy thống kê theo năm"""
        validate_agg_func(agg_func)
        df = self.data_service.data
        main_column = self.data_service.main_column
        if df is None:
            raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
        
        if 'Month' in df.columns:
            # Nếu có cột Month, tính thống kê theo năm với các hàm tổng hợp
            annual_stats = df.groupby('Year')[main_column].agg(['min', 'max', 'mean', 'sum']).reset_index()
            annual_stats.columns = ['Year', 'min', 'max', 'mean', 'sum']
        else:
            # Nếu không có cột Month, sử dụng giá trị duy nhất cho mỗi năm
            annual_stats = df.groupby('Year')[main_column].agg(agg_func).reset_index()
            annual_stats.columns = ['Year', 'Value']
            annual_stats['min'] = annual_stats['Value']
            annual_stats['max'] = annual_stats['Value']
            annual_stats['mean'] = annual_stats['Value']
            annual_stats['sum'] = annual_stats['Value']
            annual_stats = annual_stats[['Year', 'min', 'max', 'mean', 'sum']]
        
        return annual_stats.to_dict(orient="records")

    def get_descriptive_stats(self) -> Dict[str, Any]:
        """
        Tính stats descriptive (min/max/mean/sum) group by Month nếu có, hoặc overall.
        - Fix: Nếu no Month, wrap stats dict thành list[{"overall": ...}] để consistent với has_month (luôn list).
        - Lý do: Client dễ parse (expect list records), tránh inconsistency.
        """
        df = self.data_service.data
        main_column = self.data_service.main_column
        if df is None:
            raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
        has_month = 'Month' in df.columns
        if has_month:
            grouped_data = df.groupby('Month')[main_column].agg(['min', 'max', 'mean', 'sum']).reset_index()
            stats = grouped_data.to_dict(orient="records")
        else:
            agg = df[main_column].agg(['min', 'max', 'mean', 'sum']).to_dict()
            stats = [{"overall": agg}]
        return {"stats": stats, "has_month": has_month}