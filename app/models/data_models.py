from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

class UploadManualPayload(BaseModel):
    data: List[Dict[str, Any]]

class StatsResponse(BaseModel):
    stats: Union[List[Dict[str, Any]], Dict[str, Any]]
    has_month: Optional[bool] = None

class FrequencyCurveResponse(BaseModel):
    theoretical_curve: List[Dict[str, Any]]
    empirical_points: List[Dict[str, Any]]

class QQPPResponse(BaseModel):
    qq: List[Dict[str, Any]]
    pp: List[Dict[str, Any]]

class QuantileDataResponse(BaseModel):
    years: List[int]
    qmax_values: List[float]
    histogram: Dict[str, List[Any]]
    theoretical_curve: Dict[str, List[float]]

# Có thể thêm models khác nếu cần