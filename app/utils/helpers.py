from typing import Dict, Tuple, Any
from fastapi import HTTPException

def extract_params(params: Tuple) -> Dict[str, Any]:
    """
    Trích xuất tham số từ tuple fit của scipy.stats.
    - Nếu 2 params: loc, scale (shape=None).
    - Nếu 3 params: shape, loc, scale.
    - Nếu >3: shape là tuple các giá trị đầu, loc/scale cuối.
    Lý do: scipy.stats trả tuple khác nhau tùy dist, cần standardize.
    """
    if len(params) == 2:
        return {"shape": None, "loc": params[0], "scale": params[1]}
    elif len(params) == 3:
        return {"shape": params[0], "loc": params[1], "scale": params[2]}
    else:
        return {"shape": params[:-2], "loc": params[-2], "scale": params[-1]}

def validate_agg_func(agg_func: str):
    """
    Validate agg_func để tránh lặp check ở nhiều nơi (DRY).
    - Chỉ chấp nhận {"max", "min", "sum", "mean"} – các agg phổ biến cho dữ liệu thủy văn (e.g., max cho peak flow).
    - Raise HTTPException nếu invalid.
    """
    check_agg_func = {"max", "min", "sum", "mean"}
    if agg_func not in check_agg_func:
        raise HTTPException(status_code=400, detail=f"Invalid agg_func: must be one of {check_agg_func}")