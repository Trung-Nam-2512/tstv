import pandas as pd
import numpy as np
from io import BytesIO
from fastapi import UploadFile, HTTPException
import logging
import re
from typing import Dict, Any, Union
from ..models.data_models import UploadManualPayload

class DataService:
    def __init__(self):
        self.data: Union[pd.DataFrame, None] = None
        self.main_column: Union[str, None] = None

    def convert_month(self, month_value: Any) -> Union[int, None]:
        """
        Chuyển đổi month_value thành int, trả None nếu invalid.
        - Extract digits từ string (e.g., "Month 1" → 1).
        - Nếu không có số (e.g., "Jan") → None → sẽ NaN, sau dropna.
        """
        try:
            if isinstance(month_value, str):
                digits = re.findall(r'\d+', month_value)
                if digits:
                    return int(digits[0])
                else:
                    return None
            else:
                return int(month_value)
        except Exception:
            return None

    def detect_main_data_column(self, df: pd.DataFrame) -> str:
        """
        Phát hiện main_column số chính (không phải Year/Month).
        - Raise nếu không tìm thấy numeric col hoặc format sai (2/3 cols).
        - Lý do: Đảm bảo DF phù hợp cho phân tích thủy văn (yearly/monthly series).
        """
        numeric_columns = df.select_dtypes(include=np.number).columns
        if len(numeric_columns) == 0:
            raise ValueError("Không tìm thấy cột số trong dữ liệu.")
        if len(df.columns) == 3:
            if "Year" in df.columns and "Month" in df.columns:
                for col in df.columns:
                    if col not in ["Year", "Month"]:
                        return col
            else:
                raise ValueError("Phải có cột Year, Month khi dữ liệu có 3 cột.")
        elif len(df.columns) == 2:
            if "Year" in df.columns:
                for col in df.columns:
                    if col != "Year":
                        return col
            else:
                raise ValueError("Phải có cột Year khi dữ liệu có 2 cột.")
        raise ValueError("Không tìm thấy cột dữ liệu phù hợp. Vui lòng kiểm tra lại dữ liệu.")

    def process_data(self, df: pd.DataFrame, main_column: str) -> pd.DataFrame:
        """
        Xử lý DF để nhất quán: Convert Month, expand no Month thành 12 rows/year, filter >0.
        - Fix: Thêm dropna cho Month nếu có (tránh NaN từ convert_month invalid).
        - Raise nếu DF empty sau process (edge case empty data).
        - Lý do: Trong thủy văn, Month NaN có thể từ data bẩn → drop để agg chính xác.
        """
        if "Month" in df.columns:
            df["Month"] = df["Month"].apply(self.convert_month)
            df = df.dropna(subset=["Month"])  # Drop rows với Month NaN
        else:
            if "Year" not in df.columns:
                raise ValueError("Dữ liệu phải chứa cột 'Year'")
            logging.info("Không có cột 'Month'. Tạo tự động 12 tháng cho mỗi năm với giá trị của năm đó.")
            new_rows = []
            for idx, row in df.iterrows():
                year = row["Year"]
                yearly_value = row[main_column]
                for m in range(1, 13):
                    new_rows.append({"Year": year, "Month": m, main_column: yearly_value})
            df = pd.DataFrame(new_rows)
        df = df[df[main_column] > 0]  # Loại bỏ giá trị không hợp lệ
        if df.empty:
            raise HTTPException(status_code=400, detail="Dữ liệu rỗng sau xử lý (có thể tất cả giá trị <=0 hoặc NaN)")
        return df

    async def upload_file(self, file: UploadFile) -> Dict:
        try:
            contents = await file.read()
            logging.info(f"Đã nhận file: {file.filename}")
            if file.filename.endswith('.csv'):
                df = pd.read_csv(BytesIO(contents), on_bad_lines='skip')
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(BytesIO(contents))
            else:
                raise HTTPException(status_code=400, detail="File type not supported")
            
            main_column = self.detect_main_data_column(df)
            df = self.process_data(df, main_column)
            
            self.data = df
            self.main_column = main_column
            return {"status": "success", "data": df.to_dict(orient="records")}
        except Exception as e:
            logging.error(f"Lỗi khi xử lý file: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def upload_manual(self, payload: UploadManualPayload) -> Dict:
        """
        Upload manual từ JSON payload.
        - Fix: Sau to_numeric, dropna main_column để loại NaN (từ coerce invalid values).
        - Lý do: Tránh agg NaN dẫn đến stats/analysis nan.
        """
        try:
            if not isinstance(payload.data, list):
                raise HTTPException(status_code=400, detail="Payload phải chứa trường 'data' dưới dạng danh sách")
            
            df = pd.DataFrame(payload.data)
            
            if "Year" not in df.columns:
                raise HTTPException(status_code=400, detail="Dữ liệu phải chứa cột 'Year'")
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            
            main_column = self.detect_main_data_column(df)
            df[main_column] = pd.to_numeric(df[main_column], errors="coerce")
            
            df = df.dropna(subset=[main_column])  # Drop NaN in main_column
            
            df = self.process_data(df, main_column)
            
            self.data = df
            self.main_column = main_column
            
            return {"status": "success", "data": df.to_dict(orient="records")}
        except Exception as e:
            logging.error(f"Lỗi trong /upload_manual: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))