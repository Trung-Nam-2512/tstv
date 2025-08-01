# main.py
from fastapi import FastAPI, HTTPException,Request,Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import gumbel_r, genextreme, genpareto, expon, lognorm, logistic, gamma,chi2,pearson3
from fastapi import File, UploadFile
from io import BytesIO
import logging
from fastapi.responses import JSONResponse,FileResponse
from datetime import datetime
import requests
import re, os
import math
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Phần mềm phân tích dữ liệu khí tượng thủy văn")
app.state.data = None
app.state.main_column = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    
)

logging.basicConfig(level=logging.INFO)



def get_app_state(request: Request):
    """Lấy trạng thái ứng dụng từ request."""
    return request.app.state

def convert_month(month_value):
    """Chuyển đổi giá trị tháng thành số nguyên, trả về None nếu không hợp lệ."""
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

def detect_main_data_column(df):
    """Phát hiện cột dữ liệu chính (main_column) trong DataFrame."""
    numeric_columns = df.select_dtypes(include=np.number).columns
    if not numeric_columns.any():
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

def process_data(df, main_column):
    """Xử lý dữ liệu để đảm bảo cấu trúc nhất quán."""
    if "Month" in df.columns:
        df["Month"] = df["Month"].apply(convert_month)
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
    return df

# --- Endpoint upload file ---
@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Xử lý file CSV hoặc Excel được tải lên."""
    try:
        contents = await file.read()
        logging.info(f"Đã nhận file: {file.filename}")
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents), on_bad_lines='skip')
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="File type not supported")
        main_column = detect_main_data_column(df)
        df = process_data(df, main_column)
        request.app.state.data = df
        request.app.state.main_column = main_column
        return {"status": "success", "data": df.to_dict(orient="records")}
    except Exception as e:
        logging.error(f"Lỗi khi xử lý file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint upload dữ liệu nhập tay ---
@app.post("/upload_manual")
async def upload_manual(request: Request, payload: dict):
    """Xử lý dữ liệu nhập tay qua payload JSON."""
    try:
        if "data" not in payload or not isinstance(payload["data"], list):
            raise HTTPException(status_code=400, detail="Payload phải chứa trường 'data' dưới dạng danh sách")
        
        df = pd.DataFrame(payload["data"])
        
        if "Year" not in df.columns:
            raise HTTPException(status_code=400, detail="Dữ liệu phải chứa cột 'Year'")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        
        main_column = detect_main_data_column(df)
        df[main_column] = pd.to_numeric(df[main_column], errors="coerce")
        
        df = process_data(df, main_column)
        
        request.app.state.data = df
        request.app.state.main_column = main_column
        
        return {"status": "success", "data": df.to_dict(orient="records")}
    except Exception as e:
        logging.error(f"Lỗi trong /upload_manual: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_descriptive_stats(df: pd.DataFrame, main_column: str):
    """
    Nếu DataFrame có cột 'Month', nhóm theo Month và tính thống kê.
    Nếu không có cột 'Month', tính thống kê trên toàn bộ dữ liệu (giả sử mỗi năm chỉ có 1 dòng)
    và gắn biến cờ has_month = False.
    """
    has_month = 'Month' in df.columns
    if has_month:
        grouped_data = df.groupby('Month')[main_column].agg(['min', 'max', 'mean', 'sum']).reset_index()
        stats = grouped_data.to_dict(orient="records")
    else:
        # Ở đây, giả sử dữ liệu mỗi năm có 1 dòng: min, max, mean và sum đều là chính giá trị đó.
        agg = df[main_column].agg(['min', 'max', 'mean', 'sum']).to_dict()
        stats = agg
    return stats, has_month


@app.get("/data/overall")
def get_overall_stats(request: Request):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
    stats, has_month = get_descriptive_stats(data, main_column)
    # Nếu không có cột Month, có thể gắn thông tin cờ vào kết quả trả về
    return {"stats": stats, "has_month": has_month}


def get_annual_stats(df: pd.DataFrame, main_column: str):
    if 'Month' in df.columns:
        # Nếu có cột Month, nhóm theo Year và tính thống kê dựa trên dữ liệu từng tháng
        grouped_data = df.groupby('Year')[main_column].agg(['min', 'max', 'mean', 'sum']).reset_index()
        stats = grouped_data.to_dict(orient="records")
    else:
        # Nếu không có cột Month, giả sử mỗi năm nên chỉ có 1 giá trị duy nhất.
        # Loại bỏ các dòng trùng lặp theo "Year"
        df_unique = df.drop_duplicates(subset=["Year"])
        # Sau đó, các thống kê min, max, mean, sum đều bằng giá trị đó
        df_unique["min"] = df_unique[main_column]
        df_unique["max"] = df_unique[main_column]
        df_unique["mean"] = df_unique[main_column]
        df_unique["sum"] = df_unique[main_column]
        stats = df_unique[["Year", "min", "max", "mean", "sum"]].to_dict(orient="records")
    return stats



@app.get("/data/annual")
def get_annual_statistics(request: Request):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
    stats = get_annual_stats(data, main_column)
    return {"stats": stats}


# Phân tích xác định phân phối và tính AIC, chi-square cho các phân phối khác nhau
def extract_params(params):
    """
    Nếu tuple trả về có 2 phần tử: (loc, scale) → không có shape (gán None)
    Nếu tuple trả về có 3 phần tử: (shape, loc, scale)
    Nếu có nhiều hơn 3, gộp các giá trị ngoài loc và scale lại làm shape.
    """
    if len(params) == 2:
        return {"shape": None, "loc": params[0], "scale": params[1]}
    elif len(params) == 3:
        return {"shape": params[0], "loc": params[1], "scale": params[2]}
    else:
        return {"shape": params[:-2], "loc": params[-2], "scale": params[-1]}

@app.get("/analysis/distribution")
def get_distribution_analysis(request: Request, agg_func: str= Query('max')):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
    
    df = data.groupby('Year')[main_column].agg(agg_func).values

    def calculate_aic(n, log_likelihood, num_params):                                                                                                               
        return 2 * num_params - 2 * log_likelihood

    # Gumbel
    gumbel_params = gumbel_r.fit(df)
    gumbel_loglik = gumbel_r.logpdf(df, *gumbel_params).sum()
    gumbel_aic = calculate_aic(len(df), gumbel_loglik, len(gumbel_params))
    gumbel_extracted = extract_params(gumbel_params)

    # Frechet (dùng genextreme)
    frechet_params = genextreme.fit(df)
    frechet_loglik = np.sum(genextreme.logpdf(df, *frechet_params))
    frechet_aic = calculate_aic(len(df), frechet_loglik, len(frechet_params))
    frechet_extracted = extract_params(frechet_params)

  

    # Exponential
    expon_params = expon.fit(df)
    expon_loglik = np.sum(expon.logpdf(df, *expon_params))
    expon_aic = calculate_aic(len(df), expon_loglik, len(expon_params))
    expon_extracted = extract_params(expon_params)

    # Lognormal
    lognormal_params = lognorm.fit(df)
    lognormal_loglik = lognorm.logpdf(df, *lognormal_params).sum()
    lognormal_aic = calculate_aic(len(df), lognormal_loglik, len(lognormal_params))
    lognormal_extracted = extract_params(lognormal_params)

    # Logistic
    logistic_params = logistic.fit(df)
    logistic_loglik = np.sum(logistic.logpdf(df, *logistic_params))
    logistic_aic = calculate_aic(len(df), logistic_loglik, len(logistic_params))
    logistic_extracted = extract_params(logistic_params)

    # Gamma
    gamma_params = gamma.fit(df)
    gamma_loglik = np.sum(gamma.logpdf(df, *gamma_params))
    gamma_aic = calculate_aic(len(df), gamma_loglik, len(gamma_params))
    gamma_extracted = extract_params(gamma_params)


    #pearson3
    pearson3_params = pearson3.fit(df)
    pearson3_loglik = np.sum(pearson3.logpdf(df, *pearson3_params))
    pearson3_aic = calculate_aic(len(df), pearson3_loglik, len(pearson3_params))
    pearson3_extracted = extract_params(pearson3_params)


    

    # Tính chi-square và p-value cho từng phân phối
    observed_freq, bins = np.histogram(df, bins='auto', density=False)
    
    def chi_square_test(pdf, params):
        expected_freq = pdf(bins[:-1], *params) * len(df) * np.diff(bins)
        chi_square = ((observed_freq - expected_freq) ** 2 / expected_freq).sum()
        p_value = 1 - chi2.cdf(chi_square, len(observed_freq) - len(params))
        return chi_square, p_value

    chi_square_gumbel, p_value_gumbel = chi_square_test(gumbel_r.pdf, gumbel_params)
    chi_square_frechet, p_value_frechet = chi_square_test(genextreme.pdf, frechet_params)
    chi_square_expon, p_value_expon = chi_square_test(expon.pdf, expon_params)
    chi_square_lognormal, p_value_lognormal = chi_square_test(lognorm.pdf, lognormal_params)
    chi_square_logistic, p_value_logistic = chi_square_test(logistic.pdf, logistic_params)
    chi_square_gamma, p_value_gamma = chi_square_test(gamma.pdf, gamma_params)
    chi_square_pearson3, p_value_pearson3 = chi_square_test(pearson3.pdf, pearson3_params)




    checkAgg_func = {"max", "min", "sum", "mean"}
    if agg_func not in checkAgg_func:
        return JSONResponse(status_code=400, content={"message": f"giá trị không nằm trong {checkAgg_func}"})
    analysis = {
        "Gumbel": {
            "params": gumbel_extracted,
            "AIC": gumbel_aic,
            "ChiSquare": chi_square_gumbel,
            "p_value": p_value_gumbel
        },
        "Generalized Extreme Value": {
            "params": frechet_extracted,
            "AIC": frechet_aic,
            "ChiSquare": chi_square_frechet,
            "p_value": p_value_frechet
        },
        "Pearson3": {
            "params": pearson3_extracted,
            "AIC": pearson3_aic,
            "ChiSquare": chi_square_pearson3,
            "p_value": p_value_pearson3
        },
        "Exponential": {
            "params": expon_extracted,
            "AIC": expon_aic,
            "ChiSquare": chi_square_expon,
            "p_value": p_value_expon
        },
        "Lognormal": {
            "params": lognormal_extracted,
            "AIC": lognormal_aic,
            "ChiSquare": chi_square_lognormal,
            "p_value": p_value_lognormal
        },
        "Logistic": {
            "params": logistic_extracted,
            "AIC": logistic_aic,
            "ChiSquare": chi_square_logistic,
            "p_value": p_value_logistic
        },
        "Gamma": {
            "params": gamma_extracted,
            "AIC": gamma_aic,
            "ChiSquare": chi_square_gamma,
            "p_value": p_value_gamma
        },
    }
    return analysis




#------------------------------------------------------

distributions = {
    "gumbel": (gumbel_r.fit, gumbel_r.ppf, gumbel_r.cdf),
    "lognormal": (lognorm.fit, lognorm.ppf, lognorm.cdf),
    "gamma": (gamma.fit, gamma.ppf, gamma.cdf),
    "logistic": (logistic.fit, logistic.ppf, logistic.cdf),
    "exponential": (expon.fit, expon.ppf, expon.cdf),
    # "gpd": (genpareto.fit, genpareto.ppf, genpareto.cdf), # clear
    "frechet": (genextreme.fit, genextreme.ppf, genextreme.cdf),  # Frechet thường được mô hình hóa bằng genextreme
    "pearson3": (pearson3.fit, pearson3.ppf, pearson3.cdf),
    "genextreme": (genextreme.fit, genextreme.ppf, genextreme.cdf),
    
}



#test quantile

def get_quantile_data(request: Request, distribution_name: str, agg_func):
    # Lấy dữ liệu và main_column từ app.state
    data = request.app.state.data
    main_column = request.app.state.main_column

    # Kiểm tra xem dữ liệu đã được tải chưa
    if data is None:
        raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
    
    # Kiểm tra mô hình có hỗ trợ không
    if distribution_name not in distributions:
        raise HTTPException(status_code=404, detail=f"Mô hình {distribution_name} không được hỗ trợ.")
    
    # Tính toán giá trị lớn nhất mỗi năm
    df_max = data.groupby('Year', as_index=False)[main_column].agg(agg_func)
    qmax_values = df_max[main_column].tolist()
    years = df_max['Year'].tolist()
    N = len(qmax_values)
    
    # Tính toán histogram sử dụng bins="auto"
    counts, bin_edges = np.histogram(qmax_values, bins="auto")
    
    # Tính trung điểm cho mỗi bin (dùng để hiển thị)
    bin_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
    
    # Lấy các hàm của phân phối được yêu cầu (fit, ppf, cdf)
    fit_func, ppf_func, cdf_func = distributions[distribution_name]
    
    # Ước lượng tham số cho phân phối dựa trên qmax_values
    params = fit_func(qmax_values)
    
    # Tính số lượng kỳ vọng cho mỗi bin bằng hiệu CDF (chính xác hơn tích phân PDF)
    expected_counts = []
    for i in range(len(bin_edges)-1):
        a = bin_edges[i]
        b = bin_edges[i+1]
        expected_count = N * (cdf_func(b, *params) - cdf_func(a, *params))
        expected_counts.append(expected_count)
    
    # Tính đường cong lý thuyết đầy đủ
    p_values = np.linspace(0.01, 0.99, num=100)
    Q_theoretical = ppf_func(1 - p_values, *params)
    
    return {
        "years": years,
        "qmax_values": qmax_values,
        "histogram": {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bin_midpoints": bin_midpoints,
            "expected_counts": expected_counts
        },
        "theoretical_curve": {
            "p_values": p_values.tolist(),
            "Q_values": Q_theoretical.tolist()
        }
    }

@app.get("/data/quantile_data/{model}")
def call_get_quantile_data(request: Request, model: str, agg_func: str= Query('max')):
    result = get_quantile_data(request, model, agg_func)
    return result



 
#-------------------hàm gom các tính toán lại

def compute_frequency_curve(distribution: str, fit_func, ppf_func, data: pd.DataFrame, main_column: str, agg_func):
    
    Qmax = data.groupby('Year')[main_column].agg(agg_func).values
    
    if Qmax.size == 0:
        return {"theoretical_curve": [], "empirical_points": []}

    # Ước lượng tham số mô hình
    params = fit_func(Qmax)
  #  p_percent_fixed = np.array([0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70 , 80, 90, 99, 99.9 ])
#    Tạo 100 điểm p_percent từ 0.01 đến 99.9 theo thang log
    p_percent_fixed = np.logspace(np.log10(0.01), np.log10(99.9), num=200)
    p_values = p_percent_fixed / 100.0
   # p_values = p_percent_fixed / 100

    # Tính quantile lý thuyết 
    Q_theoretical = ppf_func(1 - p_values, *params)

    # Tính điểm kinh nghiệm theo công thức Weibull
    Q_sorted = sorted(Qmax, reverse=True)
    n = len(Q_sorted)
    m = np.arange(1, n + 1)
    p_empirical = m / (n + 1)
    p_percent_empirical = p_empirical * 100

   # Định dạng kết quả
    theoretical_curve = sorted(
    [{"P_percent": p, "Q": q} for p, q in zip(p_percent_fixed, Q_theoretical)],
    key=lambda item: item["P_percent"]
)
    empirical_points = [{"P_percent": p, "Q": q} for p, q in zip(p_percent_empirical, Q_sorted)]

    return {"theoretical_curve": theoretical_curve, "empirical_points": empirical_points}


# Các endpoint Frequency Curve
@app.get("/analysis/frequency_curve_gumbel")
def get_frequency_curve_gucmbel(request: Request, agg_func: str= Query('max')):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        return JSONResponse(status_code=200, content={
            "message": "Chưa có dữ liệu, vui lòng upload dữ liệu.",
            "theoretical_curve": [],
            "empirical_points": []
        })
    checkAgg_func = {"max", "min", "sum", "mean"}
    if agg_func not in checkAgg_func:
        return JSONResponse(status_code=400, content={"message": f"giá trị không nằm trong {checkAgg_func}"})
    return compute_frequency_curve("gumbel", gumbel_r.fit, gumbel_r.ppf, data, main_column, agg_func)

@app.get("/analysis/frequency_curve_lognorm")
def get_frequency_curve_lognorm(request: Request, agg_func: str= Query('max')):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        return JSONResponse(status_code=200, content={
            "message": "Chưa có dữ liệu, vui lòng upload dữ liệu.",
            "theoretical_curve": [],
            "empirical_points": []
        })
    checkAgg_func = {"max", "min", "sum", "mean"}
    if agg_func not in checkAgg_func:
        return JSONResponse(status_code=400, content={"message": f"giá trị không nằm trong {checkAgg_func}"})
    return compute_frequency_curve("lognormal", lognorm.fit, lognorm.ppf, data, main_column, agg_func)

@app.get("/analysis/frequency_curve_gamma")
def get_frequency_curve_gamma(request: Request, agg_func: str= Query('max')):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        return JSONResponse(status_code=200, content={
            "message": "Chưa có dữ liệu, vui lòng upload dữ liệu.",
            "theoretical_curve": [],
            "empirical_points": []
        })
    checkAgg_func = {"max", "min", "sum", "mean"}
    if agg_func not in checkAgg_func:
        return JSONResponse(status_code=400, content={"message": f"giá trị không nằm trong {checkAgg_func}"})
    return compute_frequency_curve("Gamma", gamma.fit, gamma.ppf, data, main_column,agg_func)

@app.get("/analysis/frequency_curve_logistic")
def get_frequency_curve_logistic(request: Request , agg_func: str= Query('max')):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        return JSONResponse(status_code=200, content={
            "message": "Chưa có dữ liệu, vui lòng upload dữ liệu.",
            "theoretical_curve": [],
            "empirical_points": []
        })
    checkAgg_func = {"max", "min", "sum", "mean"}
    if agg_func not in checkAgg_func:
        return JSONResponse(status_code=400, content={"message": f"giá trị không nằm trong {checkAgg_func}"})
    return compute_frequency_curve("Logistic", logistic.fit, logistic.ppf, data, main_column, agg_func)

@app.get("/analysis/frequency_curve_exponential")
def get_frequency_curve_exponential(request: Request, agg_func: str= Query('max')):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        return JSONResponse(status_code=200, content={
            "message": "Chưa có dữ liệu, vui lòng upload dữ liệu.",
            "theoretical_curve": [],
            "empirical_points": []
        })
    checkAgg_func = {"max", "min", "sum", "mean"}
    if agg_func not in checkAgg_func:
        return JSONResponse(status_code=400, content={"message": f"giá trị không nằm trong {checkAgg_func}"})
    return compute_frequency_curve("Exponential", expon.fit, expon.ppf, data, main_column, agg_func)

@app.get("/analysis/frequency_curve_gpd")
def get_frequency_curve_gpd(request: Request, agg_func: str= Query('max')):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        return JSONResponse(status_code=200, content={
            "message": "Chưa có dữ liệu, vui lòng upload dữ liệu.",
            "theoretical_curve": [],
            "empirical_points": []
        })
    checkAgg_func = {"max", "min", "sum", "mean"}
    if agg_func not in checkAgg_func:
        return JSONResponse(status_code=400, content={"message": f"giá trị không nằm trong {checkAgg_func}"})

    return compute_frequency_curve("GPD", genpareto.fit, genpareto.ppf, data, main_column, agg_func)

@app.get("/analysis/frequency_curve_frechet")
def get_frequency_curve_frechet(request: Request, agg_func: str= Query('max')):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        return JSONResponse(status_code=200, content={
            "message": "Chưa có dữ liệu, vui lòng upload dữ liệu.",
            "theoretical_curve": [],
            "empirical_points": []
        })
    checkAgg_func = {"max", "min", "sum", "mean"}
    
    if agg_func not in checkAgg_func:
        return JSONResponse(status_code=400, content={"message": f"giá trị không nằm trong {checkAgg_func}"})
    return compute_frequency_curve("Frechet", genextreme.fit, genextreme.ppf, data, main_column, agg_func)


@app.get("/analysis/frequency_curve_pearson3")
def get_frequency_curve_pearson3(request: Request, agg_func: str= Query('max')):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        return JSONResponse(status_code=200, content={
            "message": "Chưa có dữ liệu, vui lòng upload dữ liệu.",
            "theoretical_curve": [],
            "empirical_points": []
        })
    checkAgg_func = {"max", "min", "sum", "mean"}
    if agg_func not in checkAgg_func:
        return JSONResponse(status_code=400, content={"message": f"giá trị không nằm trong {checkAgg_func}"})
    return compute_frequency_curve("Pearson3", pearson3.fit, pearson3.ppf, data, main_column, agg_func)
    


@app.get("/analysis/frequency_curve_genextreme")
def get_frequency_curve_genextreme(request: Request, agg_func: str= Query('max')):
    data = request.app.state.data
    main_column = request.app.state.main_column
    if data is None:
        return JSONResponse(status_code=200, content={
            "message": "Chưa có dữ liệu, vui lòng upload dữ liệu.",
            "theoretical_curve": [],
            "empirical_points": []
        })
    checkAgg_func = {"max", "min", "sum", "mean"}
    if agg_func not in checkAgg_func:
        return JSONResponse(status_code=400, content={"message": f"giá trị không nằm trong {checkAgg_func}"})
    return compute_frequency_curve("genextreme", genextreme.fit, genextreme.ppf, data, main_column, agg_func)






#-------------------------------------
def compute_qq_pp(distribution_name: str, data: pd.DataFrame, main_column: str, agg_func):
    if data is None:
        raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")

    try:
        Qmax = data.groupby('Year')[main_column].agg(agg_func).values
    except Exception as e:
        logging.error(f"Lỗi khi tính Qmax: {e}")
        raise HTTPException(status_code=500, detail="Lỗi khi tính Qmax")
    
    if Qmax.size == 0:
        return {"qq": [], "pp": []}
    
    # Kiểm tra xem mô hình có được hỗ trợ không
    if distribution_name not in distributions:
        raise HTTPException(status_code=400, detail=f"Mô hình {distribution_name} không được hỗ trợ")
    
    fit_func, ppf_func, cdf_func = distributions[distribution_name]
    
    # Ước lượng tham số
    params = fit_func(Qmax)
    # Sắp xếp dữ liệu mẫu theo thứ tự tăng dần
    sorted_Q = np.sort(Qmax)
    n = len(sorted_Q)
    
    qq_data = []
    pp_data = []
    for i in range(n):
        p_empirical = (i + 1) / (n + 1)  # quantile theo công thức Weibull
        theoretical_quantile = ppf_func(p_empirical, *params)
        empirical_cdf = p_empirical
        theoretical_cdf = cdf_func(sorted_Q[i], *params)
        qq_data.append({
            "p_empirical": p_empirical,
            "sample": sorted_Q[i],
            "theoretical": theoretical_quantile
        })
        pp_data.append({
            "empirical": empirical_cdf,
            "theoretical": theoretical_cdf
        })
    
    return {"qq": qq_data, "pp": pp_data}

@app.get("/analysis/qq_pp/{model}")
def get_qq_pp_plot_data(request: Request, model: str, agg_func: str= Query('max')):
    """
    Tính toán dữ liệu cho biểu đồ QQ và PP dựa trên Qmax từ dữ liệu.
    model: tên mô hình (ví dụ: "gumbel", "lognormal", "gamma", "logistic", "exponential", "gpd", "frechet")
    """
    data = request.app.state.data
    main_column = request.app.state.main_column
    result = compute_qq_pp(model, data, main_column, agg_func)
    return result

#-------------------------------------
@app.get("/analysis/frequency")
def get_frequency_analysis(request: Request):
    data = request.app.state.data
    main_column = request.app.state.main_column
    
    if data is None:
        raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
    
    # Nhóm dữ liệu theo năm và lấy giá trị lớn nhất (Qmax) cho mỗi năm.
    df = data.groupby('Year', as_index=False).agg({main_column: 'max'})
    
    # Tạo cột "Thời gian": ví dụ "1956-1957"
    df["Thời gian"] = df["Year"].astype(str) + "-" + (df["Year"] + 1).astype(str)
    
    # Tính thứ hạng dựa trên giá trị Q (giá trị lớn nhất nhận thứ hạng 1)
    df['Thứ hạng'] = df[main_column].rank(ascending=False, method='min').astype(int)
    
    n = len(df)
    # Tính tần suất P(%) theo công thức Weibull:
    # P(%) = (Thứ hạng / (n + 1)) * 100
    df["Tần suất P(%)"] = (df['Thứ hạng'] / (n + 1)) * 100
    
    # Sắp xếp theo năm tăng dần để hiển thị theo thời gian
    df = df.sort_values("Year").reset_index(drop=True)
    # Tạo cột "Thứ tự" theo thứ tự thời gian
    df["Thứ tự"] = df.index + 1
    
    # Đổi tên cột chứa dữ liệu chính thành "Chỉ số"
    df = df.rename(columns={main_column: "Chỉ số"})
    
    # Chọn các cột cần trả về
    output_df = df[["Thứ tự", "Thời gian", "Chỉ số", "Tần suất P(%)", "Thứ hạng"]]
    
    # Làm tròn số nếu cần
    output_df.loc[:, "Tần suất P(%)"] = output_df["Tần suất P(%)"].round(2)
    output_df.loc[:, "Chỉ số"] = output_df["Chỉ số"].round(2)

    
    return output_df.to_dict(orient="records")

#-----------------------
@app.get("/analysis/frequency_by_model")
def get_frequency_by_model(request: Request, distribution_name: str, agg_func: str= Query('max')):
    """
    Tính toán và trả về bảng tần suất theo mô hình cho mô hình được chỉ định.
    
    Endpoint nhận một tham số:
      - distribution_name: tên mô hình (ví dụ "Gumbel")
      
    Kết quả trả về là một dictionary chứa:
      - theoretical_curve: danh sách các điểm theo đường cong lý thuyết
      - empirical_points: danh sách các điểm kinh nghiệm
      
    Phần trăm cho đường lý thuyết được cố định theo:
      [0.01, 0.10, 0.20, 0.33, 0.50, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00, 
       20.00, 25.00, 30.00, 40.00, 50.00, 60.00, 70.00, 75.00, 80.00, 85.00, 
       90.00, 95.00, 97.00, 99.00, 99.90, 99.99]
    """
    # Lấy dữ liệu và tên cột chính từ app.state
    data = request.app.state.data
    main_column = request.app.state.main_column
    
    if data is None:
        raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
    
    # Tính Qmax (lưu lượng mưa lớn nhất mỗi năm)
    try:
        Qmax = data.groupby('Year')[main_column].agg(agg_func).values
    except Exception as e:
        logging.error(f"Lỗi khi tính Qmax: {e}")
        raise HTTPException(status_code=500, detail="Lỗi khi tính Qmax")
    
    if Qmax.size == 0:
        return {}
    
    # Kiểm tra xem mô hình có được hỗ trợ không
    if distribution_name not in distributions:
        raise HTTPException(status_code=400, detail=f"Mô hình {distribution_name} không được hỗ trợ")
    
    # Lấy các hàm của mô hình: fit, ppf, cdf
    fit_func, ppf_func, cdf_func = distributions[distribution_name]
    
    # Ước lượng tham số từ Qmax
    params = fit_func(Qmax)
    # extracted_params = extract_params(params)
    # --- Tính đường cong lý thuyết ---
    fixed_p_percent = np.array([
        0.01, 0.10, 0.20, 0.33, 0.50, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00,
        20.00, 25.00, 30.00, 40.00, 50.00, 60.00, 70.00, 75.00, 80.00,
        85.00, 90.00, 95.00, 97.00, 99.00, 99.90, 99.99
    ])
    p_values = fixed_p_percent / 100.0
    # Tính Q lý thuyết theo hàm nghịch đảo của CDF của mô hình (lưu ý: dùng 1 - p)
    Q_theoretical = ppf_func(1 - p_values, *params)
    # Tính thời gian lặp lại theo công thức: T = 100 / P_percent
    T_theoretical = 100 / fixed_p_percent
    
    theoretical_curve = [
        {
            "Thứ tự": i,
            "Tần suất P(%)": f"{p:.2f}",
            "Lưu lượng dòng chảy Q m³/s": f"{q:.2f}",
            "Thời gian lặp lại (năm)": f"{T:.3f}"
        }
        for i, (p, q, T) in enumerate(zip(fixed_p_percent, Q_theoretical, T_theoretical), start=1)
    ]
    
    # --- Tính điểm kinh nghiệm ---
    # Sắp xếp Qmax theo thứ tự giảm dần (Q lớn nhất ở đầu)
    Q_sorted_desc = sorted(Qmax, reverse=True)
    n = len(Q_sorted_desc)
    # Gán thứ hạng: Q lớn nhất nhận rank 1, Q nhỏ nhất nhận rank n
    ranks = np.arange(1, n + 1)
    # Tần suất theo công thức Weibull: p_empirical = rank / (n+1)
    p_empirical = ranks / (n + 1)
    p_percent_empirical = p_empirical * 100
    # Tính thời gian lặp lại cho điểm kinh nghiệm: T = (n+1) / rank
    T_empirical = (n + 1) / ranks
    
    empirical_points = [
        {
            "Thứ tự": i,
            "Tần suất P(%)": f"{p:.2f}",
            "Lưu lượng dòng chảy Q m³/s": f"{q:.2f}",
            "Thời gian lặp lại (năm)": f"{T:.3f}"
        }
        for i, (p, q, T) in enumerate(zip(p_percent_empirical, Q_sorted_desc, T_empirical), start=1)
    ]
    
    
    return {
        "theoretical_curve": theoretical_curve,
        "empirical_points": empirical_points,
        # "extracted_params": extracted_params
    }

@app.get("/data/nasa_power/raw")
def get_nasa_power_data(start_year: int, end_year: int, lat: float, lon: float):
    """
    Lấy dữ liệu thời tiết monthly từ NASA POWER API cho khoảng thời gian từ start_year đến end_year và vị trí (lat, lon).

    Ví dụ sử dụng:
      GET http://127.0.0.1:8000/data/nasa_power/raw?start_year=2012&end_year=2012&lat=40.713&lon=-74.0060
    """
    if start_year > end_year:
        raise HTTPException(status_code=400, detail="start_year phải nhỏ hơn hoặc bằng end_year")
    
    # Định dạng ngày cho monthly: chỉ dùng năm (YYYY)
    start_date = str(start_year).strip()  # Ví dụ: "2012"
    end_date = str(end_year).strip()      # Ví dụ: "2012"
    
    base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    parameters = "T2M"  # Lấy dữ liệu nhiệt độ
    community = "RE"
    response_format = "JSON"
    
    url = (
        f"{base_url}?parameters={parameters}"
        f"&community={community}"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start_date}&end={end_date}"
        f"&format={response_format}"
    )
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        detail_msg = f"HTTP error occurred: {http_err} - Response: {response.text if response is not None else 'No response'}"
        raise HTTPException(status_code=response.status_code if response else 500, detail=detail_msg)
    except requests.exceptions.RequestException as err:
        raise HTTPException(status_code=500, detail=f"Error occurred: {err}")
    
    try:
        data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing JSON: {e}")
    
    return data

def clean_nasa_power_data(json_data: dict) -> pd.DataFrame:
    """
    Chuyển đổi dữ liệu JSON từ NASA POWER API thành DataFrame với 3 cột: 
    'Year', 'Month', và 'Temperature' (lấy từ T2M).
    Chỉ lấy các key có định dạng "YYYYMM" với tháng từ 01 đến 12 (bỏ qua các key như "201213").
    """
    try:
        t2m_data = json_data["properties"]["parameter"]["T2M"]
    except KeyError as e:
        raise ValueError(f"Không tìm thấy dữ liệu T2M trong JSON: {e}")
    
    records = []
    for key, value in t2m_data.items():
        # Key phải có 6 ký tự: YYYYMM
        if len(key) == 6:
            year = int(key[:4])
            month = int(key[4:])
            if 1 <= month <= 12:
                records.append({
                    "Year": year,
                    "Month": month,
                    "Temperature": value
                })
    if not records:
        raise ValueError("Không có dữ liệu hợp lệ được tìm thấy.")
    df = pd.DataFrame(records)
    df = df.sort_values(by=["Year", "Month"])
    return df

import io
from fastapi.responses import StreamingResponse

@app.get("/data/nasa_power/clean")
def process_nasa_power_data(start_year: int, end_year: int, lat: float, lon: float):
    # Gọi nội bộ endpoint raw để lấy dữ liệu
    raw_data = get_nasa_power_data(start_year, end_year, lat, lon)
    
    try:
        df = clean_nasa_power_data(raw_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Tạo CSV trong bộ nhớ thay vì lưu ra ổ đĩa
    stream = io.StringIO()
    df.to_csv(stream, index=False, encoding="utf-8")
    stream.seek(0)  # Quay lại đầu file để đọc

    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=nasa_power_temperature.csv"}
    )


timestamp = datetime.now(timezone.utc)


MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["visits_db"]
visits_collection = db["visits"]

# Endpoint ghi nhận lượt truy cập
@app.post("/visit")
async def record_visit():
    new_visit = {"timestamp": datetime.now(timezone.utc)}
    try:
        result = await visits_collection.insert_one(new_visit)
        return {"message": "Visit recorded", "visit_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint thống kê lượt truy cập theo ngày và theo tháng
@app.get("/stats-visit")
async def get_stats():
    try:
        # Tổng số lượt truy cập
        total_visits = await visits_collection.count_documents({})

        # Aggregation pipeline cho thống kê theo ngày
        pipeline_daily = [
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        daily_cursor = visits_collection.aggregate(pipeline_daily)
        daily_stats = {}
        async for doc in daily_cursor:
            daily_stats[doc["_id"]] = doc["count"]

        # Aggregation pipeline cho thống kê theo tháng
        pipeline_monthly = [
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m", "date": "$timestamp"}
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        monthly_cursor = visits_collection.aggregate(pipeline_monthly)
        monthly_stats = {}
        async for doc in monthly_cursor:
            monthly_stats[doc["_id"]] = doc["count"]

        return {
            "total_visits": total_visits,
            "daily_stats": daily_stats,
            "monthly_stats": monthly_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




