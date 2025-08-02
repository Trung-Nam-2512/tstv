import pandas as pd
import numpy as np
from scipy.stats import gumbel_r, genextreme, genpareto, expon, lognorm, logistic, gamma, chi2, pearson3
from fastapi import HTTPException
from starlette.responses import JSONResponse
from typing import Dict, Tuple, Callable, List, Any
from .data_service import DataService
from ..utils.helpers import extract_params, validate_agg_func
from datetime import datetime, timezone
import logging

class DistributionBase:
    def __init__(self, name: str, fit_func: Callable, ppf_func: Callable, cdf_func: Callable, pdf_func: Callable, logpdf_func: Callable):
        self.name = name
        self.fit = fit_func
        self.ppf = ppf_func
        self.cdf = cdf_func
        self.pdf = pdf_func
        self.logpdf = logpdf_func

distributions: Dict[str, DistributionBase] = {
    "gumbel": DistributionBase("Gumbel", gumbel_r.fit, gumbel_r.ppf, gumbel_r.cdf, gumbel_r.pdf, gumbel_r.logpdf),
    "genextreme": DistributionBase("Generalized Extreme Value", genextreme.fit, genextreme.ppf, genextreme.cdf, genextreme.pdf, genextreme.logpdf),
    "genpareto": DistributionBase("GPD", genpareto.fit, genpareto.ppf, genpareto.cdf, genpareto.pdf, genpareto.logpdf),
    "expon": DistributionBase("Exponential", expon.fit, expon.ppf, expon.cdf, expon.pdf, expon.logpdf),
    "lognorm": DistributionBase("Lognormal", lognorm.fit, lognorm.ppf, lognorm.cdf, lognorm.pdf, lognorm.logpdf),
    "logistic": DistributionBase("Logistic", logistic.fit, logistic.ppf, logistic.cdf, logistic.pdf, logistic.logpdf),
    "gamma": DistributionBase("Gamma", gamma.fit, gamma.ppf, gamma.cdf, gamma.pdf, gamma.logpdf),
    "pearson3": DistributionBase("Pearson3", pearson3.fit, pearson3.ppf, pearson3.cdf, pearson3.pdf, pearson3.logpdf),
    "frechet": DistributionBase("Frechet", genextreme.fit, genextreme.ppf, genextreme.cdf, genextreme.pdf, genextreme.logpdf),
}

class AnalysisService:
    def __init__(self, data_service: DataService):
        self.data_service = data_service

    def get_distribution_analysis(self, agg_func: str= 'max'):
        """
        Phân tích xác định phân phối và tính AIC, chi-square cho các phân phối khác nhau
        - Fix 1: validate_agg_func để DRY.
        - Fix 2: Expected_freq dùng CDF exact (cdf(b)-cdf(a))*n → sum~n chính xác hơn PDF approx (giảm bias skewed dist thủy văn).
        - Fix 3: df_chi = len(observed) - 1 - len(params) (chuẩn goodness-of-fit, tránh df cao → p-value inflated).
        - Fix 4: Handle expected<=0 bằng epsilon 1e-10 để tránh div0.
        - Fix 5: Nếu df_chi <=0 (small n/bins), p_value=None + warning (thay vì nan).
        - Fix 6: Bins dynamic: max(5, sturges formula) để int, tránh ít bin → df<0.
        - Lý do: Trong thủy văn, small n (<30 năm) phổ biến → cần robust; CDF tốt hơn cho continuous data.
        - Warning nếu n<30 hoặc df<=0 để alert user.
        """
        df = self.data_service.data
        main_column = self.data_service.main_column
        if df is None:
            raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
        validate_agg_func(agg_func)
        aggregated = df.groupby('Year')[main_column].agg(agg_func).values

        # Tính sturges bins: 1 + log2(n+1)
        n = len(aggregated)
        sturges_bins = int(np.ceil(1 + np.log2(n + 1))) if n > 0 else 5
        num_bins = max(5, sturges_bins)

        analysis = {}
        for name, dist in distributions.items():
            params = dist.fit(aggregated)
            extracted = extract_params(params)
            loglik = np.sum(dist.logpdf(aggregated, *params))
            aic = 2 * len(params) - 2 * loglik
            bins = np.histogram_bin_edges(aggregated, bins=num_bins)
            observed_freq, _ = np.histogram(aggregated, bins=bins)
            expected_freq = n * (dist.cdf(bins[1:], *params) - dist.cdf(bins[:-1], *params))
            expected_freq = np.where(expected_freq <= 0, 1e-10, expected_freq)
            chi_square = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
            df_chi = len(observed_freq) - 1 - len(params)
            p_value = 1 - chi2.cdf(chi_square, df_chi) if df_chi > 0 else None
            if n < 30 or df_chi <= 0:
                logging.warning(f"Small sample or low df for {name}: n={n}, df={df_chi}. Consider more data for reliable fit.")
            analysis[name] = {
                "params": extracted,
                "AIC": aic,
                "ChiSquare": chi_square,
                "p_value": p_value
            }
        return analysis

    def get_quantile_data(self, distribution_name: str, agg_func: str= 'max'):
        validate_agg_func(agg_func)
        if distribution_name not in distributions:
            raise HTTPException(status_code=404, detail=f"Mô hình {distribution_name} không được hỗ trợ.")
        
        df = self.data_service.data
        main_column = self.data_service.main_column

        if df is None:
            raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
        
        df_max = df.groupby('Year', as_index=False)[main_column].agg(agg_func)
        qmax_values = df_max[main_column].tolist()
        years = df_max['Year'].tolist()
        N = len(qmax_values)
        
        counts, bin_edges = np.histogram(qmax_values, bins="auto")
        
        bin_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        
        dist = distributions[distribution_name]
        
        params = dist.fit(qmax_values)
        
        expected_counts = []
        for i in range(len(bin_edges)-1):
            a = bin_edges[i]
            b = bin_edges[i+1]
            expected_count = N * (dist.cdf(b, *params) - dist.cdf(a, *params))
            expected_counts.append(expected_count)
        
        p_values = np.linspace(0.01, 0.99, num=100)
        Q_theoretical = dist.ppf(1 - p_values, *params)
        
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

    def compute_frequency_curve(self, distribution_name: str, agg_func: str= 'max'):
        validate_agg_func(agg_func)
        if distribution_name not in distributions:
            raise HTTPException(status_code=404, detail=f"Mô hình {distribution_name} không được hỗ trợ.")
        
        df = self.data_service.data
        main_column = self.data_service.main_column
        
        if df is None:
            return {"theoretical_curve": [], "empirical_points": []}
        
        Qmax = df.groupby('Year')[main_column].agg(agg_func).values
        
        if Qmax.size == 0:
            return {"theoretical_curve": [], "empirical_points": []}

        dist = distributions[distribution_name]
        params = dist.fit(Qmax)
        
        p_percent_fixed = np.logspace(np.log10(0.01), np.log10(99.9), num=200)
        p_values = p_percent_fixed / 100.0
        
        Q_theoretical = dist.ppf(1 - p_values, *params)
        
        Q_sorted = sorted(Qmax, reverse=True)
        n = len(Q_sorted)
        m = np.arange(1, n + 1)
        p_empirical = m / (n + 1)
        p_percent_empirical = p_empirical * 100

        theoretical_curve = sorted(
            [{"P_percent": p, "Q": q} for p, q in zip(p_percent_fixed, Q_theoretical)],
            key=lambda item: item["P_percent"]
        )
        empirical_points = [{"P_percent": p, "Q": q} for p, q in zip(p_percent_empirical, Q_sorted)]

        return {"theoretical_curve": theoretical_curve, "empirical_points": empirical_points}

    def compute_qq_pp(self, distribution_name: str, agg_func: str= 'max'):
        validate_agg_func(agg_func)
        if distribution_name not in distributions:
            raise HTTPException(status_code=400, detail=f"Mô hình {distribution_name} không được hỗ trợ.")
        
        df = self.data_service.data
        main_column = self.data_service.main_column
        
        if df is None:
            raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
        
        Qmax = df.groupby('Year')[main_column].agg(agg_func).values
        
        if Qmax.size == 0:
            return {"qq": [], "pp": []}
        
        dist = distributions[distribution_name]
        
        params = dist.fit(Qmax)
        
        sorted_Q = np.sort(Qmax)
        n = len(sorted_Q)
        
        qq_data = []
        pp_data = []
        for i in range(n):
            p_empirical = (i + 1) / (n + 1)
            theoretical_quantile = dist.ppf(p_empirical, *params)
            empirical_cdf = p_empirical
            theoretical_cdf = dist.cdf(sorted_Q[i], *params)
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

    def get_frequency_analysis(self):
        df = self.data_service.data
        main_column = self.data_service.main_column
        
        if df is None:
            raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
        
        agg_df = df.groupby('Year', as_index=False).agg({main_column: 'max'})
        
        agg_df["Thời gian"] = agg_df["Year"].astype(str) + "-" + (agg_df["Year"] + 1).astype(str)
        
        agg_df['Thứ hạng'] = agg_df[main_column].rank(ascending=False, method='min').astype(int)
        
        n = len(agg_df)
        agg_df["Tần suất P(%)"] = (agg_df['Thứ hạng'] / (n + 1)) * 100
        
        agg_df = agg_df.sort_values("Year").reset_index(drop=True)
        agg_df["Thứ tự"] = agg_df.index + 1
        
        agg_df = agg_df.rename(columns={main_column: "Chỉ số"})
        
        output_df = agg_df[["Thứ tự", "Thời gian", "Chỉ số", "Tần suất P(%)", "Thứ hạng"]]
        
        output_df.loc[:, "Tần suất P(%)"] = output_df["Tần suất P(%)"].round(2)
        output_df.loc[:, "Chỉ số"] = output_df["Chỉ số"].round(2)

        return output_df.to_dict(orient="records")

    def get_frequency_by_model(self, distribution_name: str, agg_func: str= 'max'):
        validate_agg_func(agg_func)
        if distribution_name not in distributions:
            raise HTTPException(status_code=400, detail=f"Mô hình {distribution_name} không được hỗ trợ.")
        
        df = self.data_service.data
        main_column = self.data_service.main_column
        
        if df is None:
            raise HTTPException(status_code=404, detail="Dữ liệu chưa được tải")
        
        Qmax = df.groupby('Year')[main_column].agg(agg_func).values
        
        if Qmax.size == 0:
            return {}
        
        dist = distributions[distribution_name]
        
        params = dist.fit(Qmax)
        
        fixed_p_percent = np.array([
            0.01, 0.10, 0.20, 0.33, 0.50, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00,
            20.00, 25.00, 30.00, 40.00, 50.00, 60.00, 70.00, 75.00, 80.00,
            85.00, 90.00, 95.00, 97.00, 99.00, 99.90, 99.99
        ])
        p_values = fixed_p_percent / 100.0
        
        Q_theoretical = dist.ppf(1 - p_values, *params)
        
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
        
        Q_sorted_desc = sorted(Qmax, reverse=True)
        n = len(Q_sorted_desc)
        
        ranks = np.arange(1, n + 1)
        
        p_empirical = ranks / (n + 1)
        p_percent_empirical = p_empirical * 100
        
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
        }