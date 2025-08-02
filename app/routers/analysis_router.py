from fastapi import APIRouter, Depends, Query, Path
from ..dependencies import get_analysis_service
from ..services.analysis_service import AnalysisService

router = APIRouter(prefix="/analysis", tags=["analysis"])

@router.get("/distribution")
def get_distribution_analysis(agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.get_distribution_analysis(agg_func)

@router.get("/quantile_data/{model}")
def call_get_quantile_data(model: str = Path(...), agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.get_quantile_data(model, agg_func)

@router.get("/frequency_curve_gumbel")
def get_frequency_curve_gumbel(agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.compute_frequency_curve("gumbel", agg_func)

@router.get("/frequency_curve_lognorm")
def get_frequency_curve_lognorm(agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.compute_frequency_curve("lognorm", agg_func)

@router.get("/frequency_curve_gamma")
def get_frequency_curve_gamma(agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.compute_frequency_curve("gamma", agg_func)

@router.get("/frequency_curve_logistic")
def get_frequency_curve_logistic(agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.compute_frequency_curve("logistic", agg_func)

@router.get("/frequency_curve_exponential")
def get_frequency_curve_exponential(agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.compute_frequency_curve("expon", agg_func)

@router.get("/frequency_curve_gpd")
def get_frequency_curve_gpd(agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.compute_frequency_curve("genpareto", agg_func)

@router.get("/frequency_curve_frechet")
def get_frequency_curve_frechet(agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.compute_frequency_curve("frechet", agg_func)

@router.get("/frequency_curve_pearson3")
def get_frequency_curve_pearson3(agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.compute_frequency_curve("pearson3", agg_func)

@router.get("/frequency_curve_genextreme")
def get_frequency_curve_genextreme(agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.compute_frequency_curve("genextreme", agg_func)

@router.get("/qq_pp/{model}")
def get_qq_pp_plot_data(model: str = Path(...), agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.compute_qq_pp(model, agg_func)

@router.get("/frequency")
def get_frequency_analysis(analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.get_frequency_analysis()

@router.get("/frequency_by_model")
def get_frequency_by_model(distribution_name: str = Query(...), agg_func: str = Query('max'), analysis_service: AnalysisService = Depends(get_analysis_service)):
    return analysis_service.get_frequency_by_model(distribution_name, agg_func)