from fastapi import Depends
from .services.data_service import DataService
from .services.stats_service import StatsService
from .services.analysis_service import AnalysisService
from .services.mongo_service import MongoService

# Singleton instance cho DataService
_data_service_instance = None

def get_data_service() -> DataService:
    global _data_service_instance
    if _data_service_instance is None:
        _data_service_instance = DataService()
    return _data_service_instance

def get_stats_service() -> StatsService:
    data_service = get_data_service()
    return StatsService(data_service)

def get_analysis_service() -> AnalysisService:
    data_service = get_data_service()
    return AnalysisService(data_service)

def get_mongo_service() -> MongoService:
    return MongoService()
