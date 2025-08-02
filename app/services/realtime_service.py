import httpx
from tenacity import retry, stop_after_attempt, wait_fixed
from fastapi import HTTPException
import pandas as pd
from urllib.parse import urlencode
from typing import List
from motor.motor_asyncio import AsyncIOMotorClient
from apscheduler.schedulers.background import BackgroundScheduler
from ..config import config
from ..models.data_models import Station, RealTimeQuery, RealTimeResponse
from .data_service import DataService
import logging

class RealTimeService:
    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self.mongo_client = AsyncIOMotorClient(config.MONGO_URI)
        self.db = self.mongo_client["hydro_db"]  # New DB for realtime history
        self.collection = self.db["realtime_depth"]
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()  # Auto-poll setup

    async def fetch_stations(self) -> List[Station]:
        """
        Fetch list trạm đo.
        - Use header auth with API_KEY.
        - Return list Station.
        """
        url = config.STATIONS_API_BASE_URL
        async with httpx.AsyncClient() as client:
            headers = {"X-API-Key": config.API_KEY} if config.API_KEY else {}
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Stations API error")
            data = response.json()
            return [Station(**item) for item in data]

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def fetch_water_level(self, query: RealTimeQuery) -> RealTimeResponse:
        """
        Fetch data thời gian thực (up to 2 months).
        - Add station_id if provided.
        - Encode params.
        - Header auth.
        """
        params = {"start_time": query.start_time, "end_time": query.end_time}
        if query.station_id:
            params["station_id"] = query.station_id
        encoded_params = urlencode(params, safe=': ')
        url = f"{config.STATS_API_BASE_URL}?{encoded_params}"
        async with httpx.AsyncClient() as client:
            headers = {"X-API-Key": config.API_KEY} if config.API_KEY else {}
            response = await client.get(url, headers=headers)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Stats API error")
            data = response.json()
            return RealTimeResponse(**data)

    def process_to_df(self, api_data: RealTimeResponse) -> pd.DataFrame:
        """
        Parse to DF, add derived columns.
        - Filter depth >0.
        - Agg max per year/station (for frequency extremes).
        - Check alert if max depth > threshold.
        """
        all_measurements = []
        for station_data in api_data.Data:
            for meas in station_data.value:
                meas_dict = meas.dict()
                meas_dict['station_id'] = station_data.station_id
                all_measurements.append(meas_dict)
        if not all_measurements:
            logging.warning("No measurements")
            return pd.DataFrame()
        df = pd.DataFrame(all_measurements)
        df['time_point'] = pd.to_datetime(df['time_point'])
        df['Year'] = df['time_point'].dt.year
        df['Month'] = df['time_point'].dt.month
        df['Day'] = df['time_point'].dt.day
        df['Hour'] = df['time_point'].dt.hour
        df['Minute'] = df['time_point'].dt.minute
        df = df[df['depth'] > 0]
        # Agg max per year/station for frequency
        agg_df = df.groupby(['station_id', 'Year'])['depth'].agg('max').reset_index(name='depth')
        # Alert check
        if agg_df['depth'].max() > config.DEPTH_THRESHOLD:
            logging.warning(f"Alert: Depth exceeded threshold {config.DEPTH_THRESHOLD} m")
        return agg_df  # Return agg for analysis

    async def integrate_to_analysis(self, df: pd.DataFrame):
        """
        Append to DataService, set main_column='depth'.
        - Save to Mongo for history (up to 2 months query).
        """
        if df.empty:
            return
        # Save to Mongo
        await self.collection.insert_many(df.to_dict('records'))
        if self.data_service.data is not None:
            combined_df = pd.concat([self.data_service.data, df], ignore_index=True)
            self.data_service.data = combined_df
        else:
            self.data_service.data = df
        self.data_service.main_column = 'depth'

    def setup_auto_poll(self):
        """
        Auto fetch every 10 min for all stations (or specific).
        - Query last 10 min, integrate.
        - Start scheduler on init if needed.
        """
        async def poll_job():
            query = RealTimeQuery(start_time=(datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"),
                                  end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            api_data = await self.fetch_water_level(query)
            df = self.process_to_df(api_data)
            self.integrate_to_analysis(df)
        self.scheduler.add_job(poll_job, 'interval', minutes=10)