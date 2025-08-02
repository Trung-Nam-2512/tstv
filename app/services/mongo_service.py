import motor.motor_asyncio
from typing import Dict, Any
from datetime import datetime, date
import logging
from ..config import config

class MongoService:
    def __init__(self):
        self.client = None
        self.db = None
        self.visit_collection = None
        self._connect()

    def _connect(self):
        """Kết nối đến MongoDB Atlas"""
        try:
            if not config.MONGO_URI:
                raise ValueError("MONGO_URI không được cấu hình trong .env")
            
            self.client = motor.motor_asyncio.AsyncIOMotorClient(config.MONGO_URI)
            self.db = self.client.visits_db  # Sử dụng database có sẵn
            self.visit_collection = self.db.visits  # Sử dụng collection có sẵn
            logging.info("Kết nối MongoDB Atlas thành công")
        except Exception as e:
            logging.error(f"Lỗi kết nối MongoDB: {str(e)}")
            raise

    async def record_visit(self) -> bool:
        """Ghi lại lượt truy cập"""
        try:
            now = datetime.utcnow()
            
            # Tạo document mới cho mỗi lượt truy cập
            await self.visit_collection.insert_one({
                "timestamp": now
            })
            
            logging.info(f"Đã ghi lại lượt truy cập: {now}")
            return True
        except Exception as e:
            logging.error(f"Lỗi ghi lượt truy cập: {str(e)}")
            return False

    async def get_visit_stats(self) -> Dict[str, Any]:
        """Lấy thống kê lượt truy cập"""
        try:
            # Lấy tổng số lượt truy cập
            total_visits = await self.visit_collection.count_documents({})
            
            # Lấy thống kê theo ngày (7 ngày gần nhất)
            from datetime import timedelta
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            
            # Pipeline để tính số lượt truy cập theo ngày
            pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": seven_days_ago}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": "$timestamp"
                            }
                        },
                        "daily_visits": {"$sum": 1}
                    }
                },
                {
                    "$sort": {"_id": 1}
                }
            ]
            
            recent_stats = await self.visit_collection.aggregate(pipeline).to_list(7)
            
            # Tạo dict cho daily_stats
            daily_stats = {}
            for stat in recent_stats:
                daily_stats[stat["_id"]] = stat["daily_visits"]
            
            return {
                "total_visits": total_visits,
                "daily_stats": daily_stats
            }
        except Exception as e:
            logging.error(f"Lỗi lấy thống kê lượt truy cập: {str(e)}")
            return {
                "total_visits": 0,
                "daily_stats": {}
            }

    async def close_connection(self):
        """Đóng kết nối MongoDB"""
        if self.client:
            self.client.close()
            logging.info("Đã đóng kết nối MongoDB") 