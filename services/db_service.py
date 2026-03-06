import os
import logging
from datetime import datetime, timedelta
from typing import Set

logger = logging.getLogger(__name__)

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None
    logger.warning("pymongo is not installed. Chatbot will run without MongoDB support.")

class DbService:
    def __init__(self):
        self.db_name = "RubberIntelligenceDb"
        self.collection_name = "DiseaseRecords"
        self.client = None
        self.db = None
        
        connection_string = os.environ.get("MONGODB_CONNECTION_STRING")
        if connection_string and MongoClient:
            try:
                self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
                # Verify connection
                self.client.admin.command('ping')
                self.db = self.client[self.db_name]
                logger.info(f"Connected to MongoDB: {self.db_name}")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                self.client = None
                self.db = None
        else:
            logger.warning("MONGODB_CONNECTION_STRING not set in .env. Running without DB support.")

    def get_nearby_diseases(self, latitude: float, longitude: float, radius_km: float = 5, days: int = 30) -> Set[str]:
        """
        Queries MongoDB for DiseaseRecords within a certain radius and time frame.
        Returns a set of unique disease 'PredictedLabel' strings.
        """
        if not self.db:
            return set()
            
        try:
            collection = self.db[self.collection_name]
            
            # 5km in radians (Earth radius approx 6378.1 km)
            radius_in_radians = radius_km / 6378.1
            
            since_date = datetime.utcnow() - timedelta(days=days)
            
            query = {
                "Timestamp": {"$gte": since_date},
                # Exclude rejected/unrecognized tests
                "PredictedLabel": {
                    "$nin": ["Rejected", "Unrecognized Domain", "Unidentified Pest", "Unidentified Plant", "Healthy"]
                },
                "Location": {
                    "$geoWithin": {
                        # $centerSphere takes [longitude, latitude], radius in radians
                        "$centerSphere": [[longitude, latitude], radius_in_radians]
                    }
                }
            }
            
            projection = {"PredictedLabel": 1, "_id": 0}
            
            cursor = collection.find(query, projection)
            
            detected_diseases = set()
            for doc in cursor:
                label = doc.get("PredictedLabel")
                if label:
                    detected_diseases.add(label)
                    
            if detected_diseases:
                logger.info(f"Detected nearby active diseases: {detected_diseases}")
                
            return detected_diseases
            
        except Exception as e:
            logger.error(f"Error querying nearby diseases from MongoDB: {e}")
            return set()
