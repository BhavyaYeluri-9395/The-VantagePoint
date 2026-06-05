# database/mongodb_handler.py
import os
from pymongo import MongoClient
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()

class MongoDBHandler:
    def __init__(self):
        self.connection_string = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        self.client = MongoClient(self.connection_string)
        self.db = self.client['vantage_point']
        self.predictions = self.db['predictions']
        self.feedback = self.db['feedback']
        self.user_sessions = self.db['user_sessions']
        self.model_metrics = self.db['model_metrics']
        self._create_indexes()
        print("✅ MongoDB connected successfully")
    
    def _create_indexes(self):
        self.predictions.create_index([("timestamp", -1)])
        self.predictions.create_index([("session_id", 1)])
        self.feedback.create_index([("timestamp", -1)])
        self.feedback.create_index([("session_id", 1)])
        self.user_sessions.create_index([("session_id", 1)], unique=True)
        self.user_sessions.create_index([("last_active", -1)])
        print("✅ Indexes created")
    
    def store_prediction(self, session_id, input_data, predicted_rating, model_version="v1", response_time_ms=None):
        prediction_record = {
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "input_data": input_data,
            "predicted_rating": predicted_rating,
            "model_version": model_version,
            "response_time_ms": response_time_ms,
            "feedback_received": False
        }
        result = self.predictions.insert_one(prediction_record)
        return result.inserted_id
    
    def store_feedback(self, session_id, prediction_id, predicted_rating, actual_rating, user_satisfaction, comments, movie_features):
        absolute_error = abs(predicted_rating - actual_rating)
        squared_error = (predicted_rating - actual_rating) ** 2
        
        feedback_record = {
            "session_id": session_id,
            "prediction_id": prediction_id,
            "timestamp": datetime.utcnow(),
            "predicted_rating": predicted_rating,
            "actual_rating": actual_rating,
            "absolute_error": absolute_error,
            "squared_error": squared_error,
            "user_satisfaction": user_satisfaction,
            "movie_features": movie_features,
            "comments": comments
        }
        result = self.feedback.insert_one(feedback_record)
        
        self.predictions.update_one(
            {"_id": prediction_id},
            {"$set": {"feedback_received": True, "feedback_id": result.inserted_id}}
        )
        return result.inserted_id
    
    def update_user_session(self, session_id, user_data):
        """Store or update user session information - FIXED VERSION"""
        result = self.user_sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "last_active": datetime.utcnow(),
                    "user_agent": user_data.get('user_agent', 'Unknown'),
                    "ip_address": user_data.get('ip_address', 'Unknown')
                },
                "$inc": {"total_predictions": 1},
                "$setOnInsert": {
                    "session_id": session_id,
                    "total_feedback": 0,
                    "preferred_genres": [],
                    "avg_rating_given": 0,
                    "created_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        return result
    
    def get_prediction_history(self, session_id, limit=50):
        cursor = self.predictions.find({"session_id": session_id}).sort("timestamp", -1).limit(limit)
        return list(cursor)
    
    def get_feedback_analytics(self, days=30):
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff_date}}},
            {"$group": {
                "_id": None,
                "total_feedback": {"$sum": 1},
                "avg_absolute_error": {"$avg": "$absolute_error"},
                "avg_satisfaction": {"$avg": "$user_satisfaction"},
                "min_error": {"$min": "$absolute_error"},
                "max_error": {"$max": "$absolute_error"}
            }}
        ]
        result = list(self.feedback.aggregate(pipeline))
        if result:
            return result[0]
        return {"total_feedback": 0, "avg_absolute_error": 0, "avg_satisfaction": 0, "min_error": 0, "max_error": 0}
    
    def get_model_performance_metrics(self):
        feedback_stats = self.get_feedback_analytics(days=365)
        total_predictions = self.predictions.count_documents({})
        return {
            "total_predictions": total_predictions,
            "total_feedback": feedback_stats["total_feedback"],
            "avg_absolute_error": feedback_stats["avg_absolute_error"],
            "avg_user_satisfaction": feedback_stats["avg_satisfaction"]
        }
    
    def close(self):
        self.client.close()
