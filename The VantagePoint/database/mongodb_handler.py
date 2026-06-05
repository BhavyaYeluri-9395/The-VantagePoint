# database/mongodb_handler.py
import os
from pymongo import MongoClient
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MongoDBHandler:
    def __init__(self):
        """Initialize MongoDB connection"""
        # For local MongoDB
        # self.client = MongoClient('mongodb://localhost:27017/')
        
        # For MongoDB Atlas (cloud)
        self.connection_string = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        self.client = MongoClient(self.connection_string)
        
        # Database and Collections
        self.db = self.client['vantage_point']
        self.predictions = self.db['predictions']
        self.feedback = self.db['feedback']
        self.user_sessions = self.db['user_sessions']
        self.model_metrics = self.db['model_metrics']
        
        # Create indexes for better query performance
        self._create_indexes()
        
        print("MongoDB connected successfully")
    
    def _create_indexes(self):
        """Create indexes for faster queries"""
        self.predictions.create_index([("timestamp", -1)])
        self.predictions.create_index([("session_id", 1)])
        self.predictions.create_index([("predicted_rating", 1)])
        
        self.feedback.create_index([("timestamp", -1)])
        self.feedback.create_index([("session_id", 1)])
        self.feedback.create_index([("user_satisfaction", 1)])
        
        self.user_sessions.create_index([("session_id", 1)], unique=True)
        self.user_sessions.create_index([("last_active", -1)])
        
        self.model_metrics.create_index([("timestamp", -1)])
        
        print("Indexes created")
    
    def store_prediction(self, session_id, input_data, predicted_rating, 
                         model_version="v1", response_time_ms=None):
        """Store each prediction made by the system"""
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
    
    def store_feedback(self, session_id, prediction_id, predicted_rating, 
                       actual_rating, user_satisfaction, comments, movie_features):
        """Store user feedback"""
        # Calculate error metrics
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
        
        # Update the prediction record to mark feedback received
        self.predictions.update_one(
            {"_id": prediction_id},
            {"$set": {"feedback_received": True, "feedback_id": result.inserted_id}}
        )
        
        return result.inserted_id
    
    def update_user_session(self, session_id, user_data):
        """Store or update user session information"""
        session_record = {
            "session_id": session_id,
            "last_active": datetime.utcnow(),
            "total_predictions": 1,
            "total_feedback": 0,
            "preferred_genres": [],
            "avg_rating_given": 0,
            **user_data
        }
        
        result = self.user_sessions.update_one(
            {"session_id": session_id},
            {"$set": {"last_active": datetime.utcnow()},
             "$inc": {"total_predictions": 1},
             "$setOnInsert": session_record},
            upsert=True
        )
        return result
    
    def get_prediction_history(self, session_id, limit=50):
        """Get prediction history for a user"""
        cursor = self.predictions.find(
            {"session_id": session_id}
        ).sort("timestamp", -1).limit(limit)
        
        return list(cursor)
    
    def get_feedback_analytics(self, days=30):
        """Get feedback analytics for the last N days"""
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
                "max_error": {"$max": "$absolute_error"},
                "satisfaction_distribution": {
                    "$push": "$user_satisfaction"
                }
            }}
        ]
        
        result = list(self.feedback.aggregate(pipeline))
        
        if result:
            return result[0]
        return {
            "total_feedback": 0,
            "avg_absolute_error": 0,
            "avg_satisfaction": 0,
            "min_error": 0,
            "max_error": 0
        }
    
    def get_model_performance_metrics(self):
        """Get comprehensive model performance metrics"""
        # Get overall feedback stats
        feedback_stats = self.get_feedback_analytics(days=365)
        
        # Get predictions count
        total_predictions = self.predictions.count_documents({})
        
        # Get satisfaction distribution
        satisfaction_pipeline = [
            {"$group": {
                "_id": "$user_satisfaction",
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id": 1}}
        ]
        satisfaction_dist = list(self.feedback.aggregate(satisfaction_pipeline))
        
        return {
            "total_predictions": total_predictions,
            "total_feedback": feedback_stats["total_feedback"],
            "avg_absolute_error": feedback_stats["avg_absolute_error"],
            "avg_user_satisfaction": feedback_stats["avg_satisfaction"],
            "satisfaction_distribution": satisfaction_dist,
            "error_range": {
                "min": feedback_stats["min_error"],
                "max": feedback_stats["max_error"]
            }
        }
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()