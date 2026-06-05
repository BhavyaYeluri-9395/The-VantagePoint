# database/mongodb_handler.py
import os
from pymongo import MongoClient, errors
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class MongoDBHandler:
    def __init__(self):
        try:
            self.connection_string = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client['vantage_point']
            self.predictions = self.db['predictions']
            self.feedback = self.db['feedback']
            self.user_sessions = self.db['user_sessions']
            self.model_metrics = self.db['model_metrics']
            print(" MongoDB connected successfully")
        except Exception as e:
            print(f" MongoDB connection error: {e}")
            self.client = None
    
    def store_prediction(self, session_id, input_data, predicted_rating, model_version="v1", response_time_ms=None):
        if not self.client:
            print(" No MongoDB connection, skipping save")
            return None
            
        try:
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
            print(f" Prediction saved with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f" Error saving prediction: {e}")
            return None
    
    def store_feedback(self, session_id, prediction_id, predicted_rating, actual_rating, user_satisfaction, comments, movie_features):
        if not self.client:
            print(" No MongoDB connection, skipping feedback save")
            return None
            
        try:
            absolute_error = abs(predicted_rating - actual_rating)
            feedback_record = {
                "session_id": session_id,
                "prediction_id": prediction_id,
                "timestamp": datetime.utcnow(),
                "predicted_rating": predicted_rating,
                "actual_rating": actual_rating,
                "absolute_error": absolute_error,
                "user_satisfaction": user_satisfaction,
                "movie_features": movie_features,
                "comments": comments
            }
            result = self.feedback.insert_one(feedback_record)
            print(f" Feedback saved with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f" Error saving feedback: {e}")
            return None
    
    def update_user_session(self, session_id, user_data):
        if not self.client:
            return None
            
        try:
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
        except Exception as e:
            print(f" Error updating session: {e}")
            return None
    
    def get_prediction_history(self, session_id, limit=50):
        if not self.client:
            return []
        try:
            cursor = self.predictions.find({"session_id": session_id}).sort("timestamp", -1).limit(limit)
            return list(cursor)
        except Exception as e:
            print(f" Error getting history: {e}")
            return []
    
    def close(self):
        if self.client:
            self.client.close()
