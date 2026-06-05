import sqlite3
import json
import numpy as np
from datetime import datetime
import os

class UserFeedbackSystem:
    """
    Complete User Feedback System with database storage and learning
    """
    
    def __init__(self, db_path="database/feedback.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing user feedback"""
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"Created directory: {db_dir}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                predicted_rating REAL,
                actual_rating REAL,
                rating_difference REAL,
                user_satisfaction INTEGER,
                movie_features TEXT,
                user_comments TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT,
                mae REAL,
                rmse REAL,
                avg_user_satisfaction REAL,
                total_feedback INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                preferred_genres TEXT,
                avg_rating_preference REAL,
                cast_preference TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("User Feedback Database initialized")
    
    def store_feedback(self, session_id, predicted_rating, actual_rating, 
                       user_satisfaction, movie_features, user_comments=""):
        """Store user feedback in database"""
        rating_difference = abs(predicted_rating - actual_rating)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback 
            (session_id, predicted_rating, actual_rating, rating_difference, 
             user_satisfaction, movie_features, user_comments)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, predicted_rating, actual_rating, rating_difference,
              user_satisfaction, json.dumps(movie_features), user_comments))
        
        conn.commit()
        conn.close()
        
        print(f"Feedback stored: Pred={predicted_rating}, Actual={actual_rating}, Diff={rating_difference}")
        
        # Trigger model retraining if enough feedback collected
        self.check_and_retrain()
    
    def get_feedback_statistics(self):
        """Get statistics from collected feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_feedback,
                AVG(rating_difference) as avg_error,
                AVG(user_satisfaction) as avg_satisfaction,
                MIN(rating_difference) as min_error,
                MAX(rating_difference) as max_error
            FROM feedback
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            "total_feedback": stats[0] or 0,
            "avg_error": round(stats[1], 4) if stats[1] else 0,
            "avg_satisfaction": round(stats[2], 2) if stats[2] else 0,
            "min_error": round(stats[3], 4) if stats[3] else 0,
            "max_error": round(stats[4], 4) if stats[4] else 0
        }
    
    def calculate_model_metrics(self):
        """Calculate current model performance metrics from feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                AVG(ABS(rating_difference)) as mae,
                AVG(rating_difference * rating_difference) as mse,
                AVG(user_satisfaction) as satisfaction
            FROM feedback
            WHERE timestamp > datetime('now', '-30 days')
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result[0]:
            return {
                "mae": round(result[0], 4),
                "rmse": round(np.sqrt(result[1]), 4) if result[1] else 0,
                "avg_satisfaction": round(result[2], 2) if result[2] else 0
            }
        return None
    
    def check_and_retrain(self):
        """Check if enough feedback collected to trigger retraining"""
        stats = self.get_feedback_statistics()
        
        # Retrain if we have more than 100 feedback entries
        if stats["total_feedback"] >= 100 and stats["total_feedback"] % 50 == 0:
            print(f"Collected {stats['total_feedback']} feedback entries. Consider retraining model!")
            return True
        return False
    
    def get_user_preferences(self, session_id):
        """Get user preferences for personalization"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT preferred_genres, avg_rating_preference, cast_preference
            FROM user_preferences
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "preferred_genres": json.loads(result[0]) if result[0] else [],
                "avg_rating_preference": result[1] or 5.0,
                "cast_preference": result[2] or "moderate"
            }
        return None
    
    def update_user_preferences(self, session_id, movie_features, rating):
        """Update user preferences based on their feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current preferences
        current = self.get_user_preferences(session_id)
        
        if current:
            # Update existing preferences with moving average
            genres = current["preferred_genres"]
            if movie_features.get("genre") not in genres:
                genres.append(movie_features.get("genre"))
                if len(genres) > 5:  # Keep only top 5
                    genres = genres[-5:]
            
            new_avg_rating = (current["avg_rating_preference"] + rating) / 2
            
            cursor.execute('''
                INSERT INTO user_preferences 
                (session_id, preferred_genres, avg_rating_preference, cast_preference)
                VALUES (?, ?, ?, ?)
            ''', (session_id, json.dumps(genres), new_avg_rating, current["cast_preference"]))
        else:
            # Create new preferences
            cursor.execute('''
                INSERT INTO user_preferences 
                (session_id, preferred_genres, avg_rating_preference, cast_preference)
                VALUES (?, ?, ?, ?)
            ''', (session_id, json.dumps([movie_features.get("genre")]), rating, "moderate"))
        
        conn.commit()
        conn.close()
    
    def get_performance_report(self):
        """Generate comprehensive performance report"""
        stats = self.get_feedback_statistics()
        metrics = self.calculate_model_metrics()
        
        report = {
            "feedback_statistics": stats,
            "recent_performance": metrics,
            "recommendation": ""
        }
        
        if metrics and metrics["mae"] > 1.0:
            report["recommendation"] = "Model needs retraining - high prediction error"
        elif metrics and metrics["avg_satisfaction"] < 3.0:
            report["recommendation"] = "User satisfaction low - consider adjusting model"
        else:
            report["recommendation"] = "Model performing well"
        
        return report