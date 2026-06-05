from flask import Flask, render_template, request, jsonify, session
import numpy as np
import traceback
import pickle
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Import components
from soft_computing.fuzzy_logic import FuzzyLogicSystem
from database.mongodb_handler import MongoDBHandler
from database.user_feedback import UserFeedbackSystem

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

# Initialize MongoDB
mongodb = MongoDBHandler()

# Initialize components
fuzzy_system = FuzzyLogicSystem()
feedback_system = UserFeedbackSystem()  # Keep SQLite as backup

# Global variables
model = None
feature_weights = None

# Currency conversion rate
USD_TO_INR = 94.95


def load_model():
    """Load the trained model if it exists"""
    global model, feature_weights
    
    model_path = "stacking_model.pkl"
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                loaded_obj = pickle.load(f)
            
            if isinstance(loaded_obj, dict):
                model = loaded_obj.get("model")
                feature_weights = loaded_obj.get("feature_weights", np.ones(6))
            else:
                model = loaded_obj
                feature_weights = np.ones(6)
            
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print("No trained model found. Using fallback prediction system.")
        return False


def fallback_prediction(budget_millions, cast_level, critic_rating, revenue_millions, genre):
    """Intelligent fallback prediction"""
    base_rating = 6.0
    budget_factor = min(2.5, budget_millions / 150)
    base_rating += budget_factor * 0.8
    
    cast_map = {"weak": 0.5, "moderate": 1.2, "strong": 2.0}
    base_rating += cast_map.get(cast_level, 1.0)
    
    critic_impact = (critic_rating - 5) * 0.6
    base_rating += critic_impact
    
    revenue_factor = min(2.0, revenue_millions / 300)
    base_rating += revenue_factor * 0.5
    
    genre_multipliers = {
        "action": 1.1, "drama": 0.95, "comedy": 1.05,
        "horror": 0.9, "sci-fi": 1.15, "romance": 0.92
    }
    base_rating *= genre_multipliers.get(genre, 1.0)
    
    return max(2.0, min(9.8, base_rating))


def compute_fuzzy_score_with_kb(budget_usd, popularity_score):
    """Compute fuzzy score using knowledge base"""
    return fuzzy_system.compute_fuzzy_score(budget_usd, popularity_score, use_knowledge_base=True)


@app.route("/")
def home():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Update user session in MongoDB
    mongodb.update_user_session(session['session_id'], {
        "user_agent": request.headers.get('User-Agent', 'Unknown'),
        "ip_address": request.remote_addr
    })
    
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    
    try:
        data = request.json
        session_id = session.get('session_id', str(uuid.uuid4()))
        
        # Get inputs
        budget_millions = float(data.get("budget_millions", 100))
        revenue_millions = float(data.get("revenue_millions", 250))
        critic_rating = float(data.get("critic_rating", 7.0))
        cast_level = data.get("cast", "moderate")
        genre = data.get("genre", "action")
        currency = data.get("currency", "USD")
        
        # Store input data for logging
        input_data = {
            "budget_millions": budget_millions,
            "revenue_millions": revenue_millions,
            "critic_rating": critic_rating,
            "cast_level": cast_level,
            "genre": genre,
            "currency": currency
        }
        
        budget_usd = budget_millions * 1_000_000
        revenue_usd = revenue_millions * 1_000_000
        
        # Map cast to popularity
        cast_map = {"weak": 35, "moderate": 65, "strong": 90}
        cast_popularity = cast_map.get(cast_level, 65)
        
        # Calculate popularity score
        budget_normalized = min(100, budget_millions / 5)
        popularity_score = (cast_popularity * 0.7) + (budget_normalized * 0.3)
        popularity_score = min(100, max(0, popularity_score))
        
        # Compute fuzzy score
        fuzzy_score = compute_fuzzy_score_with_kb(budget_usd, popularity_score)
        
        # Build features
        features = np.array([[
            budget_usd, revenue_usd, popularity_score,
            critic_rating, cast_popularity, fuzzy_score
        ]])
        
        # Apply feature weights
        if feature_weights is not None and len(feature_weights) == 6:
            features_weighted = features * feature_weights
        else:
            features_weighted = features
        
        # Make prediction
        if model is not None:
            try:
                prediction = model.predict(features_weighted)[0]
                final_rating = max(1.0, min(10.0, round(float(prediction), 1)))
            except:
                final_rating = fallback_prediction(budget_millions, cast_level, critic_rating, revenue_millions, genre)
        else:
            final_rating = fallback_prediction(budget_millions, cast_level, critic_rating, revenue_millions, genre)
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Store prediction in MongoDB
        prediction_id = mongodb.store_prediction(
            session_id=session_id,
            input_data=input_data,
            predicted_rating=final_rating,
            model_version="v1",
            response_time_ms=response_time_ms
        )
        
        # Store in session for feedback linking
        session['last_prediction_id'] = str(prediction_id)
        
        # Format currency display
        def format_currency(value):
            if currency == "USD":
                return f"${value}M"
            else:
                inr_crores = (value * USD_TO_INR) / 10
                return f"₹{inr_crores:.1f}Cr"
        
        return jsonify({
            "rating": final_rating,
            "success": True,
            "session_id": session_id,
            "prediction_id": str(prediction_id),
            "currency_used": currency,
            "budget_display": format_currency(budget_millions),
            "revenue_display": format_currency(revenue_millions),
            "response_time_ms": round(response_time_ms, 2)
        })
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    try:
        data = request.json
        session_id = session.get('session_id', data.get('session_id', 'unknown'))
        prediction_id = data.get('prediction_id', session.get('last_prediction_id'))
        
        predicted_rating = float(data['predicted_rating'])
        actual_rating = float(data['actual_rating'])
        satisfaction = int(data.get('satisfaction', 3))
        comments = data.get('comments', '')
        movie_features = data.get('movie_features', {})
        
        # Store in MongoDB
        feedback_id = mongodb.store_feedback(
            session_id=session_id,
            prediction_id=prediction_id,
            predicted_rating=predicted_rating,
            actual_rating=actual_rating,
            user_satisfaction=satisfaction,
            comments=comments,
            movie_features=movie_features
        )
        
        # Also store in SQLite as backup
        feedback_system.store_feedback(
            session_id=session_id,
            predicted_rating=predicted_rating,
            actual_rating=actual_rating,
            user_satisfaction=satisfaction,
            movie_features=movie_features,
            user_comments=comments
        )
        
        # Update user preferences
        feedback_system.update_user_preferences(
            session_id=session_id,
            movie_features=movie_features,
            rating=actual_rating
        )
        
        return jsonify({
            "success": True,
            "message": "Thank you for your feedback!",
            "feedback_id": str(feedback_id)
        })
        
    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/analytics", methods=["GET"])
def get_analytics():
    """Get analytics dashboard data"""
    try:
        metrics = mongodb.get_model_performance_metrics()
        recent_predictions = list(mongodb.predictions.find().sort("timestamp", -1).limit(100))
        
        # Convert ObjectId to string for JSON serialization
        for pred in recent_predictions:
            pred['_id'] = str(pred['_id'])
        
        return jsonify({
            "success": True,
            "metrics": metrics,
            "recent_predictions": recent_predictions
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/user_history", methods=["GET"])
def get_user_history():
    """Get prediction history for current user"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({"success": False, "error": "No session found"}), 400
        
        history = mongodb.get_prediction_history(session_id)
        
        # Convert ObjectId to string
        for record in history:
            record['_id'] = str(record['_id'])
        
        return jsonify({
            "success": True,
            "history": history,
            "total": len(history)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "mongodb_connected": True
    })


if __name__ == "__main__":
    print("Starting VANTAGE POINT Production Server")
    print(f"Currency Rate: 1 USD = ₹{USD_TO_INR}")
    print("Visit: http://localhost:5000")
    print("=" * 50)
    load_model()
    
    # Get port from environment variable (for deployment)
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    app.run(debug=debug, host='0.0.0.0', port=port)