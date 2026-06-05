"""from flask import Flask, render_template, request, jsonify, session
import numpy as np
import traceback
import pickle
import os
import uuid
from datetime import datetime

# Import new components
from soft_computing.fuzzy_logic import FuzzyLogicSystem
from optimization.mf_pso_optimizer import MFParticleSwarmOptimizer
from optimization.ga_nn_integrator import GANeuralNetworkIntegrator
from database.user_feedback import UserFeedbackSystem

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize components
fuzzy_system = FuzzyLogicSystem()
feedback_system = UserFeedbackSystem()

# Global variables
model = None
feature_weights = None
ga_optimizer = None
pso_optimizer = None


def load_model():
    global model, feature_weights, ga_optimizer, pso_optimizer
    
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
            
            # Initialize GA and PSO optimizers
            from models.base_models import get_models
            base_models = get_models()
            from ensemble.stacking import StackingEnsemble
            stacking_model = StackingEnsemble(base_models)
            
            ga_optimizer = GANeuralNetworkIntegrator(stacking_model)
            pso_optimizer = MFParticleSwarmOptimizer(fuzzy_system)
            
            print("Model and optimizers loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print("No trained model found. Using fallback prediction system.")
        return False


def fallback_prediction(budget_millions, cast_level, critic_rating, revenue_millions, genre):
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
    return fuzzy_system.compute_fuzzy_score(budget_usd, popularity_score, use_knowledge_base=True)


@app.route("/")
def home():
    # Generate or get session ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        session_id = session.get('session_id', str(uuid.uuid4()))
        
        # Get inputs
        budget_millions = float(data.get("budget_millions", 100))
        revenue_millions = float(data.get("revenue_millions", 250))
        critic_rating = float(data.get("critic_rating", 7.0))
        cast_level = data.get("cast", "moderate")
        genre = data.get("genre", "action")
        
        budget_usd = budget_millions * 1_000_000
        revenue_usd = revenue_millions * 1_000_000
        
        # Map cast to popularity
        cast_map = {"weak": 35, "moderate": 65, "strong": 90}
        cast_popularity = cast_map.get(cast_level, 65)
        
        # Calculate popularity score
        budget_normalized = min(100, budget_millions / 5)
        popularity_score = (cast_popularity * 0.7) + (budget_normalized * 0.3)
        popularity_score = min(100, max(0, popularity_score))
        
        # Compute fuzzy score using knowledge base
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
        
        return jsonify({
            "rating": final_rating,
            "success": True,
            "session_id": session_id
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
        
        # Store feedback
        feedback_system.store_feedback(
            session_id=session_id,
            predicted_rating=float(data['predicted_rating']),
            actual_rating=float(data['actual_rating']),
            user_satisfaction=int(data.get('satisfaction', 3)),
            movie_features=data.get('movie_features', {}),
            user_comments=data.get('comments', '')
        )
        
        # Update user preferences
        feedback_system.update_user_preferences(
            session_id=session_id,
            movie_features=data.get('movie_features', {}),
            rating=float(data['actual_rating'])
        )
        
        return jsonify({"success": True, "message": "Thank you for your feedback!"})
        
    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/optimize_membership", methods=["POST"])
def optimize_membership_functions():
    try:
        from utils.data_loader import load_data
        from config.config import FEATURE_COLUMNS, TARGET_COLUMN
        from soft_computing.fuzzy_logic import FuzzyLogicSystem
        
        # Load data
        train_df, test_df = load_data()
        
        # Prepare data for optimization
        X = train_df[FEATURE_COLUMNS].values
        y = train_df[TARGET_COLUMN].values
        
        # Run PSO optimization
        pso_opt = MFParticleSwarmOptimizer(fuzzy_system)
        best_params, best_fitness = pso_opt.optimize(X, y)
        
        # Save optimized knowledge base
        fuzzy_system.save_knowledge_base()
        
        return jsonify({
            "success": True,
            "best_fitness": best_fitness,
            "message": "Membership functions optimized successfully!"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/optimize_weights", methods=["POST"])
def optimize_nn_weights():
    try:
        from utils.data_loader import load_data
        from config.config import FEATURE_COLUMNS, TARGET_COLUMN
        from models.base_models import get_models
        from ensemble.stacking import StackingEnsemble
        
        # Load data
        train_df, test_df = load_data()
        
        # Prepare data
        X = train_df[FEATURE_COLUMNS].values
        y = train_df[TARGET_COLUMN].values
        
        # Create model
        base_models = get_models()
        stacking_model = StackingEnsemble(base_models)
        
        # Run GA optimization
        ga_opt = GANeuralNetworkIntegrator(stacking_model)
        best_weights, best_fitness = ga_opt.optimize(X, y)
        
        return jsonify({
            "success": True,
            "best_fitness": best_fitness,
            "message": "Neural network weights optimized successfully!"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/feedback_stats", methods=["GET"])
def get_feedback_stats():
    stats = feedback_system.get_feedback_statistics()
    report = feedback_system.get_performance_report()
    return jsonify({
        "statistics": stats,
        "report": report
    })


if __name__ == "__main__":
    print("Starting VANTAGE POINT Server with Complete Soft Computing Framework...")
    load_model()
    print("Visit: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) """
from flask import Flask, render_template, request, jsonify, session
import numpy as np
import traceback
import pickle
import os
import uuid
from datetime import datetime

# Import new components
from soft_computing.fuzzy_logic import FuzzyLogicSystem
from optimization.mf_pso_optimizer import MFParticleSwarmOptimizer
from optimization.ga_nn_integrator import GANeuralNetworkIntegrator
from database.user_feedback import UserFeedbackSystem

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# ============================================================
# CURRENCY CONVERSION CONSTANTS
# ============================================================
USD_TO_INR = 94.95  # 1 USD = 83 Indian Rupees (update as needed)
# 1 Million USD = 83 Million INR = 8.3 Crore INR

# Initialize components
fuzzy_system = FuzzyLogicSystem()
feedback_system = UserFeedbackSystem()

# Global variables
model = None
feature_weights = None
ga_optimizer = None
pso_optimizer = None


def load_model():
    """Load the trained model if it exists"""
    global model, feature_weights, ga_optimizer, pso_optimizer
    
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
            
            # Initialize GA and PSO optimizers
            from models.base_models import get_models
            base_models = get_models()
            from ensemble.stacking import StackingEnsemble
            stacking_model = StackingEnsemble(base_models)
            
            ga_optimizer = GANeuralNetworkIntegrator(stacking_model)
            pso_optimizer = MFParticleSwarmOptimizer(fuzzy_system)
            
            print("Model and optimizers loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print(" No trained model found. Using fallback prediction system.")
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


# ============================================================
# CURRENCY HELPER FUNCTIONS
# ============================================================

def convert_usd_to_inr_millions(usd_millions):
    """Convert USD millions to INR Crores"""
    return (usd_millions * USD_TO_INR) / 10  # Returns value in Crores


def convert_inr_crores_to_usd(inr_crores):
    """Convert INR Crores to USD millions"""
    return (inr_crores * 10) / USD_TO_INR


def format_currency(value_in_usd_millions, currency="USD"):
    """Format value based on selected currency for display"""
    if currency == "USD":
        return f"${value_in_usd_millions}M"
    else:
        inr_crores = convert_usd_to_inr_millions(value_in_usd_millions)
        return f"₹{inr_crores:.1f}Cr"


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def home():
    # Generate or get session ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        session_id = session.get('session_id', str(uuid.uuid4()))
        
        # Get inputs - ALWAYS in USD from frontend (converted before sending)
        budget_millions = float(data.get("budget_millions", 100))
        revenue_millions = float(data.get("revenue_millions", 250))
        critic_rating = float(data.get("critic_rating", 7.0))
        cast_level = data.get("cast", "moderate")
        genre = data.get("genre", "action")
        currency = data.get("currency", "USD")  # User's preferred currency for response
        
        budget_usd = budget_millions * 1_000_000
        revenue_usd = revenue_millions * 1_000_000
        
        # Map cast to popularity
        cast_map = {"weak": 35, "moderate": 65, "strong": 90}
        cast_popularity = cast_map.get(cast_level, 65)
        
        # Calculate popularity score
        budget_normalized = min(100, budget_millions / 5)
        popularity_score = (cast_popularity * 0.7) + (budget_normalized * 0.3)
        popularity_score = min(100, max(0, popularity_score))
        
        # Compute fuzzy score using knowledge base
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
        
        # Prepare response with currency info
        response_data = {
            "rating": final_rating,
            "success": True,
            "session_id": session_id,
            "currency_used": currency,
            "budget_display": format_currency(budget_millions, currency),
            "revenue_display": format_currency(revenue_millions, currency)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Endpoint for user feedback submission"""
    try:
        data = request.json
        session_id = session.get('session_id', data.get('session_id', 'unknown'))
        
        # Store feedback
        feedback_system.store_feedback(
            session_id=session_id,
            predicted_rating=float(data['predicted_rating']),
            actual_rating=float(data['actual_rating']),
            user_satisfaction=int(data.get('satisfaction', 3)),
            movie_features=data.get('movie_features', {}),
            user_comments=data.get('comments', '')
        )
        
        # Update user preferences
        feedback_system.update_user_preferences(
            session_id=session_id,
            movie_features=data.get('movie_features', {}),
            rating=float(data['actual_rating'])
        )
        
        return jsonify({"success": True, "message": "Thank you for your feedback!"})
        
    except Exception as e:
        print(f"Feedback error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/optimize_membership", methods=["POST"])
def optimize_membership_functions():
    """Endpoint to trigger PSO optimization of membership functions"""
    try:
        from utils.data_loader import load_data
        from config.config import FEATURE_COLUMNS, TARGET_COLUMN
        from soft_computing.fuzzy_logic import FuzzyLogicSystem
        
        # Load data
        train_df, test_df = load_data()
        
        # Prepare data for optimization
        X = train_df[FEATURE_COLUMNS].values
        y = train_df[TARGET_COLUMN].values
        
        # Run PSO optimization
        pso_opt = MFParticleSwarmOptimizer(fuzzy_system)
        best_params, best_fitness = pso_opt.optimize(X, y)
        
        # Save optimized knowledge base
        fuzzy_system.save_knowledge_base()
        
        return jsonify({
            "success": True,
            "best_fitness": best_fitness,
            "message": "Membership functions optimized successfully!"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/optimize_weights", methods=["POST"])
def optimize_nn_weights():
    """Endpoint to trigger GA optimization of neural network weights"""
    try:
        from utils.data_loader import load_data
        from config.config import FEATURE_COLUMNS, TARGET_COLUMN
        from models.base_models import get_models
        from ensemble.stacking import StackingEnsemble
        
        # Load data
        train_df, test_df = load_data()
        
        # Prepare data
        X = train_df[FEATURE_COLUMNS].values
        y = train_df[TARGET_COLUMN].values
        
        # Create model
        base_models = get_models()
        stacking_model = StackingEnsemble(base_models)
        
        # Run GA optimization
        ga_opt = GANeuralNetworkIntegrator(stacking_model)
        best_weights, best_fitness = ga_opt.optimize(X, y)
        
        return jsonify({
            "success": True,
            "best_fitness": best_fitness,
            "message": "Neural network weights optimized successfully!"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/feedback_stats", methods=["GET"])
def get_feedback_stats():
    """Get feedback statistics"""
    stats = feedback_system.get_feedback_statistics()
    report = feedback_system.get_performance_report()
    return jsonify({
        "statistics": stats,
        "report": report
    })


@app.route("/convert_currency", methods=["POST"])
def convert_currency():
    """API endpoint to convert between USD and INR"""
    try:
        data = request.json
        amount = float(data.get("amount", 0))
        from_currency = data.get("from_currency", "USD")
        to_currency = data.get("to_currency", "INR")
        
        if from_currency == "USD" and to_currency == "INR":
            converted = convert_usd_to_inr_millions(amount)
            return jsonify({
                "success": True,
                "original": f"${amount}M",
                "converted": f"₹{converted:.1f}Cr",
                "value": converted
            })
        elif from_currency == "INR" and to_currency == "USD":
            converted = convert_inr_crores_to_usd(amount)
            return jsonify({
                "success": True,
                "original": f"₹{amount}Cr",
                "converted": f"${converted:.1f}M",
                "value": converted
            })
        else:
            return jsonify({"success": False, "error": "Invalid currency conversion"}), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    print("Starting VANTAGE POINT Server")
    print(f"Currency Conversion Rate: 1 USD = ₹{USD_TO_INR}")
    print("Visit: http://localhost:5000")
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)