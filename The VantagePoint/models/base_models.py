import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def get_models():
    """
    Returns a list of base models for the stacking ensemble.
    Each model's predictions will be clipped to [1, 10] range.
    """
    
    models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        RandomForestRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        SVR(kernel='rbf', C=1.0, epsilon=0.1),
        XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    ]
    
    return models


def clip_predictions(predictions):
    """
    Clips predictions to valid rating range [1, 10].
    """
    return np.clip(predictions, 1.0, 10.0)