from sklearn.linear_model import LinearRegression
import numpy as np

class StackingEnsemble:

    def __init__(self, base_models):
        self.base_models = base_models
        self.meta_model = LinearRegression()

    def fit(self, X, y):
        """
        Fit all base models and the meta model.
        """
        self.fitted_models = []
        
        # Clip target values to [1, 10] range
        y_clipped = np.clip(y, 1.0, 10.0)

        for model in self.base_models:
            model.fit(X, y_clipped)
            self.fitted_models.append(model)
        
        # Generate meta-features from base model predictions
        meta_features = self._get_meta_features(X)
        
        # Train meta model on clipped predictions
        self.meta_model.fit(meta_features, y_clipped)

    def _get_meta_features(self, X):
        """
        Generate meta-features by getting predictions from all base models.
        """
        meta_features = []
        for model in self.fitted_models:
            preds = model.predict(X)
            preds = np.clip(preds, 1.0, 10.0)  # Clip each model's predictions
            meta_features.append(preds)
        return np.column_stack(meta_features)

    def predict(self, X):
        """
        Predict using the trained stacking ensemble.
        Returns predictions clipped to [1, 10] range.
        """
        meta_features = self._get_meta_features(X)
        predictions = self.meta_model.predict(meta_features)
        
        # Final clipping to valid rating range
        return np.clip(predictions, 1.0, 10.0)