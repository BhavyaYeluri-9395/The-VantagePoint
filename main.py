from utils.data_loader import load_data
from config.config import FEATURE_COLUMNS, TARGET_COLUMN
from soft_computing.fuzzy_logic import FuzzyLogicSystem
from models.base_models import get_models
from ensemble.stacking import StackingEnsemble
from optimization.pso_optimizer import PSOOptimizer
from evaluation.metrics import evaluate
from evaluation.plots import plot_predictions, plot_residuals, plot_convergence

import numpy as np
import joblib


def apply_fuzzy_logic(train_df, test_df):
    """
    Applies fuzzy logic transformation to both train and test datasets.
    """
    fuzzy = FuzzyLogicSystem()

    for df in [train_df, test_df]:
        df["fuzzy_score"] = df.apply(
            lambda row: fuzzy.compute_fuzzy_score(
                row["budget_usd"],
                row["popularity_score"]
            ),
            axis=1
        )

    return train_df, test_df


def main():

    print("\n🔹 Loading Dataset...")
    train_df, test_df = load_data()

    # ---------------------------
    # Layer 1 – Fuzzy Logic
    # ---------------------------
    print("🔹 Applying Fuzzy Logic Layer...")
    train_df, test_df = apply_fuzzy_logic(train_df, test_df)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMN].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values

    # ---------------------------
    # Layer 2 – PSO Optimization
    # ---------------------------
    print("🔹 Initializing Base Models...")
    base_models = get_models()

    stacking_model = StackingEnsemble(base_models)

    print("🔹 Running PSO for Feature Weight Optimization...")
    pso = PSOOptimizer()

    best_feature_weights = pso.optimize(
        X=X_train,
        y=y_train,
        model=stacking_model
    )

    print("✅ Optimized Feature Weights:")
    print(best_feature_weights)

    # Apply optimized weights
    X_train_weighted = X_train * best_feature_weights
    X_test_weighted = X_test * best_feature_weights

    # ---------------------------
    # Layer 3 – Stacking Ensemble
    # ---------------------------
    print("🔹 Training Final Stacking Model...")
    stacking_model.fit(X_train_weighted, y_train)

    predictions = stacking_model.predict(X_test_weighted)

    # ---------------------------
    # Layer 4 – Evaluation
    # ---------------------------
    print("\n📊 Evaluating Model...")
    results = evaluate(y_test, predictions)

    print("\nFinal Evaluation Results:")
    print(results)

    # ---------------------------
    # Layer 4 – Visualization
    # ---------------------------
    plot_predictions(y_test, predictions)
    plot_residuals(y_test, predictions)
    plot_convergence(pso.history)

    # ---------------------------
    # Layer 5 – Save Model
    # ---------------------------
    print("\n💾 Saving Model...")
    joblib.dump(
        {
            "model": stacking_model,
            "feature_weights": best_feature_weights
        },
        "stacking_model.pkl"
    )

    print("✅ Model saved successfully.")
    print("\n🎯 System Execution Completed.")


if __name__ == "__main__":
    main()