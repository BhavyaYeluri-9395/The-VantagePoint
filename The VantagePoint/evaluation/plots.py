import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true, y_pred):
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.show()

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution")
    plt.show()

def plot_convergence(history):
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("PSO Convergence")
    plt.show()