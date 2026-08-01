# ============================================================================
# QUESTION 1: Learning Algorithms
# Implement a simple learning algorithm (Linear Regression) using Gradient Descent.
# Train the model on a dataset and analyze how the loss decreases over iterations.
# Plot the learning curve.
# ============================================================================
# Dataset: Housing Prices Dataset
# Link: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = pd.read_csv('Housing.csv')

data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})
data['furnishingstatus'] = data['furnishingstatus'].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})

X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
          'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
          'furnishingstatus']].values
y = data['price'].values

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

y_mean = np.mean(y)
y_normalized = (y - y_mean) / np.std(y)

X_normalized = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

np.random.seed(42)
theta_initial = np.random.randn(X_normalized.shape[1])

learning_rate = 0.01
iterations = 1000

theta_final, cost_history = gradient_descent(X_normalized, y_normalized, theta_initial, learning_rate, iterations)

print("Final Parameters (theta):")
print(theta_final)
print(f"\nFinal Cost: {cost_history[-1]:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, iterations + 1), cost_history, 'b-', linewidth=2)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Cost (MSE)', fontsize=12)
plt.title('Learning Curve - Gradient Descent Linear Regression', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q1_learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n--- Model Performance ---")
predictions = X_normalized.dot(theta_final)
predictions_actual = predictions * np.std(y) + y_mean
mse = np.mean((predictions_actual - y) ** 2)
rmse = np.sqrt(mse)
ss_res = np.sum((y - predictions_actual) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y, predictions_actual, alpha=0.5, color='blue', edgecolors='k', linewidth=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.title('Actual vs Predicted Housing Prices', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q1_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nQ1 completed successfully!")
