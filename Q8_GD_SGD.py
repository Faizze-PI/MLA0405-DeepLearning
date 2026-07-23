# ============================================================================
# QUESTION 8: Gradient Descent and Stochastic Gradient Descent (SGD)
# a) Implement Gradient Descent to minimize a cost function and analyze the
#    effect of learning rate on convergence speed.
# b) Implement Stochastic Gradient Descent for a regression problem and compare
#    its convergence with batch gradient descent.
# ============================================================================
# Dataset: Synthetic Regression Data (Generated using numpy)
# Alternative: Energy Consumption Dataset
# Link: https://www.kaggle.com/datasets/govindaramsriram/energy-consumption-dataset-linear-regression
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

n_samples = 500
X = 2 * np.random.rand(n_samples, 1)
y = 4 + 3 * X + np.random.randn(n_samples, 1)

X_b = np.c_[np.ones((n_samples, 1)), X]

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def batch_gradient_descent(X, y, learning_rate, iterations):
    m = len(y)
    theta = np.random.randn(X.shape[1], 1)
    cost_history = []

    for i in range(iterations):
        gradients = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

def stochastic_gradient_descent(X, y, learning_rate, iterations):
    m = len(y)
    theta = np.random.randn(X.shape[1], 1)
    cost_history = []

    for i in range(iterations):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = xi.T.dot(xi.dot(theta) - yi)
        theta = theta - learning_rate * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

def mini_batch_gradient_descent(X, y, learning_rate, iterations, batch_size=20):
    m = len(y)
    theta = np.random.randn(X.shape[1], 1)
    cost_history = []

    for i in range(iterations):
        random_indices = np.random.choice(m, batch_size, replace=False)
        xi = X[random_indices]
        yi = y[random_indices]
        gradients = (1 / batch_size) * xi.T.dot(xi.dot(theta) - yi)
        theta = theta - learning_rate * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

print("="*70)
print("PART A: Gradient Descent - Learning Rate Analysis")
print("="*70)

learning_rates = [0.01, 0.03, 0.1, 0.3]
iterations = 100

plt.figure(figsize=(10, 6))
for lr in learning_rates:
    theta, cost_history = batch_gradient_descent(X_b, y, lr, iterations)
    plt.plot(range(1, iterations + 1), cost_history, linewidth=2, label=f'lr={lr}')
    print(f"\nLearning Rate = {lr}:")
    print(f"  Final Cost: {cost_history[-1]:.4f}")
    print(f"  Theta: {theta.flatten()}")

plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Cost (MSE)', fontsize=12)
plt.title('Effect of Learning Rate on Convergence', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q8_learning_rate_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("PART B: SGD vs Batch Gradient Descent")
print("="*70)

iterations_sgd = 50
learning_rate_sgd = 0.01

np.random.seed(42)
theta_bgd, cost_bgd = batch_gradient_descent(X_b, y, learning_rate_sgd, iterations_sgd)

np.random.seed(42)
theta_sgd, cost_sgd = stochastic_gradient_descent(X_b, y, learning_rate_sgd, iterations_sgd)

np.random.seed(42)
theta_mbgd, cost_mbgd = mini_batch_gradient_descent(X_b, y, learning_rate_sgd, iterations_sgd, batch_size=20)

print(f"\nFinal Parameters:")
print(f"  Batch GD: {theta_bgd.flatten()}")
print(f"  SGD: {theta_sgd.flatten()}")
print(f"  Mini-Batch GD: {theta_mbgd.flatten()}")

print(f"\nFinal Costs:")
print(f"  Batch GD: {cost_bgd[-1]:.4f}")
print(f"  SGD: {cost_sgd[-1]:.4f}")
print(f"  Mini-Batch GD: {cost_mbgd[-1]:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, iterations_sgd + 1), cost_bgd, 'b-', linewidth=2, label='Batch GD')
plt.plot(range(1, iterations_sgd + 1), cost_sgd, 'r-', linewidth=2, alpha=0.7, label='SGD')
plt.plot(range(1, iterations_sgd + 1), cost_mbgd, 'g-', linewidth=2, alpha=0.7, label='Mini-Batch GD')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Cost (MSE)', fontsize=12)
plt.title('SGD vs Batch Gradient Descent Convergence', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q8_sgd_vs_batch.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X, y, alpha=0.5, color='steelblue', edgecolors='k', linewidth=0.5, label='Data Points')
X_line = np.linspace(0, 2, 100).reshape(-1, 1)
X_line_b = np.c_[np.ones((100, 1)), X_line]

y_bgd = X_line_b.dot(theta_bgd)
y_sgd = X_line_b.dot(theta_sgd)
y_mbgd = X_line_b.dot(theta_mbgd)

axes[0].plot(X_line, y_bgd, 'b-', linewidth=2, label='Batch GD')
axes[0].plot(X_line, y_sgd, 'r--', linewidth=2, label='SGD')
axes[0].plot(X_line, y_mbgd, 'g:', linewidth=2, label='Mini-Batch GD')
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('y', fontsize=12)
axes[0].set_title('Fitted Lines by Different GD Methods', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, iterations_sgd + 1), cost_bgd, 'b-', linewidth=2, label='Batch GD')
axes[1].plot(range(1, iterations_sgd + 1), cost_sgd, 'r-', linewidth=2, alpha=0.7, label='SGD')
axes[1].plot(range(1, iterations_sgd + 1), cost_mbgd, 'g-', linewidth=2, alpha=0.7, label='Mini-Batch GD')
axes[1].set_xlabel('Iterations', fontsize=12)
axes[1].set_ylabel('Cost (MSE)', fontsize=12)
axes[1].set_title('Cost Convergence Comparison', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Q8_gd_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("Analysis Summary")
print("="*70)
print("\n1. Learning Rate Effect:")
print("   - Too small (0.01): Slow convergence")
print("   - Optimal (0.03-0.1): Good balance of speed and stability")
print("   - Too large (0.3): May overshoot or diverge")

print("\n2. GD Methods Comparison:")
print("   - Batch GD: Smooth convergence, uses all data per iteration")
print("   - SGD: Noisy convergence, faster per iteration, uses 1 sample")
print("   - Mini-Batch GD: Balance between Batch GD and SGD")

print("\nQ8 completed successfully!")
print("Plots saved as Q8_learning_rate_analysis.png, Q8_sgd_vs_batch.png, Q8_gd_comparison.png")
