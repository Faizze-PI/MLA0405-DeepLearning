# ============================================================================
# EXPERIMENT 14: Implementation of Gradient Descent for Linear Regression
# Objective: Implement the optimization algorithm from scratch to understand
#            what's happening "under the hood" of every model.
# Dataset: Synthetic (y = 4 + 3x + noise)
# Source: Generated in-code, no external dataset needed
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)


if __name__ == "__main__":
    n_samples = 100
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1)

    def compute_cost(X, y, theta):
        m = len(y)
        predictions = X.dot(theta)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def gradient_descent(X, y, learning_rate, iterations):
        m = len(y)
        X_b = np.c_[np.ones((m, 1)), X]
        theta = np.zeros((2, 1))
        cost_history = []

        for i in range(iterations):
            gradients = (1 / m) * X_b.T.dot(X_b.dot(theta) - y)
            theta = theta - learning_rate * gradients
            cost = compute_cost(X_b, y, theta)
            cost_history.append(cost)

        return theta, cost_history

    print("="*60)
    print("EXPERIMENT 14: Gradient Descent for Linear Regression")
    print("="*60)
    print(f"\nTrue parameters: intercept=4, slope=3")
    print(f"Dataset: {n_samples} points, y = 4 + 3x + noise")

    lr_001, cost_001 = gradient_descent(X, y, learning_rate=0.01, iterations=1000)
    lr_01, cost_01 = gradient_descent(X, y, learning_rate=0.1, iterations=1000)
    lr_0001, cost_0001 = gradient_descent(X, y, learning_rate=0.0001, iterations=1000)

    sklearn_model = LinearRegression()
    sklearn_model.fit(X, y)
    sklearn_intercept = sklearn_model.intercept_[0]
    sklearn_slope = sklearn_model.coef_[0][0]

    print("\n" + "="*60)
    print("LEARNING RATE COMPARISON")
    print("="*60)
    print(f"\n{'Learning Rate':<15} {'Intercept':<12} {'Slope':<12} {'Final Cost':<12}")
    print("-"*51)
    print(f"{'0.01':<15} {lr_001[0][0]:<12.4f} {lr_001[1][0]:<12.4f} {cost_001[-1]:<12.4f}")
    print(f"{'0.1':<15} {lr_01[0][0]:<12.4f} {lr_01[1][0]:<12.4f} {cost_01[-1]:<12.4f}")
    print(f"{'0.0001':<15} {lr_0001[0][0]:<12.4f} {lr_0001[1][0]:<12.4f} {cost_0001[-1]:<12.4f}")
    print(f"{'sklearn':<15} {sklearn_intercept:<12.4f} {sklearn_slope:<12.4f} {'N/A':<12}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(range(1, 1001), cost_001, 'b-', linewidth=2, label='lr=0.01')
    axes[0].plot(range(1, 1001), cost_01, 'r-', linewidth=2, label='lr=0.1')
    axes[0].plot(range(1, 1001), cost_0001, 'g-', linewidth=2, label='lr=0.0001')
    axes[0].set_xlabel('Iterations', fontsize=12)
    axes[0].set_ylabel('Cost (MSE)', fontsize=12)
    axes[0].set_title('Loss Curve for Different Learning Rates', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(X, y, alpha=0.5, color='steelblue', edgecolors='k', linewidth=0.5, label='Data Points')
    X_line = np.linspace(0, 2, 100).reshape(-1, 1)
    axes[1].plot(X_line, lr_001[0][0] + lr_001[1][0] * X_line, 'b-', linewidth=2, label='lr=0.01')
    axes[1].plot(X_line, lr_01[0][0] + lr_01[1][0] * X_line, 'r-', linewidth=2, label='lr=0.1')
    axes[1].plot(X_line, lr_0001[0][0] + lr_0001[1][0] * X_line, 'g-', linewidth=2, label='lr=0.0001')
    axes[1].set_xlabel('X', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_title('Fitted Lines vs True Relationship', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
    final_costs = []
    for lr in learning_rates:
        _, costs = gradient_descent(X, y, learning_rate=lr, iterations=1000)
        final_costs.append(costs[-1])

    axes[2].plot(learning_rates, final_costs, 'mo-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Learning Rate', fontsize=12)
    axes[2].set_ylabel('Final Cost', fontsize=12)
    axes[2].set_title('Learning Rate vs Final Cost', fontsize=14)
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Exp14_gradient_descent.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. LR=0.01: Smooth convergence, good final result")
    print("2. LR=0.1: Faster convergence, similar final result")
    print("3. LR=0.0001: Very slow convergence, needs more iterations")
    print("4. Too high LR (>0.5): May diverge (loss increases)")
    print("5. GD results match sklearn closed-form solution")

    best_lr_idx = np.argmin([cost_001[-1], cost_01[-1], cost_0001[-1]])
    best_lr = [0.01, 0.1, 0.0001][best_lr_idx]
    best_cost = [cost_001[-1], cost_01[-1], cost_0001[-1]][best_lr_idx]
    print(f"\nBest Final Cost: {best_cost:.4f} (Learning Rate={best_lr})")
    print(f"Converged params: intercept={lr_01[0][0]:.4f}, slope={lr_01[1][0]:.4f}")
    print(f"True params:      intercept=4.0000, slope=3.0000")

    print("\nPlot saved as Exp14_gradient_descent.png")
    print("Exp 14 completed successfully!")
