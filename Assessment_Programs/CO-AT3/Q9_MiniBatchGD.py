# ============================================================================
# QUESTION 9: Mini-Batch Gradient Descent
# Implement Mini-Batch Gradient Descent and analyze how batch size affects
# training stability and performance.
# ============================================================================
# Dataset: Synthetic Regression Data (Generated using numpy)
# Alternative: Energy Consumption Dataset
# Link: https://www.kaggle.com/datasets/govindaramsriram/energy-consumption-dataset-linear-regression
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def mini_batch_gradient_descent(X, y, learning_rate, iterations, batch_size):
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

print("="*70)
print("Mini-Batch Gradient Descent - Batch Size Analysis")
print("="*70)

batch_sizes = [8, 16, 32, 64, 128]
iterations = 100
learning_rate = 0.01

results = {}

plt.figure(figsize=(10, 6))
for batch_size in batch_sizes:
    np.random.seed(42)
    theta, cost_history = mini_batch_gradient_descent(X_b, y, learning_rate, iterations, batch_size)
    results[batch_size] = {'theta': theta, 'cost_history': cost_history}
    plt.plot(range(1, iterations + 1), cost_history, linewidth=2, label=f'Batch Size = {batch_size}')
    print(f"\nBatch Size = {batch_size}:")
    print(f"  Final Cost: {cost_history[-1]:.4f}")
    print(f"  Theta: {theta.flatten()}")

np.random.seed(42)
theta_bgd, cost_bgd = batch_gradient_descent(X_b, y, learning_rate, iterations)
plt.plot(range(1, iterations + 1), cost_bgd, 'k--', linewidth=2, label='Full Batch GD')
print(f"\nFull Batch GD:")
print(f"  Final Cost: {cost_bgd[-1]:.4f}")
print(f"  Theta: {theta_bgd.flatten()}")

plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Cost (MSE)', fontsize=12)
plt.title('Effect of Batch Size on Training Stability', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Q9_batch_size_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("Batch Size Impact Analysis")
print("="*70)

final_costs = [results[bs]['cost_history'][-1] for bs in batch_sizes]
final_costs.append(cost_bgd[-1])
all_batch_sizes = batch_sizes + ['Full']

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(all_batch_sizes)), final_costs, color='steelblue', edgecolor='black')
plt.xticks(range(len(all_batch_sizes)), [str(bs) for bs in all_batch_sizes])
plt.xlabel('Batch Size', fontsize=12)
plt.ylabel('Final Cost (MSE)', fontsize=12)
plt.title('Final Cost vs Batch Size', fontsize=14)
plt.grid(axis='y', alpha=0.3)

for bar, cost in zip(bars, final_costs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{cost:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('Q9_final_cost_vs_batch.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, batch_size in enumerate(batch_sizes):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    cost_history = results[batch_size]['cost_history']
    ax.plot(range(1, iterations + 1), cost_history, 'b-', linewidth=2)
    ax.set_title(f'Batch Size = {batch_size}', fontsize=12)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.grid(True, alpha=0.3)

axes[1, 2].plot(range(1, iterations + 1), cost_bgd, 'k-', linewidth=2)
axes[1, 2].set_title('Full Batch GD', fontsize=12)
axes[1, 2].set_xlabel('Iterations')
axes[1, 2].set_ylabel('Cost')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Q9_individual_batch_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nObservations:")
print("1. Small batch sizes (8, 16):")
print("   - More noise in convergence")
print("   - Can escape local minima")
print("   - Faster updates but less stable")

print("\n2. Medium batch sizes (32, 64):")
print("   - Good balance of speed and stability")
print("   - Common choice in practice")

print("\n3. Large batch sizes (128, Full):")
print("   - Smoother convergence")
print("   - May get stuck in local minima")
print("   - Slower updates but more stable")

print("\n4. Training Stability:")
print("   - Smaller batches = more variance in gradients")
print("   - Larger batches = more consistent gradients")
print("   - Trade-off between speed and stability")

print("\nQ9 completed successfully!")
print("Plots saved as Q9_batch_size_analysis.png, Q9_final_cost_vs_batch.png, Q9_individual_batch_curves.png")
