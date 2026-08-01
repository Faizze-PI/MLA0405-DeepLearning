# ============================================================================
# EXPERIMENT 7: Visualization of Logistic Regression (Sigmoid Function)
# Objective: Build intuition for why logistic regression uses the sigmoid to
#            map linear scores to probabilities.
# Dataset: Synthetic 2D binary blobs
# Source: Generated in-code, no external dataset needed
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)


if __name__ == "__main__":
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=42)

    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    print("="*60)
    print("EXPERIMENT 7: Logistic Regression & Sigmoid Visualization")
    print("="*60)
    print(f"\nAccuracy: {accuracy:.4f}")

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    z = np.linspace(-10, 10, 200)
    sigmoid_values = sigmoid(z)

    axes[0].plot(z, sigmoid_values, 'b-', linewidth=3)
    axes[0].axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
    axes[0].axvline(x=0, color='g', linestyle='--', linewidth=2, label='z = 0')
    axes[0].set_xlabel('z (Linear Score)', fontsize=12)
    axes[0].set_ylabel('sigmoid(z)', fontsize=12)
    axes[0].set_title('Sigmoid Function', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[1].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    axes[1].scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', edgecolors='k', s=50)
    axes[1].scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', edgecolors='k', s=50)
    axes[1].set_xlabel('Feature 1', fontsize=12)
    axes[1].set_ylabel('Feature 2', fontsize=12)
    axes[1].set_title(f'Decision Boundary (Accuracy: {accuracy:.2%})', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z_proba = Z_proba.reshape(xx.shape)

    contour = axes[2].contourf(xx, yy, Z_proba, levels=50, cmap='RdYlBu_r')
    axes[2].scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Class 0', edgecolors='k', s=50)
    axes[2].scatter(X[y==1, 0], X[y==1, 1], c='red', label='Class 1', edgecolors='k', s=50)
    axes[2].set_xlabel('Feature 1', fontsize=12)
    axes[2].set_ylabel('Feature 2', fontsize=12)
    axes[2].set_title('Probability Surface', fontsize=14)
    axes[2].legend()
    plt.colorbar(contour, ax=axes[2], label='P(Class=1)')

    plt.tight_layout()
    plt.savefig('Exp07_logistic_sigmoid.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Sigmoid maps any real number to (0, 1) - interpretable as probability")
    print("2. z = 0 -> sigmoid = 0.5 (decision boundary)")
    print("3. z >> 0 -> sigmoid approx 1 (confident Class 1)")
    print("4. z << 0 -> sigmoid approx 0 (confident Class 0)")
    print("5. Decision boundary is a straight line (linear model limitation)")

    print("\nPlot saved as Exp07_logistic_sigmoid.png")
    print("Exp 7 completed successfully!")
