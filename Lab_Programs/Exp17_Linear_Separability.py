# ============================================================================
# EXPERIMENT 17: Demonstration of Linear Separability using Logistic Regression
# Objective: Concretely show the limitation of linear models - motivating why
#            Experiments 18-32 introduce non-linear activations.
# Dataset: Synthetic (make_blobs + make_circles)
# Source: Generated in-code, no external dataset needed
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)


if __name__ == "__main__":
    X_sep, y_sep = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=42)
    X_nonsep, y_nonsep = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)

    lr_sep = LogisticRegression()
    lr_sep.fit(X_sep, y_sep)
    y_pred_sep = lr_sep.predict(X_sep)
    acc_sep = accuracy_score(y_sep, y_pred_sep)

    lr_nonsep = LogisticRegression()
    lr_nonsep.fit(X_nonsep, y_nonsep)
    y_pred_nonsep = lr_nonsep.predict(X_nonsep)
    acc_nonsep = accuracy_score(y_nonsep, y_pred_nonsep)

    print("="*60)
    print("EXPERIMENT 17: Linear Separability Demonstration")
    print("="*60)
    print(f"\nLinearly Separable Data: Accuracy = {acc_sep:.4f}")
    print(f"Non-Linear Data (Circles): Accuracy = {acc_nonsep:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    xx, yy = np.meshgrid(np.linspace(X_sep[:, 0].min()-1, X_sep[:, 0].max()+1, 200),
                          np.linspace(X_sep[:, 1].min()-1, X_sep[:, 1].max()+1, 200))
    Z_sep = lr_sep.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[0].contourf(xx, yy, Z_sep, alpha=0.4, cmap='RdYlBu')
    axes[0].scatter(X_sep[y_sep==0, 0], X_sep[y_sep==0, 1], c='blue', label='Class 0', edgecolors='k', s=50)
    axes[0].scatter(X_sep[y_sep==1, 0], X_sep[y_sep==1, 1], c='red', label='Class 1', edgecolors='k', s=50)
    axes[0].set_xlabel('Feature 1', fontsize=12)
    axes[0].set_ylabel('Feature 2', fontsize=12)
    axes[0].set_title(f'Linearly Separable Data\nAccuracy: {acc_sep:.2%}', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    xx, yy = np.meshgrid(np.linspace(X_nonsep[:, 0].min()-1, X_nonsep[:, 0].max()+1, 200),
                          np.linspace(X_nonsep[:, 1].min()-1, X_nonsep[:, 1].max()+1, 200))
    Z_nonsep = lr_nonsep.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[1].contourf(xx, yy, Z_nonsep, alpha=0.4, cmap='RdYlBu')
    axes[1].scatter(X_nonsep[y_nonsep==0, 0], X_nonsep[y_nonsep==0, 1], c='blue', label='Class 0', edgecolors='k', s=50)
    axes[1].scatter(X_nonsep[y_nonsep==1, 0], X_nonsep[y_nonsep==1, 1], c='red', label='Class 1', edgecolors='k', s=50)
    axes[1].set_xlabel('Feature 1', fontsize=12)
    axes[1].set_ylabel('Feature 2', fontsize=12)
    axes[1].set_title(f'Non-Linear Data (Circles)\nAccuracy: {acc_nonsep:.2%}', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Exp17_linear_separability.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Linear data: Logistic regression achieves ~100% accuracy")
    print("2. Non-linear data: LR fails (~50% = random guessing)")
    print("3. LR decision boundary is always a straight line")
    print("4. This motivates neural networks with non-linear activations")
    print("5. Experiments 18-32 will show how non-linear activations solve this")

    print("\nPlot saved as Exp17_linear_separability.png")
    print("Exp 17 completed successfully!")
