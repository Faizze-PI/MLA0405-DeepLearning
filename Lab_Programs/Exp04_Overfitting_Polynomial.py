# ============================================================================
# EXPERIMENT 4: Analysis of Overfitting using Polynomial Regression
# Objective: Visually and numerically demonstrate the bias-variance tradeoff.
# Dataset: Synthetic (y = sin(x) + noise)
# Source: Generated in-code, no external dataset needed
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)


if __name__ == "__main__":
    n_samples = 30
    X_train = np.random.uniform(0, 2*np.pi, n_samples).reshape(-1, 1)
    y_train = np.sin(X_train.ravel()) + np.random.normal(0, 0.3, n_samples)

    X_test_dense = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
    y_true = np.sin(X_test_dense.ravel())

    degrees = [1, 3, 9, 15]

    print("="*60)
    print("EXPERIMENT 4: Overfitting Analysis with Polynomial Regression")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    train_mse = []
    test_mse = []

    for idx, degree in enumerate(degrees):
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test_dense)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)

        train_mse.append(mean_squared_error(y_train, y_train_pred))
        test_mse.append(mean_squared_error(y_true, y_test_pred))

        axes[idx].scatter(X_train, y_train, color='blue', label='Training Data', edgecolors='k', s=50, alpha=0.7)
        axes[idx].plot(X_test_dense, y_true, 'g-', linewidth=2, label='True sin(x)')
        axes[idx].plot(X_test_dense, y_test_pred, 'r-', linewidth=2, label=f'Poly (degree={degree})')
        axes[idx].set_title(f'Degree {degree}\nTrain MSE: {train_mse[-1]:.4f}, Test MSE: {test_mse[-1]:.4f}', fontsize=12)
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('y')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-2, 2)

        print(f"Degree {degree}: Train MSE = {train_mse[-1]:.4f}, Test MSE = {test_mse[-1]:.4f}")

    plt.tight_layout()
    plt.savefig('Exp04_overfitting_fits.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_mse, 'bo-', linewidth=2, markersize=8, label='Train MSE')
    plt.plot(degrees, test_mse, 'rs-', linewidth=2, markersize=8, label='Test MSE')
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('Bias-Variance Tradeoff', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(degrees)
    plt.tight_layout()
    plt.savefig('Exp04_mse_vs_degree.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Degree 1 (Underfitting): High bias, low variance - line can't capture sin(x)")
    print("2. Degree 3 (Good fit): Balanced bias-variance - captures shape well")
    print("3. Degree 9 (Overfitting): Low bias, high variance - fits noise in training data")
    print("4. Degree 15 (Severe overfitting): Extreme variance - wild oscillations")

    best_idx = np.argmin(test_mse)
    print(f"\nBest Test MSE: {test_mse[best_idx]:.4f} (Degree {degrees[best_idx]})")
    print(f"Test MSE range: [{min(test_mse):.4f}, {max(test_mse):.4f}]")

    print("\nPlots saved as Exp04_overfitting_fits.png and Exp04_mse_vs_degree.png")
    print("Exp 4 completed successfully!")
