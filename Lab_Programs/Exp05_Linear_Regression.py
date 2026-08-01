# ============================================================================
# EXPERIMENT 5: Performance Evaluation of Linear Regression
# Objective: Apply linear regression to a real continuous-target dataset and
#            evaluate with regression metrics.
# Dataset: USA Housing Dataset
# Source: https://www.kaggle.com/datasets/aaquibsiddiqui/usa-housing-dataset
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


if __name__ == "__main__":
    data = pd.read_csv('USA_Housing.csv')

    print("="*60)
    print("EXPERIMENT 5: Linear Regression on USA Housing")
    print("="*60)
    print(f"\nDataset Shape: {data.shape}")
    print(f"\nColumns: {list(data.columns)}")
    print(f"\nFirst 5 rows:")
    print(data.head())

    feature_cols = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                    'Avg. Area Number of Bedrooms', 'Area Population']

    X = data[feature_cols]
    y = data['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"\nR-squared Score: {r2:.4f}")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"Root Mean Squared Error: ${rmse:,.2f}")

    print("\n" + "="*60)
    print("FEATURE COEFFICIENTS")
    print("="*60)
    for name, coef in zip(feature_cols, model.coef_):
        print(f"  {name:<35} {coef:>15,.2f}")
    print(f"  {'Intercept':<35} {model.intercept_:>15,.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.5, color='steelblue', edgecolors='k', linewidth=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price ($)', fontsize=12)
    axes[0].set_ylabel('Predicted Price ($)', fontsize=12)
    axes[0].set_title(f'Actual vs Predicted\nR2 = {r2:.4f}', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, color='coral', edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Price ($)', fontsize=12)
    axes[1].set_ylabel('Residuals ($)', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(residuals, bins=30, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Residual ($)', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Residual Distribution', fontsize=14)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('Exp05_linear_regression.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nPlot saved as Exp05_linear_regression.png")
    print("Exp 5 completed successfully!")
