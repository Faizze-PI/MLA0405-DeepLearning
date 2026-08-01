# ============================================================================
# EXPERIMENT 10: Performance Evaluation of Logistic Regression using Iris
# Objective: Apply multinomial logistic regression to 3-class Iris and interpret
#            learned coefficients.
# Dataset: Iris Species (same split as Exp 8/9 for consistency)
# Source: sklearn.datasets.load_iris()
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

np.random.seed(42)


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(solver='lbfgs', max_iter=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("="*60)
    print("EXPERIMENT 10: Logistic Regression on Iris Dataset")
    print("="*60)
    print(f"\nDataset: Iris ({X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes)")
    print(f"\nModel: LogisticRegression(solver='lbfgs', max_iter=200)")
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("\n" + "="*60)
    print("FEATURE COEFFICIENTS (interpretation)")
    print("="*60)
    print(f"\n{'Feature':<30} {'Setosa':<12} {'Versicolor':<12} {'Virginica':<12}")
    print("-"*66)
    for i, feature_name in enumerate(iris.feature_names):
        print(f"{feature_name:<30} {model.coef_[0][i]:<12.4f} {model.coef_[1][i]:<12.4f} {model.coef_[2][i]:<12.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, ax=axes[0], cmap='Blues')
    axes[0].set_title('Confusion Matrix - Logistic Regression', fontsize=14)

    feature_importance = np.abs(model.coef_).mean(axis=0)
    indices = np.argsort(feature_importance)[::-1]
    axes[1].bar(range(len(indices)), feature_importance[indices], color='steelblue', edgecolor='black')
    axes[1].set_xticks(range(len(indices)))
    axes[1].set_xticklabels([iris.feature_names[i] for i in indices], rotation=45, ha='right')
    axes[1].set_ylabel('Mean Absolute Coefficient')
    axes[1].set_title('Feature Importance (Multinomial LR)', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)

    x_idx = 2
    y_idx = 3
    xx, yy = np.meshgrid(np.linspace(X[:, x_idx].min()-0.5, X[:, x_idx].max()+0.5, 200),
                          np.linspace(X[:, y_idx].min()-0.5, X[:, y_idx].max()+0.5, 200))
    Z = model.predict(scaler.transform(np.c_[np.ones_like(xx.ravel())*X[:, 0].mean(),
                                              np.ones_like(xx.ravel())*X[:, 1].mean(),
                                              xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    axes[2].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    axes[2].scatter(X[y==0, x_idx], X[y==0, y_idx], c='blue', label=class_names[0], edgecolors='k', s=50)
    axes[2].scatter(X[y==1, x_idx], X[y==1, y_idx], c='red', label=class_names[1], edgecolors='k', s=50)
    axes[2].scatter(X[y==2, x_idx], X[y==2, y_idx], c='green', label=class_names[2], edgecolors='k', s=50)
    axes[2].set_xlabel(iris.feature_names[x_idx], fontsize=12)
    axes[2].set_ylabel(iris.feature_names[y_idx], fontsize=12)
    axes[2].set_title('Decision Boundary (Petal Features)', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Exp10_logistic_regression_iris.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Petal length/width have highest coefficients (most discriminative)")
    print("2. Sepal features contribute less to classification")
    print("3. Coefficients show which features drive each class decision")
    print("4. Multinomial LR trains all classes simultaneously")

    print("\nPlot saved as Exp10_logistic_regression_iris.png")
    print("Exp 10 completed successfully!")
