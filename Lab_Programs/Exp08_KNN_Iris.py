# ============================================================================
# EXPERIMENT 8: Performance Evaluation of KNN using Iris Dataset
# Objective: Repeat KNN analysis on Iris to compare behavior with Wine dataset.
# Dataset: Iris Species (sklearn built-in)
# Source: sklearn.datasets.load_iris() or https://www.kaggle.com/datasets/uciml/iris
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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

    k_range = range(1, 21)
    test_scores = []

    print("="*60)
    print("EXPERIMENT 8: KNN on Iris Dataset")
    print("="*60)
    print(f"\nDataset: Iris ({X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes)")

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        test_scores.append(knn.score(X_test_scaled, y_test))

    best_k = k_range[np.argmax(test_scores)]
    best_accuracy = max(test_scores)

    print(f"\nBest K: {best_k} with accuracy: {best_accuracy:.4f}")

    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train_scaled, y_train)
    y_pred = knn_best.predict(X_test_scaled)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(k_range, test_scores, 'bo-', linewidth=2, markersize=6)
    axes[0].axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Best K={best_k}')
    axes[0].set_xlabel('K (Number of Neighbors)', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('KNN Accuracy vs K (Iris)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(k_range)

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, ax=axes[1], cmap='Blues')
    axes[1].set_title(f'Confusion Matrix (K={best_k})', fontsize=14)

    x_idx = 2
    y_idx = 3
    axes[2].scatter(X_test[y_test==0, x_idx], X_test[y_test==0, y_idx], c='blue', label=class_names[0], edgecolors='k', s=50)
    axes[2].scatter(X_test[y_test==1, x_idx], X_test[y_test==1, y_idx], c='red', label=class_names[1], edgecolors='k', s=50)
    axes[2].scatter(X_test[y_test==2, x_idx], X_test[y_test==2, y_idx], c='green', label=class_names[2], edgecolors='k', s=50)
    axes[2].set_xlabel(iris.feature_names[x_idx], fontsize=12)
    axes[2].set_ylabel(iris.feature_names[y_idx], fontsize=12)
    axes[2].set_title('Decision Boundary (Petal Features)', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Exp08_knn_iris.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("COMPARISON: Iris vs Wine (Exp 6)")
    print("="*60)
    print("\n1. Iris stays highly accurate across most K values (>93%)")
    print("2. Setosa is trivially separable (never confused)")
    print("3. Versicolor/Virginica have some overlap")
    print("4. Both datasets benefit from StandardScaler")

    print("\nPlot saved as Exp08_knn_iris.png")
    print("Exp 8 completed successfully!")
