# ============================================================================
# EXPERIMENT 9: Performance Evaluation of Naive Bayes using Iris Dataset
# Objective: Compare a probabilistic generative classifier (Gaussian NB) against
#            the distance-based KNN from Exp 8.
# Dataset: Iris Species (same split as Exp 8 for fair comparison)
# Source: sklearn.datasets.load_iris()
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

np.random.seed(42)


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print("="*60)
    print("EXPERIMENT 9: Naive Bayes on Iris Dataset")
    print("="*60)
    print(f"\nDataset: Iris ({X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes)")
    print(f"\nModel: GaussianNB (no hyperparameters to tune)")
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, ax=axes[0], cmap='Blues')
    axes[0].set_title('Confusion Matrix - Naive Bayes', fontsize=14)

    x_idx = 2
    y_idx = 3
    xx, yy = np.meshgrid(np.linspace(X[:, x_idx].min()-0.5, X[:, x_idx].max()+0.5, 200),
                          np.linspace(X[:, y_idx].min()-0.5, X[:, y_idx].max()+0.5, 200))
    Z = model.predict(np.c_[np.ones_like(xx.ravel())*X[:, 0].mean(),
                             np.ones_like(xx.ravel())*X[:, 1].mean(),
                             xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[1].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    axes[1].scatter(X[y==0, x_idx], X[y==0, y_idx], c='blue', label=class_names[0], edgecolors='k', s=50)
    axes[1].scatter(X[y==1, x_idx], X[y==1, y_idx], c='red', label=class_names[1], edgecolors='k', s=50)
    axes[1].scatter(X[y==2, x_idx], X[y==2, y_idx], c='green', label=class_names[2], edgecolors='k', s=50)
    axes[1].set_xlabel(iris.feature_names[x_idx], fontsize=12)
    axes[1].set_ylabel(iris.feature_names[y_idx], fontsize=12)
    axes[1].set_title('Decision Boundary (Petal Features)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    algorithms = ['KNN (Exp 8)', 'Naive Bayes']
    knn_accuracy = 0.9667
    accuracies = [knn_accuracy, accuracy]
    colors = ['steelblue', 'coral']
    bars = axes[2].bar(algorithms, accuracies, color=colors, edgecolor='black')
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_title('KNN vs Naive Bayes Comparison', fontsize=14)
    axes[2].set_ylim(0.8, 1.05)
    axes[2].grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{acc:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Exp09_naive_bayes_iris.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("COMPARISON: KNN vs Naive Bayes")
    print("="*60)
    print(f"\n{'Metric':<20} {'KNN (Exp 8)':<15} {'Naive Bayes':<15}")
    print("-"*50)
    print(f"{'Accuracy':<20} {knn_accuracy:<15.4f} {accuracy:<15.4f}")
    print(f"{'Training Speed':<20} {'Instant':<15} {'Instant':<15}")
    print(f"{'Assumptions':<20} {'Distance':<15} {'Gaussian':<15}")

    print("\nPlot saved as Exp09_naive_bayes_iris.png")
    print("Exp 9 completed successfully!")
