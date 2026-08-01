# ============================================================================
# EXPERIMENT 13: Performance Evaluation of SVM using Iris Dataset
# Objective: Compare linear vs non-linear (RBF, polynomial) kernels for
#            classification and visualize margins.
# Dataset: Iris Species (same split as Exp 8-12 for consistency)
# Source: sklearn.datasets.load_iris()
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

    svm_linear = SVC(kernel='linear', random_state=42)
    svm_linear.fit(X_train_scaled, y_train)
    acc_linear = svm_linear.score(X_test_scaled, y_test)

    svm_rbf = SVC(kernel='rbf', random_state=42)
    svm_rbf.fit(X_train_scaled, y_train)
    acc_rbf = svm_rbf.score(X_test_scaled, y_test)

    svm_poly = SVC(kernel='poly', degree=3, random_state=42)
    svm_poly.fit(X_train_scaled, y_train)
    acc_poly = svm_poly.score(X_test_scaled, y_test)

    print("="*60)
    print("EXPERIMENT 13: SVM on Iris Dataset")
    print("="*60)
    print(f"\nDataset: Iris ({X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes)")

    print("\n" + "="*60)
    print("KERNEL COMPARISON")
    print("="*60)
    print(f"\n{'Kernel':<15} {'Accuracy':<12}")
    print("-"*27)
    print(f"{'Linear':<15} {acc_linear:<12.4f}")
    print(f"{'RBF':<15} {acc_rbf:<12.4f}")
    print(f"{'Polynomial':<15} {acc_poly:<12.4f}")

    x_idx = 2
    y_idx = 3
    X_train_2d = X_train_scaled[:, [x_idx, y_idx]]
    X_test_2d = X_test_scaled[:, [x_idx, y_idx]]

    svm_linear_2d = SVC(kernel='linear', random_state=42)
    svm_linear_2d.fit(X_train_2d, y_train)

    svm_rbf_2d = SVC(kernel='rbf', random_state=42)
    svm_rbf_2d.fit(X_train_2d, y_train)

    svm_poly_2d = SVC(kernel='poly', degree=3, random_state=42)
    svm_poly_2d.fit(X_train_2d, y_train)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    xx, yy = np.meshgrid(np.linspace(X_test_2d[:, 0].min()-1, X_test_2d[:, 0].max()+1, 200),
                          np.linspace(X_test_2d[:, 1].min()-1, X_test_2d[:, 1].max()+1, 200))

    models = [(svm_linear_2d, 'Linear Kernel'), (svm_rbf_2d, 'RBF Kernel'), (svm_poly_2d, 'Polynomial Kernel')]
    accuracies = [acc_linear, acc_rbf, acc_poly]

    for idx, (model, title) in enumerate(models):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        axes[idx].scatter(X_test[y_test==0, x_idx], X_test[y_test==0, y_idx], c='blue', label=class_names[0], edgecolors='k', s=50)
        axes[idx].scatter(X_test[y_test==1, x_idx], X_test[y_test==1, y_idx], c='red', label=class_names[1], edgecolors='k', s=50)
        axes[idx].scatter(X_test[y_test==2, x_idx], X_test[y_test==2, y_idx], c='green', label=class_names[2], edgecolors='k', s=50)

        if hasattr(model, 'support_vectors_'):
            axes[idx].scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                             s=200, facecolors='none', edgecolors='black', linewidths=2, label='Support Vectors')

        axes[idx].set_xlabel(iris.feature_names[x_idx], fontsize=12)
        axes[idx].set_ylabel(iris.feature_names[y_idx], fontsize=12)
        axes[idx].set_title(f'{title}\nAccuracy: {accuracies[idx]:.2%}', fontsize=14)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Exp13_svm_iris.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. RBF kernel typically performs best on Iris (non-linear boundaries)")
    print("2. Linear kernel works well when classes are mostly separable")
    print("3. Polynomial kernel adds complexity without always improving accuracy")
    print("4. Support vectors are the critical points defining the decision boundary")
    print("5. Mandatory StandardScaler for SVM (margin-based algorithm)")

    print("\nPlot saved as Exp13_svm_iris.png")
    print("Exp 13 completed successfully!")
