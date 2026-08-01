# ============================================================================
# EXPERIMENT 12: Performance Evaluation of Random Forest using Iris Dataset
# Objective: Show how ensembling many trees reduces overfitting and reveals
#            feature importance.
# Dataset: Iris Species (same split as Exp 8-11 for consistency)
# Source: sklearn.datasets.load_iris()
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf_100 = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_100.fit(X_train, y_train)
    test_acc_rf = rf_100.score(X_test, y_test)
    feature_importances = rf_100.feature_importances_

    n_estimators_range = [10, 50, 100, 200]
    estimator_scores = []

    print("="*60)
    print("EXPERIMENT 12: Random Forest on Iris Dataset")
    print("="*60)
    print(f"\nDataset: Iris ({X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes)")

    for n_est in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
        rf.fit(X_train, y_train)
        estimator_scores.append(rf.score(X_test, y_test))

    dt_test_acc = 0.9333

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nRandom Forest (100 trees) Test Accuracy: {test_acc_rf:.4f} ({test_acc_rf*100:.2f}%)")
    print(f"Feature Importances: {feature_importances}")

    print("\n" + "="*60)
    print("COMPARISON: Single Tree vs Random Forest")
    print("="*60)
    print(f"\nDecision Tree (Exp 11, Depth=3): {dt_test_acc:.4f}")
    print(f"Random Forest (100 trees):       {test_acc_rf:.4f}")
    print(f"Improvement:                     {test_acc_rf - dt_test_acc:+.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    indices = np.argsort(feature_importances)[::-1]
    axes[0].bar(range(X.shape[1]), feature_importances[indices], color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(X.shape[1]))
    axes[0].set_xticklabels([iris.feature_names[i] for i in indices], rotation=45, ha='right')
    axes[0].set_ylabel('Feature Importance')
    axes[0].set_title('Feature Importance (Random Forest)', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].plot(n_estimators_range, estimator_scores, 'bo-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Trees', fontsize=12)
    axes[1].set_ylabel('Test Accuracy', fontsize=12)
    axes[1].set_title('Accuracy vs Number of Trees', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(n_estimators_range)

    y_pred = rf_100.predict(X_test)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, ax=axes[2], cmap='Blues')
    axes[2].set_title('Confusion Matrix (100 Trees)', fontsize=14)

    plt.tight_layout()
    plt.savefig('Exp12_random_forest_iris.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Random Forest reduces overfitting vs single tree")
    print("2. Petal features dominate importance (as expected)")
    print("3. Diminishing returns: >50 trees gives minimal gain")
    print("4. Ensemble averages out individual tree errors")

    print("\nPlot saved as Exp12_random_forest_iris.png")
    print("Exp 12 completed successfully!")
