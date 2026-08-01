# ============================================================================
# EXPERIMENT 1: Demonstration of Confusion Matrix using Python
# Objective: Understand the mechanics of a confusion matrix independent of any
#            specific algorithm.
# Dataset: Synthetic (sklearn.datasets.make_classification)
# Source: Generated in-code, no external dataset needed
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

np.random.seed(42)


if __name__ == "__main__":
    X, y = make_classification(n_samples=300, n_features=2, n_classes=2, n_redundant=0, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("="*60)
    print("EXPERIMENT 1: Confusion Matrix Demonstration")
    print("="*60)
    print("\nConfusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

    print("\n" + classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Class 0', 'Class 1'], ax=axes[0], cmap='Blues')
    axes[0].set_title('Confusion Matrix Heatmap', fontsize=14)

    axes[1].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='blue', label='Class 0', edgecolors='k', s=50, alpha=0.7)
    axes[1].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='red', label='Class 1', edgecolors='k', s=50, alpha=0.7)
    axes[1].set_title('Test Data Visualization', fontsize=14)
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Exp01_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nPlot saved as Exp01_confusion_matrix.png")
    print("Exp 1 completed successfully!")
