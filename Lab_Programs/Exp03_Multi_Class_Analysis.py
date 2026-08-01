# ============================================================================
# EXPERIMENT 3: Multi-Class Confusion Matrix Performance Analysis
# Objective: Extend confusion-matrix analysis to >2 classes and understand
#            per-class vs macro/micro-averaged metrics.
# Dataset: Iris Species (sklearn built-in)
# Source: sklearn.datasets.load_iris() or https://www.kaggle.com/datasets/uciml/iris
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

np.random.seed(42)


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print("="*60)
    print("EXPERIMENT 3: Multi-Class Confusion Matrix Analysis")
    print("="*60)
    print(f"\nDataset: Iris ({X.shape[0]} samples, {X.shape[1]} features)")
    print(f"Classes: {list(class_names)}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(cm)

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, ax=axes[0], cmap='Blues')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp_norm = ConfusionMatrixDisplay(cm_normalized, display_labels=class_names)
    disp_norm.plot(ax=axes[1], cmap='Blues', values_format='.2f')
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    axes[2].bar(range(X.shape[1]), importances[indices], color='steelblue', edgecolor='black')
    axes[2].set_xticks(range(X.shape[1]))
    axes[2].set_xticklabels([iris.feature_names[i] for i in indices], rotation=45, ha='right')
    axes[2].set_ylabel('Feature Importance')
    axes[2].set_title('Feature Importance (Random Forest)', fontsize=14)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('Exp03_multi_class_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Setosa is perfectly classified (easiest - most separable)")
    print("2. Versicolor and Virginica have some overlap (confused with each other)")
    print("3. Petal length/width are the most important features")
    print("4. Macro-avg vs weighted-avg shows class balance effect")

    print("\nPlot saved as Exp03_multi_class_analysis.png")
    print("Exp 3 completed successfully!")
