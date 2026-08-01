# ============================================================================
# EXPERIMENT 11: Performance Evaluation of Decision Tree using Iris Dataset
# Objective: Visualize how a decision tree splits feature space and understand
#            overfitting via max_depth.
# Dataset: Iris Species (same split as Exp 8-10 for consistency)
# Source: sklearn.datasets.load_iris()
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    dt_unrestricted = DecisionTreeClassifier(max_depth=None, random_state=42)
    dt_unrestricted.fit(X_train, y_train)
    train_acc_unrestricted = dt_unrestricted.score(X_train, y_train)
    test_acc_unrestricted = dt_unrestricted.score(X_test, y_test)

    dt_depth3 = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_depth3.fit(X_train, y_train)
    train_acc_depth3 = dt_depth3.score(X_train, y_train)
    test_acc_depth3 = dt_depth3.score(X_test, y_test)

    y_pred = dt_depth3.predict(X_test)

    print("="*60)
    print("EXPERIMENT 11: Decision Tree on Iris Dataset")
    print("="*60)
    print(f"\nDataset: Iris ({X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes)")

    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    print(f"\n{'Model':<25} {'Train Accuracy':<18} {'Test Accuracy':<18} {'Overfitting?'}")
    print("-"*70)
    print(f"{'Unrestricted (None)':<25} {train_acc_unrestricted:<18.4f} {test_acc_unrestricted:<18.4f} {'Yes' if train_acc_unrestricted - test_acc_unrestricted > 0.1 else 'No'}")
    print(f"{'Depth=3':<25} {train_acc_depth3:<18.4f} {test_acc_depth3:<18.4f} {'Yes' if train_acc_depth3 - test_acc_depth3 > 0.1 else 'No'}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].bar(['Unrestricted', 'Depth=3'], [test_acc_unrestricted, test_acc_depth3], color=['coral', 'steelblue'], edgecolor='black')
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].set_title('Decision Tree: Unrestricted vs Depth-Limited', fontsize=14)
    axes[0].set_ylim(0.8, 1.05)
    axes[0].grid(axis='y', alpha=0.3)

    for i, v in enumerate([test_acc_unrestricted, test_acc_depth3]):
        axes[0].text(i, v + 0.01, f'{v:.2%}', ha='center', fontsize=12, fontweight='bold')

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, ax=axes[1], cmap='Blues')
    axes[1].set_title('Confusion Matrix (Depth=3)', fontsize=14)

    axes[2].clear()
    plot_tree(dt_depth3, feature_names=iris.feature_names, class_names=class_names.tolist(),
              filled=True, rounded=True, ax=axes[2], fontsize=8)
    axes[2].set_title('Decision Tree (max_depth=3)', fontsize=14)

    plt.tight_layout()
    plt.savefig('Exp11_decision_tree_iris.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Unrestricted tree: 100% train accuracy = overfitting")
    print("2. Depth=3 tree: Good generalization, fewer overfits")
    print("3. Trees split on petal length/width first (most informative)")
    print("4. Limiting max_depth reduces variance but increases bias")

    print("\nPlot saved as Exp11_decision_tree_iris.png")
    print("Exp 11 completed successfully!")
