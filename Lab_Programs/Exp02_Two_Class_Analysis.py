# ============================================================================
# EXPERIMENT 2: Two-Class Confusion Matrix Analysis using Python
# Objective: Go beyond the raw matrix - derive precision, recall, F1, specificity
#            by hand and verify against sklearn.
# Dataset: Pima Indians Diabetes Database
# Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

np.random.seed(42)

data = pd.read_csv('diabetes.csv')
print("="*60)
print("EXPERIMENT 2: Two-Class Confusion Matrix Analysis")
print("="*60)
print(f"\nDataset Shape: {data.shape}")
print(f"\nFirst 5 rows:")
print(data.head())

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    data[col] = data[col].replace(0, data[col].median())

print(f"\nAfter zero imputation:")
print(data[zero_cols].describe())

X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000, C=0.01, solver='lbfgs', random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = 2 * (precision * recall) / (precision + recall)

print("\n" + "="*60)
print("HAND-CALCULATED METRICS")
print("="*60)
print(f"\nConfusion Matrix:")
print(f"  TN={tn}, FP={fp}")
print(f"  FN={fn}, TP={tp}")
print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\n" + "="*60)
print("SKLEARN CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

print("\n" + "="*60)
print("COMPARISON: Hand-calculated vs sklearn")
print("="*60)
sk_report = classification_report(y_test, y_pred, output_dict=True)
print(f"{'Metric':<15} {'Hand-calculated':<18} {'sklearn':<15} {'Match'}")
print("-"*60)
print(f"{'Accuracy':<15} {accuracy:<18.4f} {sk_report['accuracy']:<15.4f} {'Yes' if abs(accuracy - sk_report['accuracy']) < 0.01 else 'No'}")
print(f"{'Precision':<15} {precision:<18.4f} {sk_report['1']['precision']:<15.4f} {'Yes' if abs(precision - sk_report['1']['precision']) < 0.01 else 'No'}")
print(f"{'Recall':<15} {recall:<18.4f} {sk_report['1']['recall']:<15.4f} {'Yes' if abs(recall - sk_report['1']['recall']) < 0.01 else 'No'}")
print(f"{'F1-Score':<15} {f1:<18.4f} {sk_report['1']['f1-score']:<15.4f} {'Yes' if abs(f1 - sk_report['1']['f1-score']) < 0.01 else 'No'}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

from sklearn.metrics import ConfusionMatrixDisplay


if __name__ == "__main__":
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Non-Diabetic', 'Diabetic'], ax=axes[0], cmap='Blues')
    axes[0].set_title('Confusion Matrix - Pima Diabetes', fontsize=14)

    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    values = [accuracy, precision, recall, specificity, f1]
    colors = ['steelblue', 'coral', 'forestgreen', 'orange', 'purple']
    bars = axes[1].bar(metrics, values, color=colors, edgecolor='black')
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Performance Metrics', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Exp02_two_class_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nPlot saved as Exp02_two_class_analysis.png")
    print("Exp 2 completed successfully!")
