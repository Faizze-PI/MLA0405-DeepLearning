# ============================================================================
# QUESTION 3: Building Machine Learning Algorithms
# Build a complete machine learning model (data preprocessing, training, testing)
# using any classification dataset and evaluate performance using accuracy and
# confusion matrix.
# ============================================================================
# Dataset: Breast Cancer Wisconsin (Diagnostic) Data Set
# Link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

data = pd.read_csv('data.csv')

print("Dataset Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

data = data.drop('id', axis=1, errors='ignore')
data = data.drop('Unnamed: 32', axis=1, errors='ignore')

le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

X = data.drop('diagnosis', axis=1).values
y = data['diagnosis'].values

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution: {np.bincount(y)}")
print(f"Classes: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train_scaled.shape[0]}")
print(f"Test set size: {X_test_scaled.shape[0]}")

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {'accuracy': accuracy, 'confusion_matrix': cm, 'predictions': y_pred}
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_result = results[best_model_name]

print(f"\n{'='*60}")
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {best_result['accuracy'] * 100:.2f}%")
print(f"{'='*60}")

print(f"\nClassification Report ({best_model_name}):")
print(classification_report(y_test, best_result['predictions'], target_names=le.classes_))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, result) in enumerate(results.items()):
    cm = result['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=le.classes_, yticklabels=le.classes_)
    axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]*100:.2f}%', fontsize=12)
    axes[idx].set_ylabel('Actual Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig('Q3_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

model_names = list(results.keys())
accuracies = [results[name]['accuracy'] * 100 for name in model_names]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, accuracies, color=['steelblue', 'coral', 'forestgreen'], edgecolor='black')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Comparison - Breast Cancer Classification', fontsize=14)
plt.ylim(90, 100)
plt.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('Q3_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nQ3 completed successfully!")
print("Plots saved as Q3_confusion_matrices.png and Q3_model_comparison.png")
