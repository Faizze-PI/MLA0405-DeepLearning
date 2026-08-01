# ============================================================================
# QUESTION 6: Perceptron and Multilayer Perceptron (MLP)
# a) Implement the Perceptron algorithm for binary classification and test it
#    on linearly separable and non-linearly separable datasets. Analyze the results.
# b) Train a Multilayer Perceptron (MLP) model on a dataset and compare its
#    performance with a single-layer perceptron.
# ============================================================================
# Dataset: Moons Dataset (Non-Linear Classification)
# Link: https://www.kaggle.com/datasets/berkayalan/sklearn-moons-data-set
# Note: Using sklearn.datasets.make_classification and make_moons for generation
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            errors = 0
            for idx, x_i in enumerate(X):
                prediction = self.predict(x_i)
                update = self.learning_rate * (y[idx] - prediction)
                self.weights += update * x_i
                self.bias += update
                errors += int(update != 0.0)
            self.errors.append(errors)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

class MLP:
    def __init__(self, hidden_size=10, learning_rate=0.01, n_iterations=1000):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        self.W1 = np.random.randn(n_features, self.hidden_size) * 0.5
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, 1) * 0.5
        self.b2 = np.zeros((1, 1))

        self.losses = []

        for _ in range(self.n_iterations):
            z1 = X.dot(self.W1) + self.b1
            a1 = self.sigmoid(z1)
            z2 = a1.dot(self.W2) + self.b2
            a2 = self.sigmoid(z2)

            loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))
            self.losses.append(loss)

            dz2 = a2 - y
            dW2 = (1 / n_samples) * a1.T.dot(dz2)
            db2 = (1 / n_samples) * np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2.dot(self.W2.T)
            dz1 = da1 * self.sigmoid_derivative(a1)
            dW1 = (1 / n_samples) * X.T.dot(dz1)
            db1 = (1 / n_samples) * np.sum(dz1, axis=0, keepdims=True)

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

    def predict(self, X):
        z1 = X.dot(self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return (a2 >= 0.5).astype(int).ravel()

X_linear, y_linear = make_classification(n_samples=500, n_features=2, n_redundant=0,
                                          n_informative=2, random_state=1, n_clusters_per_class=1)

X_nonlinear, y_nonlinear = make_moons(n_samples=500, noise=0.2, random_state=42)

print("="*70)
print("PART A: Perceptron Algorithm")
print("="*70)

print("\n1. Linearly Separable Dataset:")
perceptron_linear = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron_linear.fit(X_linear, y_linear)
y_pred_linear = perceptron_linear.predict(X_linear)
accuracy_linear = accuracy_score(y_linear, y_pred_linear) * 100
print(f"   Accuracy: {accuracy_linear:.2f}%")

print("\n2. Non-Linearly Separable Dataset:")
perceptron_nonlinear = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron_nonlinear.fit(X_nonlinear, y_nonlinear)
y_pred_nonlinear = perceptron_nonlinear.predict(X_nonlinear)
accuracy_nonlinear = accuracy_score(y_nonlinear, y_pred_nonlinear) * 100
print(f"   Accuracy: {accuracy_nonlinear:.2f}%")

print("\n" + "="*70)
print("PART B: Multilayer Perceptron (MLP)")
print("="*70)

print("\n1. Linearly Separable Dataset:")
mlp_linear = MLP(hidden_size=10, learning_rate=0.1, n_iterations=1000)
mlp_linear.fit(X_linear, y_linear)
y_pred_mlp_linear = mlp_linear.predict(X_linear)
accuracy_mlp_linear = accuracy_score(y_linear, y_pred_mlp_linear) * 100
print(f"   Accuracy: {accuracy_mlp_linear:.2f}%")

print("\n2. Non-Linearly Separable Dataset:")
mlp_nonlinear = MLP(hidden_size=10, learning_rate=0.1, n_iterations=1000)
mlp_nonlinear.fit(X_nonlinear, y_nonlinear)
y_pred_mlp_nonlinear = mlp_nonlinear.predict(X_nonlinear)
accuracy_mlp_nonlinear = accuracy_score(y_nonlinear, y_pred_mlp_nonlinear) * 100
print(f"   Accuracy: {accuracy_mlp_nonlinear:.2f}%")

print("\n" + "="*70)
print("Comparison Summary")
print("="*70)
print(f"\n{'Dataset':<25} {'Perceptron':<15} {'MLP':<15}")
print("-"*55)
print(f"{'Linear Separable':<25} {accuracy_linear:<15.2f} {accuracy_mlp_linear:<15.2f}")
print(f"{'Non-Linear (Moons)':<25} {accuracy_nonlinear:<15.2f} {accuracy_mlp_nonlinear:<15.2f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].scatter(X_linear[y_linear==0, 0], X_linear[y_linear==0, 1], c='red', label='Class 0', edgecolors='k', s=30)
axes[0, 0].scatter(X_linear[y_linear==1, 0], X_linear[y_linear==1, 1], c='blue', label='Class 1', edgecolors='k', s=30)
axes[0, 0].set_title('Linear Separable Data', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(X_nonlinear[y_nonlinear==0, 0], X_nonlinear[y_nonlinear==0, 1], c='red', label='Class 0', edgecolors='k', s=30)
axes[0, 1].scatter(X_nonlinear[y_nonlinear==1, 0], X_nonlinear[y_nonlinear==1, 1], c='blue', label='Class 1', edgecolors='k', s=30)
axes[0, 1].set_title('Non-Linear Data (Moons)', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(perceptron_linear.errors, 'b-', linewidth=2)
axes[0, 2].set_title('Perceptron Errors (Linear Data)', fontsize=12)
axes[0, 2].set_xlabel('Iteration')
axes[0, 2].set_ylabel('Misclassifications')
axes[0, 2].grid(True, alpha=0.3)

def plot_decision_boundary(ax, X, y, model, title, is_mlp=False):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    if is_mlp:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Class 0', edgecolors='k', s=30)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Class 1', edgecolors='k', s=30)
    ax.set_title(title, fontsize=12)
    ax.legend()

plot_decision_boundary(axes[1, 0], X_linear, y_linear, perceptron_linear, 'Perceptron (Linear Data)')
plot_decision_boundary(axes[1, 1], X_nonlinear, y_nonlinear, perceptron_nonlinear, 'Perceptron (Non-Linear Data)')
plot_decision_boundary(axes[1, 2], X_nonlinear, y_nonlinear, mlp_nonlinear, 'MLP (Non-Linear Data)', is_mlp=True)

plt.tight_layout()
plt.savefig('Q6_perceptron_mlp.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nQ6 completed successfully!")
print("Plot saved as Q6_perceptron_mlp.png")
