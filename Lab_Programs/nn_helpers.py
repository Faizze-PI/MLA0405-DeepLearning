"""
Shared utilities for Deep Learning Lab Experiments 18-32.
Provides: SimpleNeuralNetwork class, data generators, plotting helpers.
Architecture: Input(2) -> Dense(hidden, activation) -> Dense(output, sigmoid/softmax)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Activation Functions and Their Derivatives ---
# Each derivative receives the POST-ACTIVATION value (a) or PRE-ACTIVATION (z)
# depending on the mathematical form that simplifies the computation.

def sigmoid(z):
    """sigma(z) = 1 / (1 + e^-z). Clips to avoid overflow."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    """d/da sigma = sigma * (1 - sigma). Takes post-activation a=sigma(z)."""
    return a * (1 - a)

def relu(z):
    """ReLU(z) = max(0, z)."""
    return np.maximum(0, z)

def relu_derivative(z):
    """d/dz ReLU = 1 if z>0, else 0. Takes pre-activation z (step function)."""
    return (z > 0).astype(float)

def tanh(z):
    """tanh(z). Output in (-1, 1)."""
    return np.tanh(z)

def tanh_derivative(a):
    """d/da tanh = 1 - a^2. Takes post-activation a=tanh(z)."""
    return 1 - a ** 2

def linear(z):
    """Identity activation: f(z) = z."""
    return z

def linear_derivative(z):
    """d/dz linear = 1 everywhere. Takes pre-activation z."""
    return np.ones_like(z)

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'linear': (linear, linear_derivative)
}

class SimpleNeuralNetwork:
    """
    2-layer feedforward neural network for binary/multi-class classification.

    Architecture:
        z1 = X @ W1 + b1          (linear transform)
        a1 = activation(z1)       (non-linearity)
        z2 = a1 @ W2 + b2         (linear transform)
        a2 = sigmoid(z2) or softmax(z2)  (output activation)

    Loss: Binary cross-entropy (binary) or categorical cross-entropy (multi-class).
    Training: Full-batch gradient descent via backpropagation.
    """

    def __init__(self, input_size, hidden_size, output_size, activation='relu', learning_rate=0.01):
        self.lr = learning_rate
        self.activation_fn, self.activation_deriv = ACTIVATIONS[activation]

        # He-inspired init: scale by 0.5 for stability with small networks
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """Forward pass: compute a2 = output probabilities from input X."""
        # Hidden layer: linear transform + activation
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.activation_fn(self.z1)

        # Output layer: linear transform + output activation
        self.z2 = self.a1.dot(self.W2) + self.b2

        if self.W2.shape[1] > 1:
            # Multi-class: softmax with numerical stability (subtract max)
            exp_z = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
            self.a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            # Binary: sigmoid
            self.a2 = sigmoid(self.z2)

        return self.a2

    def compute_loss(self, y_true, y_pred):
        """
        Cross-entropy loss.
        Multi-class: L = -mean(sum(y * log(y_pred)))
        Binary: L = -mean(y*log(p) + (1-y)*log(1-p))
        1e-8 added for numerical stability (avoid log(0)).
        """
        if self.W2.shape[1] > 1:
            loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
        else:
            loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return loss

    def backward(self, X, y_true):
        """
        Backpropagation: compute gradients and update weights.

        Gradient flow (chain rule):
          Output layer:
            dz2 = dL/dz2 = a2 - y           (for cross-entropy + sigmoid/softmax)
            dW2 = (1/m) * a1.T @ dz2        (gradient for W2)
            db2 = (1/m) * sum(dz2)           (gradient for b2)

          Hidden layer:
            da1 = dz2 @ W2.T                 (propagate error to hidden)
            dz1 = da1 * activation'(a1)      (apply chain rule through activation)
            dW1 = (1/m) * X.T @ dz1          (gradient for W1)
            db1 = (1/m) * sum(dz1)           (gradient for b1)

          Weight update: W -= lr * dW (gradient descent).
        """
        m = X.shape[0]

        # Output layer gradients
        dz2 = self.a2 - y_true                                       # dL/dz2
        dW2 = (1 / m) * self.a1.T.dot(dz2)                           # dL/dW2
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)           # dL/db2

        # Hidden layer gradients (chain rule through activation)
        da1 = dz2.dot(self.W2.T)                                     # dL/da1
        dz1 = da1 * self.activation_deriv(self.a1)                   # dL/dz1 (chain rule)
        dW1 = (1 / m) * X.T.dot(dz1)                                 # dL/dW1
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)           # dL/db1

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=500, verbose=False):
        """Train the network for N epochs using full-batch gradient descent."""
        losses = []
        accuracies = []

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)

            self.backward(X, y)

            if verbose and (epoch + 1) % 100 == 0:
                if y.shape[1] > 1:
                    acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                else:
                    acc = np.mean((y_pred > 0.5).astype(int) == y)
                accuracies.append(acc)
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        return losses, accuracies

    def predict(self, X):
        """Return class predictions (hard labels) from forward pass."""
        y_pred = self.forward(X)
        if self.W2.shape[1] > 1:
            return np.argmax(y_pred, axis=1)
        else:
            return (y_pred > 0.5).astype(int).ravel()

def make_spiral(n_points=150, noise=0.5):
    """Generate a two-arm spiral dataset (non-linearly separable)."""
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))
    return X, y

def prepare_binary_data(X, y, test_size=0.2, random_state=42):
    """Split, scale, and reshape binary data for the neural network."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1), scaler

def prepare_multiclass_data(X, y, test_size=0.2, random_state=42):
    """Split, scale, one-hot-encode, and return multiclass data."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = np.eye(len(np.unique(y_encoded)))[y_encoded]

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def plot_decision_boundary(model, X, y, title, scaler=None, ax=None):
    """Plot 2D decision boundary by predicting on a meshgrid."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    if scaler is not None:
        grid_points = scaler.transform(grid_points)

    if grid_points.shape[1] > 2:
        grid_points_2d = np.c_[grid_points[:, 0], grid_points[:, 1]]
    else:
        grid_points_2d = grid_points

    Z = model.predict(grid_points_2d)
    Z = Z.reshape(xx.shape)

    if y.ndim > 1 and y.shape[1] > 1:
        y_labels = np.argmax(y, axis=1)
    else:
        y_labels = y.ravel()

    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax.scatter(X[y_labels==0, 0], X[y_labels==0, 1], c='blue', label='Class 0', edgecolors='k', s=30)
    ax.scatter(X[y_labels==1, 0], X[y_labels==1, 1], c='red', label='Class 1', edgecolors='k', s=30)
    if len(np.unique(y_labels)) > 2:
        ax.scatter(X[y_labels==2, 0], X[y_labels==2, 1], c='green', label='Class 2', edgecolors='k', s=30)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
