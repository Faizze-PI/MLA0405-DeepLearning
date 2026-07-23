# ============================================================================
# QUESTION 4: Neural Networks
# Design a simple neural network with one hidden layer and train it on a small
# dataset. Analyze how the network learns non-linear patterns.
# ============================================================================
# Dataset: Moons Dataset (Non-Linear Classification)
# Link: https://www.kaggle.com/datasets/emadmakhlouf/linearly-inseperable-dataset
# Note: Using sklearn.datasets.make_moons() for synthetic generation
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        return loss

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]

        dz2 = y_pred - y_true
        dW2 = (1 / m) * self.a1.T.dot(dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = (1 / m) * X.T.dot(dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)

            dW1, db1, dW2, db2 = self.backward(X, y, y_pred)
            self.update_weights(dW1, db1, dW2, db2, learning_rate)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        return losses

nn = NeuralNetwork(input_size=2, hidden_size=8, output_size=1)

print("Training Neural Network...")
print("="*50)
losses = nn.train(X_train_scaled, y_train, epochs=1000, learning_rate=0.1)

y_train_pred = nn.forward(X_train_scaled)
y_test_pred = nn.forward(X_test_scaled)

train_accuracy = np.mean((y_train_pred > 0.5).astype(int) == y_train) * 100
test_accuracy = np.mean((y_test_pred > 0.5).astype(int) == y_test) * 100

print(f"\nTraining Accuracy: {train_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(losses, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss Curve', fontsize=14)
axes[0].grid(True, alpha=0.3)

xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                      np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid_points)
Z = nn.forward(grid_scaled)
Z = Z.reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.8)
axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
axes[1].scatter(X[y.ravel()==0, 0], X[y.ravel()==0, 1], c='red', label='Class 0', edgecolors='k', s=30, alpha=0.7)
axes[1].scatter(X[y.ravel()==1, 0], X[y.ravel()==1, 1], c='blue', label='Class 1', edgecolors='k', s=30, alpha=0.7)
axes[1].set_xlabel('Feature 1', fontsize=12)
axes[1].set_ylabel('Feature 2', fontsize=12)
axes[1].set_title('Decision Boundary', fontsize=14)
axes[1].legend(fontsize=10)

axes[2].bar(['Training', 'Test'], [train_accuracy, test_accuracy], color=['steelblue', 'coral'], edgecolor='black')
axes[2].set_ylabel('Accuracy (%)', fontsize=12)
axes[2].set_title('Model Accuracy', fontsize=14)
axes[2].set_ylim(85, 100)
axes[2].grid(axis='y', alpha=0.3)
for i, v in enumerate([train_accuracy, test_accuracy]):
    axes[2].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('Q4_neural_network.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nQ4 completed successfully!")
print("Plot saved as Q4_neural_network.png")
