# ============================================================================
# EXPERIMENT 19: Neural Network Analysis for Circular Data (Linear Activation)
# Objective: Show linear activation fails on non-linear data.
# Dataset: make_circles (non-linear)
# Source: Generated in-code, no external dataset needed
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from nn_helpers import SimpleNeuralNetwork, prepare_binary_data, plot_decision_boundary

np.random.seed(42)


if __name__ == "__main__":
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.4, random_state=42)
    X_train, X_test, y_train, y_test, scaler = prepare_binary_data(X, y)

    print("="*60)
    print("EXPERIMENT 19: NN Circular Data (Linear Activation)")
    print("="*60)
    print(f"\nDataset: make_circles (300 samples, 2 features)")
    print(f"Activation: Linear")
    print(f"Hidden neurons: 4")

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation='linear', learning_rate=0.01)
    losses, accuracies = nn.fit(X_train, y_train, epochs=500, verbose=True)

    train_acc = np.mean(nn.predict(X_train) == y_train.ravel())
    test_acc = np.mean(nn.predict(X_test) == y_test.ravel())

    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_decision_boundary(nn, X_train, y_train, f'Decision Boundary (Train)\nAccuracy: {train_acc:.2%}', scaler, axes[0])
    plot_decision_boundary(nn, X_test, y_test, f'Decision Boundary (Test)\nAccuracy: {test_acc:.2%}', scaler, axes[1])

    axes[2].plot(losses, 'b-', linewidth=2, label='Loss')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('Training Loss Curve', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Exp19_nn_circular_linear.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Linear activation CANNOT solve circular data")
    print("2. Decision boundary is still a straight line (~50% accuracy)")
    print("3. This is the same limitation shown in Exp 17")
    print("4. Need non-linear activations (ReLU, tanh, sigmoid) to solve this")

    print("\nPlot saved as Exp19_nn_circular_linear.png")
    print("Exp 19 completed successfully!")
