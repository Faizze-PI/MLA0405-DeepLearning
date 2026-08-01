# ============================================================================
# EXPERIMENT 25: Neural Network Analysis for Circular Data (Tanh Activation)
# Objective: Complete the 3-way activation comparison (linear/ReLU/tanh) on
#            circular data (paired with Exp 19, Exp 21).
# Dataset: make_circles (same as Exp 19/21 for fair comparison)
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
    print("EXPERIMENT 25: NN Circular Data (Tanh Activation)")
    print("="*60)
    print(f"\nDataset: make_circles (300 samples, 2 features)")
    print(f"Activation: Tanh")
    print(f"Hidden neurons: 8")

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=8, output_size=1, activation='tanh', learning_rate=0.01)
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
    plt.savefig('Exp25_nn_circular_tanh.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("3-WAY ACTIVATION COMPARISON (Circular Data)")
    print("="*60)
    print(f"\n{'Activation':<15} {'Train Acc':<12} {'Test Acc':<12}")
    print("-"*39)
    print(f"{'Linear (Exp19)':<15} {0.4208:<12.4f} {0.4000:<12.4f}")
    print(f"{'ReLU (Exp21)':<15} {0.6625:<12.4f} {0.6833:<12.4f}")
    print(f"{'Tanh (Exp25)':<15} {train_acc:<12.4f} {test_acc:<12.4f}")

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Tanh outperforms linear and may outperform ReLU on circular data")
    print("2. Tanh is zero-centered (-1 to 1) -> better gradient flow")
    print("3. Smooth activation allows smoother decision boundaries")
    print("4. This is the most instructive 3-way comparison figure")

    print("\nPlot saved as Exp25_nn_circular_tanh.png")
    print("Exp 25 completed successfully!")
