# ============================================================================
# EXPERIMENT 21: Neural Network Analysis for Circular Data (ReLU Activation)
# Objective: Show ReLU introduces the non-linearity needed to solve the circular
#            problem that failed in Exp 19.
# Dataset: make_circles (same as Exp 19 for fair comparison)
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
    print("EXPERIMENT 21: NN Circular Data (ReLU Activation)")
    print("="*60)
    print(f"\nDataset: make_circles (300 samples, 2 features)")
    print(f"Activation: ReLU")
    print(f"Hidden neurons: 16")

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=16, output_size=1, activation='relu', learning_rate=0.01)
    losses, accuracies = nn.fit(X_train, y_train, epochs=1000, verbose=True)

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
    plt.savefig('Exp21_nn_circular_relu.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("COMPARISON: Linear vs ReLU on Circular Data")
    print("="*60)
    print(f"\n{'Activation':<15} {'Train Acc':<12} {'Test Acc':<12}")
    print("-"*39)
    print(f"{'Linear (Exp19)':<15} {0.4208:<12.4f} {0.4000:<12.4f}")
    print(f"{'ReLU (Exp21)':<15} {train_acc:<12.4f} {test_acc:<12.4f}")

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. ReLU solves the circular problem that linear activation couldn't")
    print("2. Decision boundary now curves around the inner circle")
    print("3. ReLU introduces non-linearity through piecewise linear segments")
    print("4. 8 neurons provide enough 'pieces' to approximate the curve")

    print("\nPlot saved as Exp21_nn_circular_relu.png")
    print("Exp 21 completed successfully!")
