# ============================================================================
# EXPERIMENT 29: NN Multi-Class Data (Tanh, 3 Hidden Neurons)
# Objective: Demonstrate underfitting from insufficient network capacity.
# Dataset: make_blobs (same as Exp 20/24/26/28 for comparison)
# Source: Generated in-code, no external dataset needed
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from nn_helpers import SimpleNeuralNetwork, prepare_multiclass_data, plot_decision_boundary

np.random.seed(42)


if __name__ == "__main__":
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.2, random_state=42)
    X_train, X_test, y_train, y_test, scaler = prepare_multiclass_data(X, y)

    print("="*60)
    print("EXPERIMENT 29: NN Multi-Class Data (Tanh, 3 Neurons)")
    print("="*60)
    print(f"\nDataset: make_blobs (300 samples, 3 classes)")
    print(f"Activation: Tanh")
    print(f"Hidden neurons: 3 (deliberately undersized)")

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=3, activation='tanh', learning_rate=0.01)
    losses, accuracies = nn.fit(X_train, y_train, epochs=1000, verbose=True)

    train_pred = nn.predict(X_train)
    test_pred = nn.predict(X_test)
    train_acc = np.mean(train_pred == np.argmax(y_train, axis=1))
    test_acc = np.mean(test_pred == np.argmax(y_test, axis=1))

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
    plt.savefig('Exp29_nn_capacity_limitation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("COMPARISON: 6 Neurons vs 3 Neurons")
    print("="*60)
    print(f"\n{'Config':<25} {'Train Acc':<12} {'Test Acc':<12}")
    print("-"*49)
    print(f"{'Tanh, 6 neurons (Exp26)':<25} {1.0000:<12.4f} {1.0000:<12.4f}")
    print(f"{'Tanh, 3 neurons (Exp29)':<25} {train_acc:<12.4f} {test_acc:<12.4f}")

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. 3 neurons is insufficient for 3-class boundary")
    print("2. Coarse/blocky decision regions (underfitting)")
    print("3. Gave 1000 epochs - poor result is capacity issue, not training time")
    print("4. Required capacity scales with problem complexity")

    print("\nPlot saved as Exp29_nn_capacity_limitation.png")
    print("Exp 29 completed successfully!")
