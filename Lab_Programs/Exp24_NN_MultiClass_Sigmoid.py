# ============================================================================
# EXPERIMENT 24: Neural Network Analysis for Multi-Class Data (Sigmoid Activation)
# Objective: Apply sigmoid to multi-class blob problem for activation comparison.
# Dataset: make_blobs (same as Exp 20 for fair comparison)
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
    print("EXPERIMENT 24: NN Multi-Class Data (Sigmoid Activation)")
    print("="*60)
    print(f"\nDataset: make_blobs (300 samples, 3 classes)")
    print(f"Activation: Sigmoid")
    print(f"Hidden neurons: 6")

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=6, output_size=3, activation='sigmoid', learning_rate=0.03)
    losses, accuracies = nn.fit(X_train, y_train, epochs=800, verbose=True)

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
    plt.savefig('Exp24_nn_multi_class_sigmoid.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("COMPARISON: Linear vs Sigmoid on Multi-Class Data")
    print("="*60)
    print(f"\n{'Activation':<15} {'Train Acc':<12} {'Test Acc':<12}")
    print("-"*39)
    print(f"{'Linear (Exp20)':<15} {1.0000:<12.4f} {1.0000:<12.4f}")
    print(f"{'Sigmoid (Exp24)':<15} {train_acc:<12.4f} {test_acc:<12.4f}")

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Sigmoid can produce non-linear boundaries (unlike linear)")
    print("2. Converges slower than ReLU due to vanishing gradients")
    print("3. Needs more epochs (800 vs 500) to reach similar accuracy")
    print("4. May converge more slowly but can still solve the problem")

    print("\nPlot saved as Exp24_nn_multi_class_sigmoid.png")
    print("Exp 24 completed successfully!")
