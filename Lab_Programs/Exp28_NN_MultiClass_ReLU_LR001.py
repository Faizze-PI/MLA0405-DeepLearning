# ============================================================================
# EXPERIMENT 28: NN Multi-Class Data (ReLU, Learning Rate 0.001)
# Objective: Same low-learning-rate effect on multi-class data (paired with Exp 31).
# Dataset: make_blobs (same as Exp 20/24/26 for comparison)
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
    print("EXPERIMENT 28: NN Multi-Class Data (ReLU, LR=0.001)")
    print("="*60)
    print(f"\nDataset: make_blobs (300 samples, 3 classes)")
    print(f"Activation: ReLU")
    print(f"Hidden neurons: 6")
    print(f"Learning Rate: 0.001 (very small)")

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=6, output_size=3, activation='relu', learning_rate=0.001)
    losses, accuracies = nn.fit(X_train, y_train, epochs=500, verbose=True)

    train_pred = nn.predict(X_train)
    test_pred = nn.predict(X_test)
    train_acc = np.mean(train_pred == np.argmax(y_train, axis=1))
    test_acc = np.mean(test_pred == np.argmax(y_test, axis=1))

    print(f"\nAt 500 epochs:")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

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
    plt.savefig('Exp28_nn_multi_class_low_lr.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. LR=0.001 shows slow, gradual improvement")
    print("2. Likely still under-converged at 500 epochs")
    print("3. Compare with Exp 31 (LR=0.03) for dramatic speed difference")
    print("4. Report exact accuracy reached vs 'converged' baseline")

    print("\nPlot saved as Exp28_nn_multi_class_low_lr.png")
    print("Exp 28 completed successfully!")
