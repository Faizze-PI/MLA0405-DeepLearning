# ============================================================================
# EXPERIMENT 20: Neural Network Analysis for Multi-Class Data (Linear Activation)
# Objective: Extend the linear-activation limitation to a 3+ class setting.
# Dataset: make_blobs (3 classes)
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
    print("EXPERIMENT 20: NN Multi-Class Data (Linear Activation)")
    print("="*60)
    print(f"\nDataset: make_blobs (300 samples, 3 classes)")
    print(f"Activation: Linear")
    print(f"Hidden neurons: 6")

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=6, output_size=3, activation='linear', learning_rate=0.01)
    losses, accuracies = nn.fit(X_train, y_train, epochs=500, verbose=True)

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
    plt.savefig('Exp20_nn_multi_class_linear.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Linear activation still produces straight-line boundaries")
    print("2. May work if classes are already linearly separable")
    print("3. Multi-class softmax + linear = multinomial logistic regression")
    print("4. Compare with Exp 24 (sigmoid) and Exp 26 (tanh)")

    print("\nPlot saved as Exp20_nn_multi_class_linear.png")
    print("Exp 20 completed successfully!")
