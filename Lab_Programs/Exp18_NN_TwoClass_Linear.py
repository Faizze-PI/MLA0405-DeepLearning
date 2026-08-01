# ============================================================================
# EXPERIMENT 18: Neural Network Analysis for Two-Class Data (Linear Activation)
# Objective: Baseline - show a network with linear activation behaves like plain
#            logistic regression (no real advantage from "depth").
# Dataset: make_classification (linearly separable)
# Source: Generated in-code, no external dataset needed
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from nn_helpers import SimpleNeuralNetwork, prepare_binary_data, plot_decision_boundary

np.random.seed(42)


if __name__ == "__main__":
    X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, class_sep=1.5, random_state=42)

    X_train, X_test, y_train, y_test, scaler = prepare_binary_data(X, y)

    print("="*60)
    print("EXPERIMENT 18: NN Two-Class Data (Linear Activation)")
    print("="*60)
    print(f"\nDataset: make_classification (300 samples, 2 features)")
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
    plt.savefig('Exp18_nn_two_class_linear.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Linear activation = straight-line boundary (same as logistic regression)")
    print("2. Network with linear activation is still a linear model")
    print("3. Stacking linear layers doesn't add non-linearity")
    print("4. This is the baseline for comparison with non-linear activations")

    print("\nPlot saved as Exp18_nn_two_class_linear.png")
    print("Exp 18 completed successfully!")
