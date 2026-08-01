# ============================================================================
# EXPERIMENT 23: Neural Network Analysis for Spiral Data (Sigmoid Activation)
# Objective: Test sigmoid activation on the hardest toy dataset - the two-arm spiral.
# Dataset: Custom spiral generator
# Source: Generated in-code, no external dataset needed
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nn_helpers import SimpleNeuralNetwork, prepare_binary_data, plot_decision_boundary, make_spiral

np.random.seed(42)


if __name__ == "__main__":
    X, y = make_spiral(n_points=150, noise=0.5)
    X_train, X_test, y_train, y_test, scaler = prepare_binary_data(X, y)

    print("="*60)
    print("EXPERIMENT 23: NN Spiral Data (Sigmoid Activation)")
    print("="*60)
    print(f"\nDataset: Custom spiral (300 samples, 2 features)")
    print(f"Activation: Sigmoid")
    print(f"Hidden layers: 2 layers of 16 neurons each")

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=16, output_size=1, activation='sigmoid', learning_rate=0.03)
    losses, accuracies = nn.fit(X_train, y_train, epochs=2000, verbose=True)

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
    plt.savefig('Exp23_nn_spiral_sigmoid.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Spiral is genuinely hard - sigmoid struggles with shallow networks")
    print("2. Sigmoid saturates (vanishing gradient problem)")
    print("3. Needs more epochs than ReLU to converge")
    print("4. May only achieve moderate accuracy (~70-85%)")
    print("5. This is expected - spirals need deeper networks or more neurons")

    print("\nPlot saved as Exp23_nn_spiral_sigmoid.png")
    print("Exp 23 completed successfully!")
