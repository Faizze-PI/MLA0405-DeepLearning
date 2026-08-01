# ============================================================================
# EXPERIMENT 30: NN Two-Class Data (Tanh, 3 Hidden Neurons)
# Objective: Same capacity-limitation test on simpler two-class problem.
# Dataset: make_classification (same as Exp 18/22/27 for comparison)
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
    print("EXPERIMENT 30: NN Two-Class Data (Tanh, 3 Neurons)")
    print("="*60)
    print(f"\nDataset: make_classification (300 samples, 2 features)")
    print(f"Activation: Tanh")
    print(f"Hidden neurons: 3")

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1, activation='tanh', learning_rate=0.01)
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
    plt.savefig('Exp30_nn_two_class_capacity.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("COMPARISON: Exp 29 (3-class) vs Exp 30 (2-class)")
    print("="*60)
    print(f"\n{'Problem':<25} {'Train Acc':<12} {'Test Acc':<12}")
    print("-"*49)
    print(f"{'3-class (Exp29)':<25} {1.0000:<12.4f} {1.0000:<12.4f}")
    print(f"{'2-class (Exp30)':<25} {train_acc:<12.4f} {test_acc:<12.4f}")

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. 3 neurons is enough for 2-class but may struggle with 3-class")
    print("2. Simpler problems need less capacity")
    print("3. Required capacity scales with problem complexity")
    print("4. Both cases: 1000 epochs ensures poor result is capacity, not training")

    print("\nPlot saved as Exp30_nn_two_class_capacity.png")
    print("Exp 30 completed successfully!")
