# ============================================================================
# EXPERIMENT 27: NN Two-Class Data (ReLU, Learning Rate 0.001)
# Objective: Show the effect of a very small learning rate - slow convergence.
# Dataset: make_classification (same as Exp 18/22 for comparison)
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
    print("EXPERIMENT 27: NN Two-Class Data (ReLU, LR=0.001)")
    print("="*60)
    print(f"\nDataset: make_classification (300 samples, 2 features)")
    print(f"Activation: ReLU")
    print(f"Hidden neurons: 4")
    print(f"Learning Rate: 0.001 (very small)")

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation='relu', learning_rate=0.001)
    losses_500, accs_500 = nn.fit(X_train, y_train, epochs=500, verbose=True)

    train_acc_500 = np.mean(nn.predict(X_train) == y_train.ravel())
    test_acc_500 = np.mean(nn.predict(X_test) == y_test.ravel())

    print(f"\nAt 500 epochs:")
    print(f"  Training Accuracy: {train_acc_500:.4f}")
    print(f"  Test Accuracy: {test_acc_500:.4f}")

    nn2 = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation='relu', learning_rate=0.001)
    losses_3000, accs_3000 = nn2.fit(X_train, y_train, epochs=3000, verbose=False)

    train_acc_3000 = np.mean(nn2.predict(X_train) == y_train.ravel())
    test_acc_3000 = np.mean(nn2.predict(X_test) == y_test.ravel())

    print(f"\nAt 3000 epochs:")
    print(f"  Training Accuracy: {train_acc_3000:.4f}")
    print(f"  Test Accuracy: {test_acc_3000:.4f}")

    nn_high = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation='relu', learning_rate=0.01)
    losses_high, accs_high = nn_high.fit(X_train, y_train, epochs=500, verbose=False)
    train_acc_high = np.mean(nn_high.predict(X_train) == y_train.ravel())
    test_acc_high = np.mean(nn_high.predict(X_test) == y_test.ravel())

    print(f"\nComparison with LR=0.01 at 500 epochs:")
    print(f"  Training Accuracy: {train_acc_high:.4f}")
    print(f"  Test Accuracy: {test_acc_high:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(losses_500, 'b-', linewidth=2, label='LR=0.001 (500 epochs)')
    axes[0].plot(losses_high, 'r-', linewidth=2, label='LR=0.01 (500 epochs)')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curve Comparison', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(accs_500, 'b-', linewidth=2, label='LR=0.001 (500 epochs)')
    axes[1].plot(accs_high, 'r-', linewidth=2, label='LR=0.01 (500 epochs)')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Curve Comparison', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plot_decision_boundary(nn, X_test, y_test, f'Decision Boundary (LR=0.001, 500ep)\nAccuracy: {test_acc_500:.2%}', scaler, axes[2])

    plt.tight_layout()
    plt.savefig('Exp27_nn_low_learning_rate.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. LR=0.001 converges much slower than LR=0.01")
    print("2. At equal epoch budget (500), LR=0.001 is under-converged")
    print("3. Given enough epochs (3000), LR=0.001 eventually catches up")
    print("4. Small LR = slow but stable convergence")

    print("\nPlot saved as Exp27_nn_low_learning_rate.png")
    print("Exp 27 completed successfully!")
