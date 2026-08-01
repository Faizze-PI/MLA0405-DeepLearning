# ============================================================================
# EXPERIMENT 32: NN Two Circular Data (Tanh Activation, Two Factor Values)
# Objective: Test tanh's boundary-fitting as classes get closer together.
# Dataset: make_circles with factor=0.3 and factor=0.7
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
    X_03, y_03 = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
    X_train_03, X_test_03, y_train_03, y_test_03, scaler_03 = prepare_binary_data(X_03, y_03)

    X_07, y_07 = make_circles(n_samples=300, noise=0.1, factor=0.7, random_state=42)
    X_train_07, X_test_07, y_train_07, y_test_07, scaler_07 = prepare_binary_data(X_07, y_07)

    print("="*60)
    print("EXPERIMENT 32: NN Circular Data (Tanh, Two Factors)")
    print("="*60)
    print(f"\nDataset: make_circles with different factor values")
    print(f"Activation: Tanh")
    print(f"Hidden neurons: 8")

    nn_03 = SimpleNeuralNetwork(input_size=2, hidden_size=8, output_size=1, activation='tanh', learning_rate=0.01)
    losses_03, accs_03 = nn_03.fit(X_train_03, y_train_03, epochs=500, verbose=True)

    train_acc_03 = np.mean(nn_03.predict(X_train_03) == y_train_03.ravel())
    test_acc_03 = np.mean(nn_03.predict(X_test_03) == y_test_03.ravel())

    print(f"\nFactor=0.3:")
    print(f"  Training Accuracy: {train_acc_03:.4f}")
    print(f"  Test Accuracy: {test_acc_03:.4f}")

    nn_07 = SimpleNeuralNetwork(input_size=2, hidden_size=8, output_size=1, activation='tanh', learning_rate=0.01)
    losses_07, accs_07 = nn_07.fit(X_train_07, y_train_07, epochs=500, verbose=False)

    train_acc_07 = np.mean(nn_07.predict(X_train_07) == y_train_07.ravel())
    test_acc_07 = np.mean(nn_07.predict(X_test_07) == y_test_07.ravel())

    print(f"\nFactor=0.7:")
    print(f"  Training Accuracy: {train_acc_07:.4f}")
    print(f"  Test Accuracy: {test_acc_07:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_decision_boundary(nn_03, X_train_03, y_train_03, f'Factor=0.3 (Train)\nAccuracy: {train_acc_03:.2%}', scaler_03, axes[0, 0])
    plot_decision_boundary(nn_03, X_test_03, y_test_03, f'Factor=0.3 (Test)\nAccuracy: {test_acc_03:.2%}', scaler_03, axes[0, 1])
    axes[0, 2].plot(losses_03, 'b-', linewidth=2, label='Factor=0.3')
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Loss', fontsize=12)
    axes[0, 2].set_title('Loss Curve (Factor=0.3)', fontsize=14)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    plot_decision_boundary(nn_07, X_train_07, y_train_07, f'Factor=0.7 (Train)\nAccuracy: {train_acc_07:.2%}', scaler_07, axes[1, 0])
    plot_decision_boundary(nn_07, X_test_07, y_test_07, f'Factor=0.7 (Test)\nAccuracy: {test_acc_07:.2%}', scaler_07, axes[1, 1])
    axes[1, 2].plot(losses_07, 'r-', linewidth=2, label='Factor=0.7')
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('Loss', fontsize=12)
    axes[1, 2].set_title('Loss Curve (Factor=0.7)', fontsize=14)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Exp32_nn_circular_two_factors.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Factor=0.3: Classes far apart -> easier to separate")
    print("2. Factor=0.7: Classes closer together -> harder to separate")
    print("3. Accuracy drops as factor increases (boundary harder)")
    print("4. Tanh's smooth boundaries work well for both cases")

    print("\nPlot saved as Exp32_nn_circular_two_factors.png")
    print("Exp 32 completed successfully!")
