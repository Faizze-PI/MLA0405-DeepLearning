# ============================================================================
# QUESTION 5: Artificial Neurons
# Implement a single artificial neuron using different activation functions
# (Sigmoid, ReLU) and compare outputs for the same input values.
# ============================================================================
# Dataset: Manual Input Values (No external dataset needed)
# Using predefined input vectors for demonstration
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

inputs = np.array([
    [0.5, 0.3],
    [-0.2, 0.8],
    [1.0, -0.5],
    [-0.7, -0.4],
    [0.0, 0.0],
    [2.0, 1.5],
    [-1.5, 2.0],
    [0.8, -1.2]
])

weights = np.array([0.6, -0.4])
bias = 0.1

def artificial_neuron(inputs, weights, bias, activation_func):
    z = np.dot(inputs, weights) + bias
    output = activation_func(z)
    return z, output

z_linear = inputs.dot(weights) + bias

z_sigmoid, output_sigmoid = artificial_neuron(inputs, weights, bias, sigmoid)
z_relu, output_relu = artificial_neuron(inputs, weights, bias, relu)
z_tanh, output_tanh = artificial_neuron(inputs, weights, bias, tanh)
z_leaky_relu, output_leaky_relu = artificial_neuron(inputs, weights, bias, leaky_relu)

print("="*70)
print("Artificial Neuron Implementation with Different Activation Functions")
print("="*70)
print(f"\nWeights: {weights}")
print(f"Bias: {bias}")
print(f"\nInput Samples:")
for i, inp in enumerate(inputs):
    print(f"  Sample {i+1}: {inp}")

print(f"\n{'='*70}")
print("Results Comparison")
print("="*70)
print(f"\n{'Input':<20} {'Linear':<12} {'Sigmoid':<12} {'ReLU':<12} {'Tanh':<12} {'LeakyReLU':<12}")
print("-"*80)
for i in range(len(inputs)):
    print(f"{str(inputs[i]):<20} {z_linear[i]:<12.4f} {output_sigmoid[i]:<12.4f} {output_relu[i]:<12.4f} {output_tanh[i]:<12.4f} {output_leaky_relu[i]:<12.4f}")

print(f"\n{'='*70}")
print("Activation Function Characteristics")
print("="*70)
print("\n1. Sigmoid: Output range (0, 1) - Good for probability outputs")
print("2. ReLU: Output range [0, inf) - Most popular, avoids vanishing gradient")
print("3. Tanh: Output range (-1, 1) - Zero-centered output")
print("4. Leaky ReLU: Output range (-inf, inf) - Addresses dying ReLU problem")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

z_range = np.linspace(-5, 5, 1000)

axes[0, 0].plot(z_range, z_range, 'b-', linewidth=2)
axes[0, 0].set_title('Linear (No Activation)', fontsize=12)
axes[0, 0].set_xlabel('z')
axes[0, 0].set_ylabel('f(z)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)

axes[0, 1].plot(z_range, sigmoid(z_range), 'r-', linewidth=2)
axes[0, 1].set_title('Sigmoid', fontsize=12)
axes[0, 1].set_xlabel('z')
axes[0, 1].set_ylabel('f(z)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linewidth=0.5)

axes[0, 2].plot(z_range, relu(z_range), 'g-', linewidth=2)
axes[0, 2].set_title('ReLU', fontsize=12)
axes[0, 2].set_xlabel('z')
axes[0, 2].set_ylabel('f(z)')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axhline(y=0, color='k', linewidth=0.5)
axes[0, 2].axvline(x=0, color='k', linewidth=0.5)

axes[0, 3].plot(z_range, tanh(z_range), 'm-', linewidth=2)
axes[0, 3].set_title('Tanh', fontsize=12)
axes[0, 3].set_xlabel('z')
axes[0, 3].set_ylabel('f(z)')
axes[0, 3].grid(True, alpha=0.3)
axes[0, 3].axhline(y=0, color='k', linewidth=0.5)
axes[0, 3].axvline(x=0, color='k', linewidth=0.5)

x_labels = [f'S{i+1}' for i in range(len(inputs))]
x_pos = np.arange(len(inputs))

axes[1, 0].bar(x_pos, z_linear, color='steelblue', edgecolor='black')
axes[1, 0].set_title('Linear Output', fontsize=12)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(x_labels)
axes[1, 0].grid(axis='y', alpha=0.3)

axes[1, 1].bar(x_pos, output_sigmoid, color='coral', edgecolor='black')
axes[1, 1].set_title('Sigmoid Output', fontsize=12)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(x_labels)
axes[1, 1].grid(axis='y', alpha=0.3)

axes[1, 2].bar(x_pos, output_relu, color='forestgreen', edgecolor='black')
axes[1, 2].set_title('ReLU Output', fontsize=12)
axes[1, 2].set_xticks(x_pos)
axes[1, 2].set_xticklabels(x_labels)
axes[1, 2].grid(axis='y', alpha=0.3)

axes[1, 3].bar(x_pos, output_leaky_relu, color='purple', edgecolor='black')
axes[1, 3].set_title('Leaky ReLU Output', fontsize=12)
axes[1, 3].set_xticks(x_pos)
axes[1, 3].set_xticklabels(x_labels)
axes[1, 3].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('Q5_artificial_neurons.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nQ5 completed successfully!")
print("Plot saved as Q5_artificial_neurons.png")
