# ============================================================================
# QUESTION 7: Forward Propagation and Backpropagation Algorithm
# a) Manually compute forward propagation for a small neural network (given
#    weights and inputs) and verify the output using code.
# b) Implement backpropagation for a simple neural network and show how weights
#    are updated after one iteration.
# ============================================================================
# Dataset: Manual Weights and Inputs (No external dataset needed)
# Using predefined weights and inputs for demonstration
# ============================================================================

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

print("="*70)
print("PART A: Forward Propagation - Manual Computation & Code Verification")
print("="*70)

print("\nNeural Network Structure:")
print("  Input Layer: 2 neurons (x1, x2)")
print("  Hidden Layer: 2 neurons (h1, h2)")
print("  Output Layer: 1 neuron (y)")

x1, x2 = 0.5, 0.3
print(f"\nInput Values: x1 = {x1}, x2 = {x2}")

w11, w12 = 0.1, 0.3
w21, w22 = 0.2, 0.4
print(f"\nWeights (Input to Hidden):")
print(f"  w11 = {w11} (x1 to h1)")
print(f"  w12 = {w12} (x1 to h2)")
print(f"  w21 = {w21} (x2 to h1)")
print(f"  w22 = {w22} (x2 to h2)")

b1, b2 = 0.5, 0.6
print(f"\nBiases (Hidden Layer):")
print(f"  b1 = {b1} (h1)")
print(f"  b2 = {b2} (h2)")

v1, v2 = 0.7, 0.8
b3 = 0.9
print(f"\nWeights (Hidden to Output):")
print(f"  v1 = {v1} (h1 to y)")
print(f"  v2 = {v2} (h2 to y)")
print(f"  b3 = {b3} (output bias)")

print("\n" + "-"*70)
print("Manual Forward Propagation Calculation:")
print("-"*70)

z1 = x1 * w11 + x2 * w21 + b1
a1 = sigmoid(z1)
print(f"\nHidden Neuron h1:")
print(f"  z1 = x1*w11 + x2*w21 + b1 = {x1}*{w11} + {x2}*{w21} + {b1} = {z1:.4f}")
print(f"  a1 = sigmoid(z1) = sigmoid({z1:.4f}) = {a1:.4f}")

z2 = x1 * w12 + x2 * w22 + b2
a2 = sigmoid(z2)
print(f"\nHidden Neuron h2:")
print(f"  z2 = x1*w12 + x2*w22 + b2 = {x1}*{w12} + {x2}*{w22} + {b2} = {z2:.4f}")
print(f"  a2 = sigmoid(z2) = sigmoid({z2:.4f}) = {a2:.4f}")

z3 = a1 * v1 + a2 * v2 + b3
a3 = sigmoid(z3)
print(f"\nOutput Neuron y:")
print(f"  z3 = a1*v1 + a2*v2 + b3 = {a1:.4f}*{v1} + {a2:.4f}*{v2} + {b3} = {z3:.4f}")
print(f"  a3 = sigmoid(z3) = sigmoid({z3:.4f}) = {a3:.4f}")

print("\n" + "-"*70)
print("Code Verification:")
print("-"*70)

X = np.array([[x1, x2]])
W1 = np.array([[w11, w12], [w21, w22]])
b1_vec = np.array([[b1, b2]])
W2 = np.array([[v1], [v2]])
b2_vec = np.array([[b3]])

z1_code = X.dot(W1) + b1_vec
a1_code = sigmoid(z1_code)
z2_code = a1_code.dot(W2) + b2_vec
a2_code = sigmoid(z2_code)

print(f"\nCode Output:")
print(f"  Hidden Layer (a1, a2): {a1_code[0]}")
print(f"  Output (a3): {a2_code[0][0]:.4f}")

print(f"\nVerification: Manual = {a3:.4f}, Code = {a2_code[0][0]:.4f}")
print(f"Match: {np.isclose(a3, a2_code[0][0])}")

print("\n" + "="*70)
print("PART B: Backpropagation - Weight Updates After One Iteration")
print("="*70)

target = 1.0
print(f"\nTarget Output: {target}")

print("\n" + "-"*70)
print("Manual Backpropagation Calculation:")
print("-"*70)

output_error = a3 - target
print(f"\nOutput Error: (a3 - target) = {a3:.4f} - {target} = {output_error:.4f}")

d_output = output_error * sigmoid_derivative(a3)
print(f"Output Delta: d_output = error * sigmoid'(z3) = {output_error:.4f} * {sigmoid_derivative(a3):.4f} = {d_output:.4f}")

d_hidden = d_output * W2.T * sigmoid_derivative(a1_code)
print(f"\nHidden Deltas:")
print(f"  d_h1 = d_output * v1 * sigmoid'(z1) = {d_output:.4f} * {v1} * {sigmoid_derivative(a1):.4f} = {d_hidden[0][0]:.4f}")
print(f"  d_h2 = d_output * v2 * sigmoid'(z2) = {d_output:.4f} * {v2} * {sigmoid_derivative(a2):.4f} = {d_hidden[0][1]:.4f}")

learning_rate = 0.5
print(f"\nLearning Rate: {learning_rate}")

v1_new = v1 - learning_rate * d_output * a1
v2_new = v2 - learning_rate * d_output * a2
b3_new = b3 - learning_rate * d_output
print(f"\nUpdated Weights (Hidden to Output):")
print(f"  v1_new = v1 - lr * d_output * a1 = {v1} - {learning_rate} * {d_output:.4f} * {a1:.4f} = {v1_new:.4f}")
print(f"  v2_new = v2 - lr * d_output * a2 = {v2} - {learning_rate} * {d_output:.4f} * {a2:.4f} = {v2_new:.4f}")
print(f"  b3_new = b3 - lr * d_output = {b3} - {learning_rate} * {d_output:.4f} = {b3_new:.4f}")

w11_new = w11 - learning_rate * d_hidden[0][0] * x1
w12_new = w12 - learning_rate * d_hidden[0][1] * x1
w21_new = w21 - learning_rate * d_hidden[0][0] * x2
w22_new = w22 - learning_rate * d_hidden[0][1] * x2
b1_new = b1 - learning_rate * d_hidden[0][0]
b2_new = b2 - learning_rate * d_hidden[0][1]

print(f"\nUpdated Weights (Input to Hidden):")
print(f"  w11_new = w11 - lr * d_h1 * x1 = {w11} - {learning_rate} * {d_hidden[0][0]:.4f} * {x1} = {w11_new:.4f}")
print(f"  w12_new = w12 - lr * d_h2 * x1 = {w12} - {learning_rate} * {d_hidden[0][1]:.4f} * {x1} = {w12_new:.4f}")
print(f"  w21_new = w21 - lr * d_h1 * x2 = {w21} - {learning_rate} * {d_hidden[0][0]:.4f} * {x2} = {w21_new:.4f}")
print(f"  w22_new = w22 - lr * d_h2 * x2 = {w22} - {learning_rate} * {d_hidden[0][1]:.4f} * {x2} = {w22_new:.4f}")
print(f"  b1_new = b1 - lr * d_h1 = {b1} - {learning_rate} * {d_hidden[0][0]:.4f} = {b1_new:.4f}")
print(f"  b2_new = b2 - lr * d_h2 = {b2} - {learning_rate} * {d_hidden[0][1]:.4f} = {b2_new:.4f}")

print("\n" + "-"*70)
print("Code Verification of Backpropagation:")
print("-"*70)

W2_new = W2 - learning_rate * a1_code.T * d_output
b2_new_vec = b2_vec - learning_rate * d_output
W1_new = W1 - learning_rate * X.T * d_hidden
b1_new_vec = b1_vec - learning_rate * d_hidden

print(f"\nCode Updated Weights:")
print(f"  W2 (Hidden to Output):\n    {W2_new.flatten()}")
print(f"  W1 (Input to Hidden):\n    {W1_new}")
print(f"  b2 (Output Bias): {b2_new_vec}")
print(f"  b1 (Hidden Biases): {b1_new_vec}")

print("\n" + "-"*70)
print("Weight Update Summary:")
print("-"*70)
print(f"\n{'Weight':<10} {'Before':<12} {'After (Manual)':<15} {'After (Code)':<15}")
print("-"*52)
print(f"{'w11':<10} {w11:<12.4f} {w11_new:<15.4f} {W1_new[0][0]:<15.4f}")
print(f"{'w12':<10} {w12:<12.4f} {w12_new:<15.4f} {W1_new[0][1]:<15.4f}")
print(f"{'w21':<10} {w21:<12.4f} {w21_new:<15.4f} {W1_new[1][0]:<15.4f}")
print(f"{'w22':<10} {w22:<12.4f} {w22_new:<15.4f} {W1_new[1][1]:<15.4f}")
print(f"{'v1':<10} {v1:<12.4f} {v1_new:<15.4f} {W2_new[0][0]:<15.4f}")
print(f"{'v2':<10} {v2:<12.4f} {v2_new:<15.4f} {W2_new[1][0]:<15.4f}")

print("\nQ7 completed successfully!")
