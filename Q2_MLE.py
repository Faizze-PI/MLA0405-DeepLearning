# ============================================================================
# QUESTION 2: Maximum Likelihood Estimation (MLE)
# Generate synthetic data from a normal distribution and use MLE to estimate
# the mean and variance. Compare estimated values with actual parameters.
# ============================================================================
# Dataset: Synthetic Data (Normal Distribution - No external dataset needed)
# Generated using numpy.random.normal()
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

true_mean = 5.0
true_variance = 2.0
true_std = np.sqrt(true_variance)
sample_size = 1000

data = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)

def mle_estimates(samples):
    n = len(samples)
    mean_estimate = np.sum(samples) / n
    variance_estimate = np.sum((samples - mean_estimate) ** 2) / n
    return mean_estimate, variance_estimate

estimated_mean, estimated_variance = mle_estimates(data)

print("=" * 50)
print("Maximum Likelihood Estimation (MLE) Results")
print("=" * 50)
print(f"\nTrue Parameters:")
print(f"  Mean (mu): {true_mean}")
print(f"  Variance (sigma2): {true_variance}")
print(f"  Std Dev (sigma): {true_std:.4f}")

print(f"\nMLE Estimated Parameters:")
print(f"  Mean (mu_hat): {estimated_mean:.4f}")
print(f"  Variance (sigma2_hat): {estimated_variance:.4f}")
print(f"  Std Dev (sigma_hat): {np.sqrt(estimated_variance):.4f}")

print(f"\nEstimation Errors:")
print(f"  Mean Error: {abs(estimated_mean - true_mean):.4f}")
print(f"  Variance Error: {abs(estimated_variance - true_variance):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', label='Sample Data')
x = np.linspace(min(data), max(data), 1000)
true_pdf = (1 / (true_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - true_mean) / true_std) ** 2)
estimated_pdf = (1 / (np.sqrt(estimated_variance) * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - estimated_mean) / np.sqrt(estimated_variance)) ** 2)

axes[0].plot(x, true_pdf, 'r-', linewidth=2, label=f'True (mu={true_mean}, sigma2={true_variance})')
axes[0].plot(x, estimated_pdf, 'g--', linewidth=2, label=f'MLE (mu_hat={estimated_mean:.2f}, sigma2_hat={estimated_variance:.2f})')
axes[0].set_xlabel('Value', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Normal Distribution: True vs MLE Estimated', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

sample_sizes = [10, 50, 100, 500, 1000, 5000]
mean_errors = []
var_errors = []

for n in sample_sizes:
    samples = np.random.normal(true_mean, true_std, n)
    m_mean, m_var = mle_estimates(samples)
    mean_errors.append(abs(m_mean - true_mean))
    var_errors.append(abs(m_var - true_variance))

axes[1].plot(sample_sizes, mean_errors, 'bo-', linewidth=2, markersize=8, label='Mean Error')
axes[1].plot(sample_sizes, var_errors, 'rs-', linewidth=2, markersize=8, label='Variance Error')
axes[1].set_xlabel('Sample Size', fontsize=12)
axes[1].set_ylabel('Absolute Error', fontsize=12)
axes[1].set_title('MLE Convergence with Sample Size', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xscale('log')

plt.tight_layout()
plt.savefig('Q2_MLE_results.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 50)
print("Q2 completed successfully!")
print("Plot saved as Q2_MLE_results.png")
