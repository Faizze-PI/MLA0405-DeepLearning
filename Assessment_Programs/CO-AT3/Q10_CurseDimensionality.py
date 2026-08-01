# ============================================================================
# QUESTION 10: Curse of Dimensionality
# Create datasets with increasing dimensions and analyze how distance between
# points and model accuracy change with dimensionality.
# ============================================================================
# Dataset: Synthetic High-Dimensional Data (Generated using numpy)
# No external dataset needed - using random data generation
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform

np.random.seed(42)

def generate_high_dim_data(n_samples=200, n_features=2, n_classes=2):
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

dimensions = [2, 5, 10, 20, 50, 100]
n_samples = 200

print("="*70)
print("Curse of Dimensionality Analysis")
print("="*70)

distance_stats = {}
accuracies = {}

for dim in dimensions:
    print(f"\nDimension: {dim}")

    X, y = generate_high_dim_data(n_samples=n_samples, n_features=dim)

    distances = squareform(pdist(X, metric='euclidean'))
    np.fill_diagonal(distances, np.inf)

    min_distances = np.min(distances, axis=1)
    max_distances = np.max(distances, axis=1)
    mean_distances = np.mean(distances, axis=1)

    distance_ratio = max_distances / min_distances

    distance_stats[dim] = {
        'min_mean': np.mean(min_distances),
        'max_mean': np.mean(max_distances),
        'ratio_mean': np.mean(distance_ratio),
        'std_min': np.std(min_distances),
        'std_max': np.std(max_distances)
    }

    print(f"  Mean Min Distance: {distance_stats[dim]['min_mean']:.4f}")
    print(f"  Mean Max Distance: {distance_stats[dim]['max_mean']:.4f}")
    print(f"  Mean Distance Ratio (max/min): {distance_stats[dim]['ratio_mean']:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred) * 100
    accuracies[dim] = accuracy
    print(f"  KNN Accuracy: {accuracy:.2f}%")

print("\n" + "="*70)
print("Distance Analysis Summary")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

dim_list = list(distance_stats.keys())
min_dists = [distance_stats[d]['min_mean'] for d in dim_list]
max_dists = [distance_stats[d]['max_mean'] for d in dim_list]
ratios = [distance_stats[d]['ratio_mean'] for d in dim_list]

axes[0, 0].plot(dim_list, min_dists, 'bo-', linewidth=2, markersize=8, label='Min Distance')
axes[0, 0].plot(dim_list, max_dists, 'ro-', linewidth=2, markersize=8, label='Max Distance')
axes[0, 0].set_xlabel('Dimensions', fontsize=12)
axes[0, 0].set_ylabel('Distance', fontsize=12)
axes[0, 0].set_title('Distance vs Dimensions', fontsize=14)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(dim_list, ratios, 'go-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Dimensions', fontsize=12)
axes[0, 1].set_ylabel('Ratio (Max/Min)', fontsize=12)
axes[0, 1].set_title('Distance Ratio vs Dimensions', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(dim_list, list(accuracies.values()), 'mo-', linewidth=2, markersize=8)
axes[0, 2].set_xlabel('Dimensions', fontsize=12)
axes[0, 2].set_ylabel('Accuracy (%)', fontsize=12)
axes[0, 2].set_title('KNN Accuracy vs Dimensions', fontsize=14)
axes[0, 2].grid(True, alpha=0.3)

np.random.seed(42)
sample_data_2d = np.random.randn(100, 2)
distances_2d = squareform(pdist(sample_data_2d, metric='euclidean'))
axes[1, 0].hist(distances_2d[np.triu_indices(100, k=1)], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Distance', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Distance Distribution (2D)', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

np.random.seed(42)
sample_data_100d = np.random.randn(100, 100)
distances_100d = squareform(pdist(sample_data_100d, metric='euclidean'))
axes[1, 1].hist(distances_100d[np.triu_indices(100, k=1)], bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Distance', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Distance Distribution (100D)', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)

n_points = [10, 50, 100, 500, 1000]
dim_for_coverage = []
for n in n_points:
    dim = int(np.log(n) * 10)
    dim_for_coverage.append(dim)

axes[1, 2].plot(n_points, dim_for_coverage, 'ko-', linewidth=2, markersize=8)
axes[1, 2].set_xlabel('Number of Points', fontsize=12)
axes[1, 2].set_ylabel('Dimensions Needed', fontsize=12)
axes[1, 2].set_title('Points vs Dimensions for Coverage', fontsize=14)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Q10_curse_of_dimensionality.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("Key Observations")
print("="*70)
print("\n1. Distance Concentration:")
print("   - As dimensions increase, distances between points become more similar")
print("   - The ratio of max/min distance approaches 1 in high dimensions")
print("   - This makes distance-based algorithms less effective")

print("\n2. Model Performance:")
print("   - KNN accuracy generally decreases with increasing dimensions")
print("   - The 'curse' makes it harder to find meaningful neighbors")
print("   - More data is needed to maintain performance in high dimensions")

print("\n3. Data Sparsity:")
print("   - High-dimensional space is mostly empty")
print("   - Points become increasingly isolated")
print("   - Volume of space grows exponentially with dimensions")

print("\n4. Practical Implications:")
print("   - Feature selection/reduction is crucial")
print("   - Dimensionality reduction techniques (PCA, t-SNE) are important")
print("   - More training data needed for high-dimensional problems")

print("\nQ10 completed successfully!")
print("Plot saved as Q10_curse_of_dimensionality.png")
