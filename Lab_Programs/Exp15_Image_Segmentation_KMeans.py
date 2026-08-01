# ============================================================================
# EXPERIMENT 15: Image Segmentation using K-Means Clustering
# Objective: Treat image segmentation as an unsupervised clustering problem
#            on pixel color values.
# Dataset: Fruit Image (from Fruits_Plant_Leaf dataset)
# Source: https://www.kaggle.com/datasets/pathanajahar/fruits-plant-leaf
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import os


if __name__ == "__main__":
    def load_and_resize_image(image_path, max_size=400):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        return img

    def segment_image_kmeans(img, n_clusters):
        h, w, c = img.shape
        pixels = img.reshape(-1, 3).astype(np.float32)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)

        segmented = kmeans.cluster_centers_[labels].reshape(h, w, 3).astype(np.uint8)

        return segmented, labels, kmeans

    image_dir = "Fruits_Plant_Leaf"
    image_subdir = "Apple"
    image_files = [f for f in os.listdir(os.path.join(image_dir, image_subdir)) if f.endswith('.jpg')]

    if not image_files:
        raise FileNotFoundError("No images found in Fruits_Plant_Leaf/Apple/")

    image_path = os.path.join(image_dir, image_subdir, image_files[0])
    img = load_and_resize_image(image_path)

    print("="*60)
    print("EXPERIMENT 15: Image Segmentation using K-Means")
    print("="*60)
    print(f"\nImage: {image_path}")
    print(f"Shape: {img.shape}")
    print(f"File size: {os.path.getsize(image_path) / 1024:.1f} KB")

    k_values = [2, 4, 6, 8]
    segmented_images = {}

    for k in k_values:
        segmented, labels, model = segment_image_kmeans(img, k)
        segmented_images[k] = segmented
        print(f"K={k}: {len(np.unique(labels))} unique colors")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')

    for idx, k in enumerate(k_values):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        axes[row, col].imshow(segmented_images[k])
        axes[row, col].set_title(f'K = {k}', fontsize=14)
        axes[row, col].axis('off')

    for idx, k in enumerate(k_values):
        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3).astype(np.float32)
        _, labels, _ = segment_image_kmeans(img, k)

        color_counts = np.bincount(labels)
        total_pixels = len(labels)

        row = 1
        col = idx % 3
        if idx < 3:
            percentages = (color_counts / total_pixels * 100)
            colors_rgb = segmented_images[k].reshape(-1, 3)[::len(labels)//10]
            axes[1, idx].pie(color_counts, labels=[f'C{i}' for i in range(k)], autopct='%1.1f%%')
            axes[1, idx].set_title(f'Color Distribution (K={k})', fontsize=14)

    plt.tight_layout()
    plt.savefig('Exp15_kmeans_segmentation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. K=2: Coarse segmentation (foreground/background)")
    print("2. K=4-6: Captures main color regions")
    print("3. K=8: Fine-grained, may over-segment")
    print("4. Higher K = more detail but slower computation")
    print("5. K-Means groups similar colors regardless of spatial location")

    print("\nPlot saved as Exp15_kmeans_segmentation.png")
    print("Exp 15 completed successfully!")
