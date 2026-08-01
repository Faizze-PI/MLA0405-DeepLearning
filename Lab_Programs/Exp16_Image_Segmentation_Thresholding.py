# ============================================================================
# EXPERIMENT 16: Image Segmentation using Thresholding & Morphological Ops
# Objective: Perform classical (non-clustering) segmentation using intensity
#            thresholds, then clean up the binary mask.
# Dataset: Same fruit image as Exp 15
# Source: Same as Exp 15
# ============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os


if __name__ == "__main__":
    def load_and_resize_image(image_path, max_size=400):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

        return img_rgb, img

    image_dir = "Fruits_Plant_Leaf"
    image_subdir = "Apple"
    image_files = [f for f in os.listdir(os.path.join(image_dir, image_subdir)) if f.endswith('.jpg')]

    if not image_files:
        raise FileNotFoundError("No images found in Fruits_Plant_Leaf/Apple/")

    image_path = os.path.join(image_dir, image_subdir, image_files[0])
    img_rgb, img_bgr = load_and_resize_image(image_path)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    print("="*60)
    print("EXPERIMENT 16: Image Segmentation using Thresholding")
    print("="*60)
    print(f"\nImage: {image_path}")
    print(f"Shape: {img_rgb.shape}")
    print(f"Otsu Threshold Value: {ret}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale', fontsize=14)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(blurred, cmap='gray')
    axes[0, 2].set_title('Gaussian Blurred', fontsize=14)
    axes[0, 2].axis('off')

    axes[1, 0].imshow(otsu_thresh, cmap='gray')
    axes[1, 0].set_title(f"Otsu Threshold (t={ret:.0f})", fontsize=14)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(adaptive_thresh, cmap='gray')
    axes[1, 1].set_title('Adaptive Threshold', fontsize=14)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(closing, cmap='gray')
    axes[1, 2].set_title('After Morphological Ops', fontsize=14)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('Exp16_thresholding_segmentation.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    axes2[0].imshow(otsu_thresh, cmap='gray')
    axes2[0].set_title('Otsu Threshold', fontsize=14)
    axes2[0].axis('off')

    axes2[1].imshow(opening, cmap='gray')
    axes2[1].set_title('After Opening (Remove Noise)', fontsize=14)
    axes2[1].axis('off')

    axes2[2].imshow(closing, cmap='gray')
    axes2[2].set_title('After Closing (Fill Holes)', fontsize=14)
    axes2[2].axis('off')

    plt.tight_layout()
    plt.savefig('Exp16_morphological_ops.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*60)
    print("KEY OBSERVATIONS")
    print("="*60)
    print("\n1. Otsu: Automatic threshold selection (bimodal histogram)")
    print("2. Adaptive: Better for uneven lighting conditions")
    print("3. Opening: Removes small noise specks (erosion -> dilation)")
    print("4. Closing: Fills small holes (dilation -> erosion)")
    print("5. Pre-processing (blurring) reduces noise before thresholding")

    print("\nPlots saved as Exp16_thresholding_segmentation.png and Exp16_morphological_ops.png")
    print("Exp 16 completed successfully!")
