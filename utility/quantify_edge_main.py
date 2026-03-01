import cv2
import numpy as np

n = "4"
img_a = cv2.imread(f"output/conv_feature_map/spray/{n}-A.png", 0) #  cv2.imread(f"output/conv_feature_map/wood/{n}-A.png", 0) # 
img_b = cv2.imread(f"output/conv_feature_map/spray/{n}-B.png", 0) # cv2.imread(f"output/conv_feature_map/wood/{n}-B.png", 0) #  

# Compute gradient magnitude again for analysis
gradient_a = np.sqrt(cv2.Sobel(img_a, cv2.CV_64F, 1, 0, ksize=3)**2 + cv2.Sobel(img_a, cv2.CV_64F, 0, 1, ksize=3)**2)
gradient_b = np.sqrt(cv2.Sobel(img_b, cv2.CV_64F, 1, 0, ksize=3)**2 + cv2.Sobel(img_b, cv2.CV_64F, 0, 1, ksize=3)**2)

# Flatten and sort gradient magnitudes (non-zero only)
nonzero_grad_a = gradient_a[gradient_a > 0]
nonzero_grad_b = gradient_b[gradient_b > 0]

# Basic gradient magnitude statistics
gradient_stats = {
    "Image A": {
        "mean": np.mean(nonzero_grad_a),
        "std": np.std(nonzero_grad_a),
        "max": np.max(nonzero_grad_a)
    },
    "Image B": {
        "mean": np.mean(nonzero_grad_b),
        "std": np.std(nonzero_grad_b),
        "max": np.max(nonzero_grad_b)
    }
}

print(f"""--- Edge Thickness Quantification ---
- Original Image Edge Thickness (pixels): {gradient_stats['Image A']}
- Blurred Image Edge Thickness (pixels): {gradient_stats['Image B']}""")
print()
print(f"""--- Increase in Edge Thickness ---
Increase in mean thickness: {gradient_stats['Image B']['mean'] - gradient_stats['Image A']['mean']}
Increase in std thickness: {gradient_stats['Image B']['std'] - gradient_stats['Image A']['std']}
Increase in max thickness: {gradient_stats['Image B']['max'] - gradient_stats['Image A']['max']}""")
