import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab import files

# Load grayscale image
def load_image():
    print("📤 Upload an image:")
    uploaded = files.upload()
    img_path = list(uploaded.keys())[0]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img

# Load kernel from text file 
def load_kernel():
    print("📤 Upload a kernel (ASCII text file):")
    uploaded = files.upload()
    kernel_path = list(uploaded.keys())[0]
    with open(kernel_path, 'r') as f:
        lines = f.readlines()
    kernel = [list(map(float, line.strip().split())) for line in lines]
    return np.array(kernel, dtype=np.float32)

# Convolution
def apply_convolution(img, kernel):
    return cv2.filter2D(img, -1, kernel)

# Correlation (via flipping kernel)
def apply_correlation(img, kernel):
    flipped_kernel = cv2.flip(kernel, -1)  # flip both axes
    return cv2.filter2D(img, -1, flipped_kernel)

# Show Images
def show_images(img_list, title_list):
    plt.figure(figsize=(12, 4))
    for i in range(len(img_list)):
        plt.subplot(1, len(img_list), i+1)
        plt.imshow(img_list[i], cmap='gray')
        plt.title(title_list[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main
img = load_image()
kernel = load_kernel()

convolved_img = apply_convolution(img, kernel)
correlated_img = apply_correlation(img, kernel)

# Show results
show_images([img, convolved_img, correlated_img],
            ['Original Image', 'Convolution Result', 'Correlation Result'])
