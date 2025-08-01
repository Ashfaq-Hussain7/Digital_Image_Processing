import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab import files

# ----- Load and Prepare Image -----
def load_image(path, gray=True):
    img = cv2.imread(path)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# ----- Linear Filter Functions -----
def apply_mean_filter(img, ksize=5):
    return cv2.blur(img, (ksize, ksize))

def apply_gaussian_filter(img, ksize=5, sigma=1):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def apply_laplacian_filter(img):
    return cv2.Laplacian(img, cv2.CV_64F)

def apply_sobel_filter(img, direction='x'):
    if direction == 'x':
        return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    else:
        return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

def apply_box_filter(img, ksize=5):
    return cv2.boxFilter(img, -1, (ksize, ksize))

def apply_custom_filter(img, kernel):
    return cv2.filter2D(img, -1, kernel)

# ----- Display Helper -----
def show_images(images, titles, cmap='gray'):
    n = len(images)
    plt.figure(figsize=(16, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ----- Main Program -----
if __name__ == "__main__":
    # Upload image
    print("📤 Please upload an image...")
    uploaded = files.upload()
    img_path = list(uploaded.keys())[0]

    # Load uploaded image
    img = load_image(img_path)

    # Apply filters
    mean_img = apply_mean_filter(img)
    gaussian_img = apply_gaussian_filter(img)
    laplacian_img = apply_laplacian_filter(img)
    sobel_x_img = apply_sobel_filter(img, 'x')
    sobel_y_img = apply_sobel_filter(img, 'y')
    box_img = apply_box_filter(img)

    # Custom kernel (Sharpening)
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened_img = apply_custom_filter(img, sharpen_kernel)

    # Display all images
    show_images(
        [img, mean_img, gaussian_img, laplacian_img, sobel_x_img, sobel_y_img, box_img, sharpened_img],
        ['Original', 'Mean Filter', 'Gaussian', 'Laplacian', 'Sobel X', 'Sobel Y', 'Box Filter', 'Sharpened']
    )
