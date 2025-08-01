import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload an image
print("Upload an image:")
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img = cv2.imread(img_path)

# Resize if needed
# img = cv2.resize(img, (512, 512))  # Optional for large images


# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Enhancement operations
# a. Brightness enhancement
bright = cv2.convertScaleAbs(img, alpha=1, beta=50)

# b. Contrast enhancement
contrast = cv2.convertScaleAbs(img, alpha=2, beta=0)

# c. Complement of image
complement = 255 - img

# d. Bi-level contrast
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# e. Brightness slicing
brightness_sliced = np.where((gray > 100) & (gray < 180), 255, gray).astype(np.uint8)


# Add noise to the image
def add_gaussian_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    return noisy

# Try salt & pepper noise:
# def add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02):
#     noisy = img.copy()
#     h, w, c = img.shape
#     num_salt = np.ceil(salt_prob * h * w)
#     num_pepper = np.ceil(pepper_prob * h * w)
#     coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
#     noisy[coords[0], coords[1]] = 255
#     coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
#     noisy[coords[0], coords[1]] = 0
#     return noisy

img = add_gaussian_noise(img)  # ✅ Now working with noisy image


# f. Low-pass filtering
low_pass = cv2.GaussianBlur(img, (11, 11), 0)

# g. High-pass filtering
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
high_pass = cv2.filter2D(img, -1, kernel)

# Display results
titles = ['Noisy Image', 'Brightness Enhanced', 'Contrast Enhanced', 'Complement',
          'Bi-level Contrast', 'Brightness Sliced', 'Low-pass Filtered', 'High-pass Filtered']
images = [img, bright, contrast, complement, binary, brightness_sliced, low_pass, high_pass]

for title, image in zip(titles, images):
    print(f"\n✅ {title}")
    if len(image.shape) == 2:
        cv2_imshow(image)
    else:
        cv2_imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
