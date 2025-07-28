import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload Image
print("Upload an image for local histogram equalization:")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]


# Read and convert to grayscale
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply Local Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # You can adjust these parameters
local_equalized = clahe.apply(gray)


# Display Original and Local Equalized Images
print("\n✅ Displaying original grayscale and locally equalized image:")
cv2_imshow(gray)
cv2_imshow(local_equalized)


# Plot Histograms
def plot_histogram(img, title, color='black'):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_histogram(gray, 'Original Grayscale Histogram')

plt.subplot(1, 2, 2)
plot_histogram(local_equalized, 'CLAHE (Local Equalized) Histogram')

plt.tight_layout()
plt.show()