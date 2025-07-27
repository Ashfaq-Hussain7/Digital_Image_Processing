import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload Image
print("Upload an image for histogram equalization:")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Read and convert to grayscale
image = cv2.imread(image_path)
if image is None:
    print("Error: Cannot read image.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply Histogram Equalization
equalized = cv2.equalizeHist(gray)


# Show Original and Equalized Images
print("\n✅ Displaying original grayscale and equalized image:")
cv2_imshow(gray)
cv2_imshow(equalized)

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
plot_histogram(equalized, 'Equalized Histogram')

plt.tight_layout()
plt.show()