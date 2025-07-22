import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload Image
print("Please upload your image (e.g., input.jpg):")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Read the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image.")
else:
    print("✅ Image loaded. Displaying it below:")
    cv2_imshow(image)

    # Convert to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot histograms
    colors = ('r', 'g', 'b')  # RGB channels
    plt.figure(figsize=(10, 5))
    plt.title("Histogram for RGB Channels")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.grid(True)
    plt.show()


