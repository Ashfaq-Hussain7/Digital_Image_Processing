import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload two images
print("Upload two images:")
uploaded = files.upload()

# Load and resize both images
img1 = cv2.imread(list(uploaded.keys())[0])
img2 = cv2.imread(list(uploaded.keys())[1])
img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))

# Basic operations
added = cv2.add(img1, img2)
subtracted = cv2.subtract(img1, img2)

# Multiply and normalize
multiplied = cv2.multiply(img1.astype(np.float32), img2.astype(np.float32))
multiplied = cv2.normalize(multiplied, None, 0, 255, cv2.NORM_MINMAX)
multiplied = multiplied.astype(np.uint8)

# Division with safe handling
img2_safe = img2.astype(np.float32)
img2_safe[img2_safe == 0] = 1
divided = cv2.divide(img1.astype(np.float32), img2_safe)
divided = cv2.normalize(divided, None, 0, 255, cv2.NORM_MINMAX)
divided = divided.astype(np.uint8)

# Show results
print("\n✅ Added Image:")
cv2_imshow(added)

print("\n✅ Subtracted Image:")
cv2_imshow(subtracted)

print("\n✅ Multiplied Image (Normalized):")
cv2_imshow(multiplied)

print("\n✅ Divided Image (Normalized):")
cv2_imshow(divided)
