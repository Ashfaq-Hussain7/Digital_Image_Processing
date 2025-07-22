import cv2
from PIL import Image
import os
import numpy as np

# A. Read the image
image_path = '/content/bird_image.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()
else :
  print("Image loaded successfully!!")
  cv2_imshow(image)

# B. Get image info
height, width, channels = image.shape
print("Image Information:")
print(f"Dimensions: {width}x{height}")
print(f"Channels: {channels}")
print(f"Data Type: {image.dtype}")

# C. Save copy with compression and find compression ratio
# Save original as PNG (no compression)
original_png = 'original.png'
cv2.imwrite(original_png, image)

# Save compressed copy as JPEG
compressed_jpeg = 'compressed.jpg'
cv2.imwrite(compressed_jpeg, image, [cv2.IMWRITE_JPEG_QUALITY, 50])  # 50% quality

# Compression Ratio = Uncompressed Size / Compressed Size
uncompressed_size = os.path.getsize(original_png)
compressed_size = os.path.getsize(compressed_jpeg)
compression_ratio = uncompressed_size / compressed_size

print("\nCompression Info:")
print(f"Uncompressed Size (PNG): {uncompressed_size} bytes")
print(f"Compressed Size (JPEG): {compressed_size} bytes")
print(f"Compression Ratio: {compression_ratio:.2f}")

# D. Display the negative of the image
from google.colab.patches import cv2_imshow
negative_image = 255 - image
print("\nDisplaying negative image:")
cv2_imshow(negative_image)