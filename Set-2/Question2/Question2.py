import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload a clean image
print("Upload a clean image:")
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img = cv2.imread(img_path)
img = cv2.resize(img, (400, 400))

# Add Gaussian noise
mean = 0
stddev = 25
gaussian_noise = np.random.normal(mean, stddev, img.shape).astype(np.float32)
noisy_img = img.astype(np.float32) + gaussian_noise
noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

# Convert noisy image to float for averaging
noisy_float = noisy_img.astype(np.float32)

print("\n📷 Noisy Image:")
cv2_imshow(noisy_img)

# Averaging counts
averaging_counts = [2, 8, 16, 32, 128]

# Store results for later PSNR comparison
averaged_results = []

for count in averaging_counts:
    avg_img = np.zeros_like(noisy_float)
    for _ in range(count):
        noisy_instance = noisy_float + np.random.normal(0, 10, noisy_float.shape)
        avg_img += noisy_instance
    avg_img /= count
    avg_img = np.clip(avg_img, 0, 255).astype(np.uint8)
    averaged_results.append((count, avg_img))  # <-- store for matplotlib
    print(f"\n✅ Averaged over {count} samples:")
    cv2_imshow(avg_img)

import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

# Show images in a grid with PSNR values
plt.figure(figsize=(15, 10))

# Add original noisy image for comparison
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
plt.title("Noisy (PSNR: N/A)")
plt.axis('off')

# Display each averaged image
for idx, (count, img_avg) in enumerate(averaged_results):
    # PSNR compared to the original clean image
    score = psnr(img, img_avg)
    plt.subplot(2, 4, idx + 2)
    plt.imshow(cv2.cvtColor(img_avg, cv2.COLOR_BGR2RGB))
    plt.title(f"Avg {count}\nPSNR: {score:.2f} dB")
    plt.axis('off')

plt.tight_layout()
plt.show()
