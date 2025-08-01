import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload an image
print("Upload an image:")
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img = cv2.imread(img_path)

rows, cols = img.shape[:2]

# 6a. Translation
tx, ty = 100, 50  # translate x and y
M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img, M_translate, (cols, rows))

# 6b. Rotation
angle = 45
M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
rotated = cv2.warpAffine(img, M_rotate, (cols, rows))

# 6c. Scaling
scaled = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# 6d. Skewing (Shearing)
M_skew = np.float32([[1, 0.5, 0], [0.5, 1, 0]])
skewed = cv2.warpAffine(img, M_skew, (int(cols * 1.5), int(rows * 1.5)))

# Display results
titles_geo = ['Translated', 'Rotated', 'Scaled', 'Skewed']
images_geo = [translated, rotated, scaled, skewed]

for title, image in zip(titles_geo, images_geo):
    print(f"\n✅ {title}")
    cv2_imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
