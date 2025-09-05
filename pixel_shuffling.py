import cv2
import numpy as np
import random

# === Load the image ===
img = cv2.imread("dct_lsb_watermarked_color.jpg")
if img is None:
    print(" Image not found.")
    exit()

h, w, c = img.shape
flattened = img.reshape(-1, c)
num_pixels = h * w

# === Embed known marker at index 0 (safe) ===
marker_index = 0
marker = np.array([123, 45, 67], dtype=np.uint8)
flattened[marker_index] = marker

# === Shuffle ===
seed = random.randint(0, 2**32 - 1)
rng = np.random.default_rng(seed)
key = rng.permutation(num_pixels)

shuffled_flat = flattened[key]
shuffled_img = shuffled_flat.reshape(h, w, c)

# === Save using PNG to avoid compression artifacts ===
cv2.imwrite("pixel_shuffled_image.png", shuffled_img)
with open("shuffle_seed.txt", "w") as f:
    f.write(str(seed))

print("Pixel shuffling completed.")
print(f"🔑 Shuffle seed saved to 'shuffle_seed.txt': {seed}")
print("📁 Shuffled image saved as 'pixel_shuffled_image.png'")
cv2.imshow("Pixel Shuffled Image", shuffled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
