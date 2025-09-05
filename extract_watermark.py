import cv2
import numpy as np
from scipy.fftpack import dct

# === Load unshuffled image ===
img = cv2.imread("unshuffled_image.png")
if img is None:
    print("❌ Image not found: unshuffled_image.png")
    exit()

# === Load watermark to get bit length ===
try:
    with open("watermark_bits.txt", "r") as f:
        watermark_bits = f.read().strip()
        desired_bits = len(watermark_bits)
except FileNotFoundError:
    print("❌ watermark_bits.txt not found.")
    exit()

# === Convert to YCrCb and extract Y channel ===
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, _, _ = cv2.split(img_ycrcb)

h, w = y.shape
extracted_bits = []

# === Extraction loop (limited to desired bits) ===
for i in range(0, h - 8 + 1, 8):
    for j in range(0, w - 8 + 1, 8):
        if len(extracted_bits) >= desired_bits:
            break

        block = y[i:i+8, j:j+8]
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        coef = dct_block[4, 3]
        coef_int = int(np.round(coef))
        extracted_bit = coef_int & 1
        extracted_bits.append(str(extracted_bit))
    if len(extracted_bits) >= desired_bits:
        break

# === Save extracted bits ===
with open("extracted_bits.txt", "w") as f:
    f.write("".join(extracted_bits))

print(f"✅ Extracted {len(extracted_bits)} bits (expected: {desired_bits})")
print("💾 Saved as 'extracted_bits.txt'")
