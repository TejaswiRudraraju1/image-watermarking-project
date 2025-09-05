import cv2
import numpy as np
from scipy.fftpack import dct, idct

# Load watermark bits
with open("watermark_bits.txt", "r") as f:
    watermark_bits = f.read().strip()

# Load image
img = cv2.imread("image.jpg")
if img is None:
    print("❌ Image not found.")
    exit()

# Convert to YCrCb
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(img_ycrcb)

h, w = y.shape
watermarked_y = np.copy(y)
bit_idx = 0

# Embed watermark using DCT-LSB
for i in range(0, h - 8 + 1, 8):
    for j in range(0, w - 8 + 1, 8):
        if bit_idx >= len(watermark_bits):
            break

        block = y[i:i+8, j:j+8]
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        # Target coefficient
        coef = dct_block[4, 3]
        coef_int = int(np.round(coef))

        # Force LSB
        new_coef = (coef_int & ~1) | int(watermark_bits[bit_idx])
        dct_block[4, 3] = new_coef

        # Inverse DCT
        idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
        watermarked_y[i:i+8, j:j+8] = np.clip(idct_block, 0, 255)

        bit_idx += 1
    if bit_idx >= len(watermark_bits):
        break

# Recombine and save
final_y = watermarked_y.astype(np.uint8)
merged = cv2.merge([final_y, cr, cb])
final_img = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

cv2.imwrite("dct_lsb_watermarked_color.jpg", final_img)
cv2.imshow("Watermarked Image", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()