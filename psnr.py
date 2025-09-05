import cv2
import numpy as np

def calculate_psnr(original_path, processed_path):
    original = cv2.imread(original_path)
    processed = cv2.imread(processed_path)
    
    if original is None or processed is None:
        print("❌ One of the images not found.")
        return None
    
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

# Example usage:
psnr_val = calculate_psnr("image.jpg", "dct_lsb_watermarked_color.jpg")
print(f"PSNR between original and watermarked image: {psnr_val:.2f} dB")
