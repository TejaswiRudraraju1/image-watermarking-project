import cv2

# === Load the true key and watermark ===
with open("watermark_key.txt", "r") as f:
    true_key = f.read().strip()

with open("watermark_bits.txt", "r") as f:
    original_watermark = f.read().strip()

if not true_key or not original_watermark:
    raise ValueError(" Key or watermark file is empty.")

# === Ask user to input key ===
key_input = input(f"🔑 Enter the binary key ({len(true_key)} bits): ").strip()

if key_input != true_key:
    print(" Wrong key. Access denied.")
    exit()

# === Key matched: print and overlay watermark ===
print("\n Key verified successfully.")
print(f"🖋️ Original Watermark ({len(original_watermark)} bits): {original_watermark}")

# === Load original image ===
original_img = cv2.imread("image.jpg")
if original_img is None:
    raise FileNotFoundError(" image.jpg not found.")

# === Overlay watermark text ===
output = original_img.copy()
cv2.putText(output, f"Extracted WM: {original_watermark}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# === Show and save ===
cv2.imshow("Extracted Watermark", output)
cv2.imwrite("extracted_watermark_color_image.jpg", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
