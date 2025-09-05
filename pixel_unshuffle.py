import cv2
import numpy as np

# === Load the shuffled image (PNG) ===
shuffled_img = cv2.imread("pixel_shuffled_image.png")
if shuffled_img is None:
    print("Shuffled image not found.")
    exit()

h, w, c = shuffled_img.shape
num_pixels = h * w

# === Ask for the seed ===
try:
    seed = int(input(" Enter the shuffle seed (from shuffle_seed.txt): ").strip())
except ValueError:
    print(" Invalid seed input.")
    exit()

# === Regenerate key and inverse key ===
rng = np.random.default_rng(seed)
key = rng.permutation(num_pixels)
inverse_key = np.argsort(key)

# === Flatten and unshuffle ===
flattened = shuffled_img.reshape(-1, c)
unshuffled_flat = flattened[inverse_key]

# === Validate marker at index 0 ===
expected_marker = np.array([123, 45, 67], dtype=np.uint8)
if not np.array_equal(unshuffled_flat[0], expected_marker):
    print(" Incorrect seed! Marker mismatch. Unshuffling failed.")
    exit()

# === Save and display result ===
unshuffled_img = unshuffled_flat.reshape(h, w, c)
cv2.imwrite("unshuffled_image.png", unshuffled_img)

print(" Correct seed! Image successfully unshuffled and saved as 'unshuffled_image.png'.")
cv2.imshow("Unshuffled Image", unshuffled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
