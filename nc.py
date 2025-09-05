import numpy as np
import os

def generate_similar_bitstreams():
    # Original watermark bits (1000 bits)
    original_bits = ''.join(np.random.choice(['0', '1'], size=1000))

    # Slightly modify the original to simulate extraction errors (e.g., 2% difference)
    extracted_bits = list(original_bits)
    num_flips = int(0.02 * len(original_bits))  # Flip 2% bits

    flip_indices = np.random.choice(len(original_bits), size=num_flips, replace=False)
    for idx in flip_indices:
        extracted_bits[idx] = '0' if extracted_bits[idx] == '1' else '1'

    # Save to files
    with open("watermark_bits.txt", "w") as f:
        f.write(original_bits)
    with open("extracted_bits.txt", "w") as f:
        f.write(''.join(extracted_bits))


def calculate_nc(original_file="watermark_bits.txt", extracted_file="extracted_bits.txt"):
    if not os.path.exists(original_file) or not os.path.exists(extracted_file):
        print("❌ Required files not found.")
        return

    with open(original_file, "r") as f:
        original_bits = f.read().strip()

    with open(extracted_file, "r") as f:
        extracted_bits = f.read().strip()

    # Use the minimum length
    min_len = min(len(original_bits), len(extracted_bits))
    original_bits = original_bits[:min_len]
    extracted_bits = extracted_bits[:min_len]

    # Convert to numpy arrays
    orig_arr = np.array([int(b) for b in original_bits])
    ext_arr = np.array([int(b) for b in extracted_bits])

    # Compute NC
    numerator = np.sum(orig_arr * ext_arr)
    denominator = np.sqrt(np.sum(orig_arr ** 2) * np.sum(ext_arr ** 2))
    nc = numerator / denominator if denominator != 0 else 0

    print(f"🔍 Normalized Correlation (NC): {nc:.4f}")
    return nc


if __name__ == "__main__":
    generate_similar_bitstreams()
    calculate_nc()