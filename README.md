Image Watermarking and Encryption Project
This project implements an image encryption and invisible watermarking pipeline using a combination of DCT (Discrete Cosine Transform) and LSB (Least Significant Bit) hybrid watermarking, along with pixel shuffling. It enables embedding, encrypting, and extracting invisible watermarks from images.

Features
Hybrid Watermarking: Combines DCT and LSB for robust invisible watermarking.
Pixel Shuffling: Encrypts images by scrambling pixels with a key.
Watermark Embedding: Embed an invisible watermark into a cover image.
Watermark Extraction: Extract watermark from the encrypted/decrypted image using the same key.
Complete pipeline: Watermark → Encrypt → Decrypt → Extract.

Tech Stack
Python
NumPy
OpenCV
SciPy (DCT, IDCT)
Matplotlib (for visualization)

Getting Started
Clone this repository:
git clone https://github.com/TejaswiRudraraju1/image-watermarking-project.git
cd image-watermarking-project


Install dependencies:
pip install -r requirements.txt

Run the pipeline:
python main.py

Project Structure
image-watermarking-project/
│── watermark_embed.py     # Embed invisible watermark
│── watermark_extract.py   # Extract watermark
│── pixel_shuffle.py       # Pixel-level shuffling and unshuffling
│── utils.py               # Helper functions
│── watermarked_image.jpg  # Example output
│── main.py                # Run the full pipeline
│── requirements.txt       # Dependencies
│── README.md              # Project documentation

Workflow
Input an original image and watermark.
Apply DCT + LSB watermarking → get watermarked_image.jpg.
Apply pixel shuffling → get encrypted image2.jpg.
Use the key to unshuffle → recover the original watermarked image.
Extract the watermark from the image.
Future Improvements
Add a GUI or web application (Flask/Streamlit).
Support multiple watermark embedding.
Use Git LFS for large files such as .weights.

License
This project is open-source under the MIT License. You are free to fork, improve, and share.

requirements.txt
numpy
opencv-python
scipy
matplotlib
