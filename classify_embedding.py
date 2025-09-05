import cv2
import random

# Path to the image
image_path = "image.jpg"

# Load YOLOv4-tiny
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip().lower() for line in f.readlines()]
layer_names = net.getUnconnectedOutLayersNames()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def classify_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        return 'face', 'high'

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    detected_classes = set()
    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = int(scores.argmax())
            confidence = scores[class_id]
            if confidence > 0.5:
                label = classes[class_id]
                detected_classes.add(label)

    high_priority = {'person', 'face', 'document', 'laptop', 'keyboard', 'cell phone', 'book'}
    if any(obj in detected_classes for obj in high_priority):
        return ', '.join(detected_classes), 'high'
    else:
        return ', '.join(detected_classes) if detected_classes else 'unknown', 'low'

def generate_watermark_and_key(security_level):
    bit_lengths = {"high": 16, "low": 8}
    bits = ''.join(str(random.randint(0, 1)) for _ in range(bit_lengths[security_level]))
    key = ''.join(str(random.randint(0, 1)) for _ in range(bit_lengths[security_level]))
    return bits, key

def embed_visible_watermark(image, watermark_text, label):
    overlay = image.copy()
    height, width = image.shape[:2]

    cv2.rectangle(overlay, (0, 0), (width, 50), (0, 0, 0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    position = (10, 30)

    cv2.putText(
        overlay,
        f"Watermark ({label.upper()}): {watermark_text}",
        position,
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    alpha = 0.8
    watermarked = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return watermarked

def main():
    print("🔍 Classifying image and generating watermark...")
    image_class, security_level = classify_image(image_path)
    watermark, key = generate_watermark_and_key(security_level)

    image = cv2.imread(image_path)
    watermarked_image = embed_visible_watermark(image, watermark, security_level)

    print("\n Classification and Watermarking Complete:")
    print(f"📸 Classification : {image_class.upper()}")
    print(f"🔐 Security Level : {security_level.upper()}")
    print(f"🖋️ Watermark      : {watermark}")
    print(f"🔑 Key            : {key}")

    # Save watermark and key to files
    with open("watermark_bits.txt", "w") as f:
        f.write(watermark)
    with open("watermark_key.txt", "w") as f:
        f.write(key)

    # Show and save the image
    cv2.imshow("Watermarked Image", watermarked_image)
    cv2.imwrite("watermarked_output.jpg", watermarked_image)
    print("💾 Image saved as watermarked_output.jpg")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
