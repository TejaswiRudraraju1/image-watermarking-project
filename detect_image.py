import cv2  #for image processing and object detection
import numpy as np  #for numerical operations

# Load YOLOv4-tiny model
print("Loading YOLO model...")
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
img = cv2.imread("image.jpg")
if img is None:
    print("❌ Image not found. Check the file path.")
    exit()

height, width, channels = img.shape
print(f"Image loaded: {height}x{width}x{channels}")

"""
Converting the image into a "blob"(a standardized data format) that can be fed into the network.
0.00392: Normalizes the image by multiplying pixel values by 0.00392 (1/255) to bring them into the [0, 1] range.
(416, 416): Resizes the image to 416x416 pixels (required input size for YOLO).
(0, 0, 0): Subtracts mean values for each channel (not needed in this case).
True: Means to swap the BGR channels to RGB (if necessary).
"""
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Collect YOLO detections
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.2:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply NMS to YOLO boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.1)

# Draw YOLO detections
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# ---------- Haar Cascade Face Detection ----------
print("Detecting image...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Draw face boxes/
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# ---------- Display and Save Output ----------
cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Also save to file in case GUI fails
cv2.imwrite("output_with_faces.jpg", img)
print("✅ Output saved as output_with_faces.jpg")