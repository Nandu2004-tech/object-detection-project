import cv2
import numpy as np
import os

# === Load YOLOv3-SPP Model ===
net = cv2.dnn.readNet(
    r"C:\Users\knand\Desktop\opencv_project\yolov3-spp.weights",
    r"C:\Users\knand\Desktop\opencv_project\yolov3-spp.cfg"
)

# === Use CPU (or uncomment for GPU) ===
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# === Load COCO class labels ===
coco_file = r"C:\Users\knand\Desktop\opencv_project\coco.names"
if not os.path.exists(coco_file):
    print(f"[ERROR] File not found: {coco_file}")
    exit()

with open(coco_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(f"[INFO] Loaded {len(classes)} classes from coco.names")

# === Get YOLO Output Layers ===
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# === Assign Random Colors to Classes ===
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# === Open Webcam and Try 4K ===
cap = cv2.VideoCapture(0)

# Request 4K
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# Get actual camera resolution
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Actual camera resolution: {actual_width} x {actual_height}")

# If not 4K, fall back to 1080p
if actual_width < 3840 or actual_height < 2160:
    print("[WARN] 4K not supported. Falling back to 1920x1080.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Fallback resolution: {actual_width} x {actual_height}")

# === Optional: Save Output ===
save_output = False
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("output_spp.avi", fourcc, 20.0, (actual_width, actual_height))

# === Create Display Window ===
cv2.namedWindow("YOLOv3-SPP Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv3-SPP Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("[INFO] Detection started. Press ESC to exit.")

# === Detection Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        break

    height, width = frame.shape[:2]

    # === Prepare input for YOLO ===
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # === Process Detections ===
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # === Draw Bounding Boxes ===
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # === Show Detection Frame ===
    cv2.imshow("YOLOv3-SPP Detection", frame)

    if save_output:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        print("[INFO] Exiting...")
        break

# === Cleanup ===
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
