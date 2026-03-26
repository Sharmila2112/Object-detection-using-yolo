import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet(
    r"C:\Users\sharm\OneDrive\Documents\face_detection\yolov3-tiny.weights",
    r"C:\Users\sharm\OneDrive\Documents\face_detection\yolov3-tiny.cfg"
)
# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Camera not working")
    exit()

print("✅ YOLO Detection Started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    height, width, _ = frame.shape

    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(
        frame,
        1/255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected = False  # Track if anything detected

    # Draw bounding boxes
    if len(indexes) > 0:
        for i in indexes.flatten():
            detected = True

            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]

            color = (0, 255, 0)

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Show label
            text = f"{label} ({confidence:.2f})"
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # Global status text
    if detected:
        status_text = "Object Detected"
        status_color = (0, 255, 0)
    else:
        status_text = "No Object Detected"
        status_color = (0, 0, 255)

    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    # Show output
    cv2.imshow("YOLO Object Detection", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()