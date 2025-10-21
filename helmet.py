from ultralytics import YOLO
import cv2

# Load the custom YOLOv11n or YOLOv8 model
model = YOLO("best.pt")  # or "best.pt" if that's your model

# Define class names manually
class_names = ['helmet']

# Start video capture
cap = cv2.VideoCapture('f.mp4')

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize frame to 640x640 for consistent detection size
    resized_frame = cv2.resize(frame, (640, 640))

    # Run YOLO detection
    results = model(resized_frame, stream=True)

    # Process detection results
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{class_names[cls_id]} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(resized_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show the resized frame
    cv2.imshow("Helmet Detection - YOLOv11n (640x640)", resized_frame)

    # Press 'q' to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()