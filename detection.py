import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Allowed vehicle classes (COCO dataset IDs)
VEHICLE_CLASSES = {
    2: "car",
    1: "bicycle",
    3: "motorcycle",
    7: "truck"
}

def process_video(input_path, output_path, lane_side="left"):

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    model_path = "models/yolov8n.pt"

    # Load YOLO model (downloads automatically first time)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs("data/processed", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    class_counts = defaultdict(int)
    counted_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False)

        mid_x = width // 2

        # Draw vertical lane divider
        cv2.line(frame, (mid_x, 0), (mid_x, height), (0, 255, 255), 2)

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, cls_id, obj_id in zip(boxes, class_ids, ids):

                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2

                # Lane filtering
                if lane_side == "left" and center_x > mid_x:
                    continue
                if lane_side == "right" and center_x < mid_x:
                    continue

                label = VEHICLE_CLASSES[cls_id]

                # Count only once per vehicle ID
                if obj_id not in counted_ids:
                    class_counts[label] += 1
                    counted_ids.add(obj_id)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Show ID + Class
                text = f"ID:{obj_id} {label}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        # ---- Display Only 5 Values ----
        car = class_counts["car"]
        bicycle = class_counts["bicycle"]
        motorcycle = class_counts["motorcycle"]
        truck = class_counts["truck"]
        total = car + bicycle + motorcycle + truck

        cv2.putText(frame, f"Car: {car}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(frame, f"Bicycle: {bicycle}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(frame, f"Motorcycle: {motorcycle}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(frame, f"Truck: {truck}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(frame, f"Total: {total}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print("✅ Processing completed successfully!")