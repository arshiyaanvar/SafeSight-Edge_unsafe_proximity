
from ultralytics import YOLO
import cv2
import time

# ==========================================
# 1. LOAD TWO YOLO MODELS
# ==========================================

# Pretrained COCO model for detecting workers
person_model = YOLO("yolov8s.pt")

# Your existing custom model for forklift detection
forklift_model = YOLO("models/best.pt")

print("Person Model Loaded")
print("Forklift Model Classes:", forklift_model.names)

# ==========================================
# 2. CONFIGURATION
# ==========================================

PERSON_CONF = 0.20
FORKLIFT_CONF = 0.25

MARGIN = 100

# ==========================================
# 3. OPEN FACTORY VIDEO
# ==========================================

cap = cv2.VideoCapture("videos/factory23.mp4")

if not cap.isOpened():
    print("ERROR: Cannot open factory video.")
    exit()

prev_time = time.time()

# ==========================================
# 4. MAIN VIDEO LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video Finished.")
        break

    frame_height, frame_width = frame.shape[:2]

    # ======================================
    # 5. DETECT WORKERS USING COCO YOLO
    # ======================================

    person_results = person_model.predict(
        frame,
        classes=[0],
        conf=PERSON_CONF,
        verbose=False
    )[0]

    persons = []

    for box in person_results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        conf = float(box.conf[0])

        # Worker centre point
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        persons.append({
            "box": (x1, y1, x2, y2),
            "center": (cx, cy),
            "confidence": conf,
            "danger": False
        })

    # ======================================
    # 6. DETECT FORKLIFTS USING CUSTOM YOLO
    # ======================================

    forklift_results = forklift_model.predict(
        frame,
        classes=[0],
        conf=FORKLIFT_CONF,
        verbose=False
    )[0]

    forklifts = []

    for box in forklift_results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        forklifts.append((x1, y1, x2, y2))

    # ======================================
    # 7. DRAW FORKLIFT DANGER ZONES
    # ======================================

    danger_zones = []

    for fx1, fy1, fx2, fy2 in forklifts:

        dx1 = max(0, fx1 - MARGIN)
        dy1 = max(0, fy1 - MARGIN)

        dx2 = min(frame_width, fx2 + MARGIN)
        dy2 = min(frame_height, fy2 + MARGIN)

        danger_zones.append(
            (dx1, dy1, dx2, dy2)
        )

        # Yellow danger zone
        cv2.rectangle(
            frame,
            (dx1, dy1),
            (dx2, dy2),
            (0, 255, 255),
            2
        )

        # Blue forklift bounding box
        cv2.rectangle(
            frame,
            (fx1, fy1),
            (fx2, fy2),
            (255, 0, 0),
            3
        )

        cv2.putText(
            frame,
            "FORKLIFT",
            (fx1, max(25, fy1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    # ======================================
    # 8. CHECK WORKERS INSIDE DANGER ZONES
    # ======================================

    workers_near_machine = 0

    for person in persons:

        px, py = person["center"]

        for dx1, dy1, dx2, dy2 in danger_zones:

            if dx1 <= px <= dx2 and dy1 <= py <= dy2:

                person["danger"] = True
                break

        if person["danger"]:
            workers_near_machine += 1

    # ======================================
    # 9. DRAW INDIVIDUAL WORKER BOXES
    # ======================================

    for index, person in enumerate(persons):

        x1, y1, x2, y2 = person["box"]

        conf = person["confidence"]

        if person["danger"]:

            color = (0, 0, 255)
            status = "DANGER"

        else:

            color = (0, 255, 0)
            status = "SAFE"

        # Individual worker bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = f"Worker {index + 1} | {status} | {conf:.2f}"

        # Label background
        label_width = min(
            x1 + 250,
            frame_width
        )

        cv2.rectangle(
            frame,
            (x1, max(0, y1 - 25)),
            (label_width, y1),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    # ======================================
    # 10. RISK CLASSIFICATION
    # ======================================

    if workers_near_machine == 0:

        risk = "SAFE"
        risk_color = (0, 255, 0)

    elif workers_near_machine == 1:

        risk = "WARNING"
        risk_color = (0, 255, 255)

    else:

        risk = "CRITICAL"
        risk_color = (0, 0, 255)

    # ======================================
    # 11. FPS
    # ======================================

    current_time = time.time()

    fps = 1 / max(current_time - prev_time, 0.001)

    prev_time = current_time

    # ======================================
    # 12. DASHBOARD
    # ======================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (490, 205),
        (25, 25, 25),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (10, 10),
        (490, 205),
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SAFESIGHT EDGE - PROXIMITY MONITOR",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Workers Detected: {len(persons)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Forklifts Detected: {len(forklifts)}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Workers Near Forklift: {workers_near_machine}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    # ======================================
    # 13. RISK LEVEL
    # ======================================

    cv2.putText(
        frame,
        f"RISK LEVEL: {risk}",
        (20, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        risk_color,
        3
    )

    # ======================================
    # 14. SAFETY ALERT
    # ======================================

    if risk == "CRITICAL":

        cv2.putText(
            frame,
            "UNSAFE PROXIMITY DETECTED!",
            (180, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )

    elif risk == "WARNING":

        cv2.putText(
            frame,
            "WARNING: WORKER NEAR FORKLIFT",
            (180, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # ======================================
    # 15. DISPLAY
    # ======================================

    cv2.imshow(
        "SafeSight Edge - Unsafe Proximity Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# 16. RELEASE RESOURCES
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Unsafe Proximity Detection Completed.")
