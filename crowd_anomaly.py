from ultralytics import YOLO
import cv2

# Load model
model = YOLO("models/best.pt")

# Open video
cap = cv2.VideoCapture("videos/factory.mp4")

# Danger zone expansion around forklift
MARGIN = 100

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]

    persons = []
    forklifts = []

    # ---------------- DETECTIONS ----------------
    for box in results.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        label = model.names[cls]

        # PERSON
        if label == "Person":

            # Store full information
            persons.append((x1, y1, x2, y2, cx, cy))

        # FORKLIFT
        elif label == "Forklift":

            forklifts.append((x1, y1, x2, y2))

            # Blue box
            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (255, 0, 0),
                          2)

            cv2.putText(frame,
                        "Forklift",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2)

    workers_near_machine = 0

    # ---------------- DANGER ZONE ----------------
    for fx1, fy1, fx2, fy2 in forklifts:

        dx1 = max(0, fx1 - MARGIN)
        dy1 = max(0, fy1 - MARGIN)
        dx2 = fx2 + MARGIN
        dy2 = fy2 + MARGIN

        # Yellow danger zone
        cv2.rectangle(frame,
                      (dx1, dy1),
                      (dx2, dy2),
                      (0, 255, 255),
                      2)

        worker_number = 1

        for px1, py1, px2, py2, px, py in persons:

            # Person inside danger zone
            if dx1 <= px <= dx2 and dy1 <= py <= dy2:

                workers_near_machine += 1

                # Red box
                cv2.rectangle(frame,
                              (px1, py1),
                              (px2, py2),
                              (0, 0, 255),
                              3)

                cv2.putText(frame,
                            f"Worker {worker_number}",
                            (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2)

            else:

                # Green box
                cv2.rectangle(frame,
                              (px1, py1),
                              (px2, py2),
                              (0, 255, 0),
                              2)

                cv2.putText(frame,
                            f"Worker {worker_number}",
                            (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

            worker_number += 1

    # ---------------- RISK ----------------
    if workers_near_machine == 0:
        risk = "SAFE"
        color = (0, 255, 0)

    elif workers_near_machine == 1:
        risk = "WARNING"
        color = (0, 255, 255)

    else:
        risk = "CRITICAL"
        color = (0, 0, 255)

        cv2.putText(frame,
                    "!!! OVERCROWDING DETECTED !!!",
                    (180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    # ---------------- DISPLAY ----------------
    cv2.putText(frame,
                f"Workers In Video : {len(persons)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                f"Workers Near Forklift : {workers_near_machine}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                f"Risk : {risk}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3)

    cv2.imshow("Crowd Anomaly Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()