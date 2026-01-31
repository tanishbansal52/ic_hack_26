import time
import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# NOTE: need these installed:
# pip install ultralytics opencv-python insightface onnxruntime numpy

# ---------------------------
# Config you may tweak
# ---------------------------
YOLO_MODEL = "yolov8n.pt"        # small & fast; try yolov8s.pt for better detection (slower)
PERSON_CONF = 0.35              # person detection confidence threshold
FACE_DET_SIZE = (640, 640)      # face detector input size
COSINE_THRESH = 0.45            # stricter = fewer false matches (0.40-0.55 typical range)
MIN_FACE_AREA = 60 * 60         # skip tiny faces (helps accuracy on webcam)
CAM_INDEX = 0

# ---------------------------
# Helper functions
# ---------------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def bbox_contains(person_xyxy, point_xy) -> bool:
    x1, y1, x2, y2 = person_xyxy
    x, y = point_xy
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def draw_label(img, text, x, y, bg=(0, 0, 0), fg=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 10, y), bg, -1)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2, cv2.LINE_AA)

# ---------------------------
# Initialize models
# ---------------------------
print("Loading YOLO person detector...")
yolo = YOLO(YOLO_MODEL)

print("Loading InsightFace (ArcFace embeddings)...")
# ctx_id = -1 => CPU
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=FACE_DET_SIZE)

# ---------------------------
# Identity database:
# known_people: list of dicts: {"id": int, "embs": [np.ndarray, ...], "last_seen": float}
# ---------------------------
known_people = []
next_person_id = 1

def match_or_create_identity(face_emb: np.ndarray, now: float):
    """
    Returns (person_id, best_sim, was_new)
    """
    global next_person_id

    emb = l2_normalize(face_emb.astype(np.float32))

    best_id = None
    best_sim = -1.0

    for person in known_people:
        # compare against that person's stored embeddings (take the best)
        sims = [cosine_sim(emb, e) for e in person["embs"]]
        s = max(sims) if sims else -1.0
        if s > best_sim:
            best_sim = s
            best_id = person["id"]

    if best_id is not None and best_sim >= COSINE_THRESH:
        # update that person's embedding set (keep a small set)
        for person in known_people:
            if person["id"] == best_id:
                person["last_seen"] = now
                # store only if it's not near-duplicate
                if all(cosine_sim(emb, e) < 0.98 for e in person["embs"]):
                    person["embs"].append(emb)
                    # cap memory per person
                    if len(person["embs"]) > 10:
                        person["embs"] = person["embs"][-10:]
                break
        return best_id, best_sim, False

    # otherwise, create a new identity
    pid = next_person_id
    next_person_id += 1
    known_people.append({"id": pid, "embs": [emb], "last_seen": now})
    return pid, best_sim, True

# ---------------------------
# Main loop
# ---------------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing CAM_INDEX.")

print("Running... Press 'q' to quit.")
prev_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    now = time.time()

    # --- Person detection ---
    # YOLO class 0 is "person" for COCO models
    y = yolo.predict(frame, conf=PERSON_CONF, classes=[0], verbose=False)[0]

    person_boxes = []
    if y.boxes is not None and len(y.boxes) > 0:
        for b in y.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            conf = float(b.conf[0].cpu().numpy())
            person_boxes.append((x1, y1, x2, y2, conf))

    people_count = len(person_boxes)

    # --- Face detection + embeddings ---
    faces = face_app.get(frame)

    # For each person box, we will assign a label if a face falls inside it
    person_labels = {}  # index -> "Person N"
    used_face_person_ids = set()

    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        if w * h < MIN_FACE_AREA:
            continue

        # Face center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Find which person box contains this face center
        best_person_idx = None
        best_area = None
        for i, (px1, py1, px2, py2, pconf) in enumerate(person_boxes):
            if bbox_contains((px1, py1, px2, py2), (cx, cy)):
                area = (px2 - px1) * (py2 - py1)
                if best_area is None or area < best_area:
                    # prefer the tightest containing box (usually correct)
                    best_area = area
                    best_person_idx = i

        if best_person_idx is None:
            continue

        emb = f.embedding
        pid, sim, was_new = match_or_create_identity(emb, now)

        # Avoid assigning multiple faces to different IDs for the same person box
        if pid in used_face_person_ids:
            continue
        used_face_person_ids.add(pid)

        person_labels[best_person_idx] = f"Person {pid}"

        # draw face box (optional)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 200, 40), 2)

    # --- Draw person boxes + labels ---
    for i, (x1, y1, x2, y2, conf) in enumerate(person_boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 40), 2)
        label = person_labels.get(i, "Unknown")
        draw_label(frame, label, x1, y1)

    # --- HUD: count + FPS ---
    dt = now - prev_time
    prev_time = now
    fps = 1.0 / (dt + 1e-9)
    draw_label(frame, f"People in frame: {people_count}", 10, 30)
    draw_label(frame, f"FPS: {fps:.1f}", 10, 60)

    cv2.imshow("Live People Count + Face Identity", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
