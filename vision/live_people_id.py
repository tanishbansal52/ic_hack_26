import os
import time
import cv2
import numpy as np

from ultralytics import YOLO
from insightface.app import FaceAnalysis
import mediapipe as mp

import sqlite3
import sqlite_vec
import struct



# ======================
# CONFIG
# ======================
CAM_INDEX = 0

# People detection (counting)
YOLO_MODEL = "yolov8n.pt"
PERSON_CONF = 0.35

# Face recognition (identity)
FACE_DET_SIZE = (640, 640)
COSINE_THRESH = 0.50              # stricter => fewer false matches (0.45–0.60 typical)
DB_PATH = "faces_db.npz"

# Face/landmark quality gates
MIN_FACE_AREA = 40 * 40           # lower => works when you are further away; raise later for stability

# Attention (binary)
EAR_CLOSED_THRESH = 0.20          # if false "closed": lower to 0.18; if never closed: raise to 0.22
EYES_CLOSED_LONG_SEC = 1.0        # closed this long => inattentive
NOSE_CENTER_RATIO_THRESH = 0.17   # smaller => stricter "forward". Typical 0.12–0.22

# Debug
SHOW_DEBUG_OVERLAY = True
PRINT_DEBUG_EVERY_N_FRAMES = 30   # set 0 to disable terminal prints


# ======================
# Persistence (save/load identities)
# ======================
def save_db(path, people, next_id):
    ids = np.array([p["id"] for p in people], dtype=np.int32)
    embs = np.array([np.stack(p["embs"]) for p in people], dtype=object)
    np.savez_compressed(path, ids=ids, embs=embs, next_id=np.int32(next_id))


def load_db(path):
    if not os.path.exists(path):
        return [], 1

    data = np.load(path, allow_pickle=True)

    # Backward compatible: some older files might not have next_id
    if "next_id" in data:
        next_id = int(data["next_id"])
    else:
        ids_arr = data["ids"] if "ids" in data else np.array([], dtype=np.int32)
        next_id = (int(np.max(ids_arr)) + 1) if len(ids_arr) > 0 else 1

    people = []
    if "ids" in data and "embs" in data:
        for pid, e in zip(data["ids"], data["embs"]):
            people.append({"id": int(pid), "embs": [x.astype(np.float32) for x in e]})

    return people, next_id


# ======================
# Helpers
# ======================
def l2norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


def cos_sim(a, b):
    return float(np.dot(a, b))


def bbox_area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area_xyxy(a) + bbox_area_xyxy(b) - inter + 1e-9
    return inter / union


def draw_label(img, text, x, y, bg=(0, 0, 0), fg=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 10, y), bg, -1)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 2, cv2.LINE_AA)


# ======================
# MediaPipe indices for EAR (eye openness)
# ======================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def ear_from_landmarks(pts, idx):
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-9)


def simple_head_forward_ratio(pts):
    """
    Simple 2D head "forward" proxy:
    - Take nose x
    - Compare to midpoint between left and right eye corners
    - Normalize by eye distance
    If nose stays near the middle, head is likely facing forward.
    """
    nose = pts[1]         # nose tip-ish in FaceMesh
    left_eye = pts[33]    # left eye outer corner
    right_eye = pts[263]  # right eye outer corner

    eye_mid_x = 0.5 * (left_eye[0] + right_eye[0])
    eye_dist = abs(right_eye[0] - left_eye[0]) + 1e-6
    offset = abs(nose[0] - eye_mid_x)

    ratio = offset / eye_dist  # normalized
    return ratio


# ======================
# Identity matching
# ======================
def match_or_create_identity(face_emb):
    e = l2norm(face_emb.astype(np.float32))

    best_person = None
    best_s = -1.0

    # for p in people:
    #     if not p["embs"]:
    #         continue
    #     s = max(cos_sim(e, emb) for emb in p["embs"])
    #     if s > best_s:
    #         best_s = s
    #         best_person = p

    
    db = sqlite3.connect("database.db")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False) # Best practice: disable after loading

    cur = db.cursor()

    embedding_bytes = struct.pack(f'{len(face_emb)}f', *face_emb)
    query = """
    SELECT rowid, embedding
    FROM people_vectors
    ORDER BY embedding <-> ?
    LIMIT 1
    """
    result = cur.execute(query, (embedding_bytes,)).fetchone()

    if result:
        person_id, person_embedding_bytes = result
        person_embedding = np.array(struct.unpack(f'{len(face_emb)}f', person_embedding_bytes), dtype=np.float32)
        # Calculate cosine similarity
        dot_product = np.dot(face_emb, person_embedding)
        norm_embedding = np.linalg.norm(face_emb)
        norm_person = np.linalg.norm(person_embedding)
        cosine_similarity = dot_product / (norm_embedding * norm_person)
        best_person = person_id
        best_s = cosine_similarity

    
    

    if best_person is not None and best_s >= COSINE_THRESH:
        # get the user id given the row id
        cur.execute("SELECT id FROM people WHERE embedding_row_id = ?", (best_person,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Inconsistent DB state: embedding found but no corresponding person ID.")
        best_person = row[0]

        # Add some variety, but cap it
        if all(cos_sim(e, emb) < 0.98 for emb in best_person["embs"]):
            best_person["embs"].append(e)
            if len(best_person["embs"]) > 12:
                best_person["embs"] = best_person["embs"][-12:]
        cur.close()
        db.commit()
        return best_person, best_s, False

    # insert into the db
    cursor = db.execute("INSERT INTO people_vectors(embedding) VALUES (?)", (embedding_bytes,))
    embedding_id = cursor.lastrowid  # Get the auto-generated rowid

    db.execute("""
        SELECT max(id) FROM people
    """)

    row = cur.fetchone()
    next_id = 1 if row[0] is None else row[0] + 1

    db.execute("""
        INSERT INTO people
        (id, embedding_row_id) VALUES (?, ?)
    """, (next_id, embedding_id))

    cur.close()
    db.commit()
    return next_id, best_s, True


# ======================
# MAIN
# ======================
def main():
    print("Loading models...")
    yolo = YOLO(YOLO_MODEL)

    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=FACE_DET_SIZE)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=10,
        refine_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    # people, next_id = load_db(DB_PATH)
    # print(f"Loaded {len(people)} saved identities from {DB_PATH}. Next id={next_id}")

    # Per-identity state for attention
    # state[pid] = {"closed_since": t or None, "last_attentive": 0/1/None, "last_update": t}
    state = {}

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try CAM_INDEX=0/1.")

    frame_i = 0
    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_i += 1
            now = time.time()
            H, W = frame.shape[:2]

            # ----------------------
            # People count (YOLO)
            # ----------------------
            y = yolo.predict(frame, conf=PERSON_CONF, classes=[0], verbose=False)[0]
            person_boxes = []
            if y.boxes is not None and len(y.boxes) > 0:
                for b in y.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                    person_boxes.append((x1, y1, x2, y2))

            # ----------------------
            # Face recognition (InsightFace)
            # ----------------------
            faces_if = face_app.get(frame)
            if_faces = []
            for f in faces_if:
                fx1, fy1, fx2, fy2 = [int(v) for v in f.bbox]
                if bbox_area_xyxy((fx1, fy1, fx2, fy2)) < MIN_FACE_AREA:
                    continue
                if_faces.append({"bbox": (fx1, fy1, fx2, fy2), "emb": f.embedding})

            # ----------------------
            # Face landmarks (MediaPipe)
            # ----------------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            mp_faces = []
            if res.multi_face_landmarks:
                for fm in res.multi_face_landmarks:
                    pts = np.array([(lm.x * W, lm.y * H) for lm in fm.landmark], dtype=np.float32)
                    x1 = int(np.clip(np.min(pts[:, 0]), 0, W - 1))
                    y1 = int(np.clip(np.min(pts[:, 1]), 0, H - 1))
                    x2 = int(np.clip(np.max(pts[:, 0]), 0, W - 1))
                    y2 = int(np.clip(np.max(pts[:, 1]), 0, H - 1))
                    if bbox_area_xyxy((x1, y1, x2, y2)) < MIN_FACE_AREA:
                        continue
                    mp_faces.append({"bbox": (x1, y1, x2, y2), "pts": pts})

            # ----------------------
            # Pair MP faces to IF faces by IoU (robust pairing)
            # ----------------------
            # For each MP face, find best IF face match
            matched = []
            used_if = set()

            for mi, mpf in enumerate(mp_faces):
                best_j = None
                best_iou = 0.0
                for j, iff in enumerate(if_faces):
                    if j in used_if:
                        continue
                    s = iou_xyxy(mpf["bbox"], iff["bbox"])
                    if s > best_iou:
                        best_iou = s
                        best_j = j

                # If IoU is tiny, still allow a weak match if there is only one face around.
                if best_j is not None and (best_iou >= 0.10 or (len(if_faces) == 1 and len(mp_faces) == 1)):
                    used_if.add(best_j)
                    matched.append((mpf, if_faces[best_j], best_iou))
                else:
                    matched.append((mpf, None, best_iou))

            # ----------------------
            # Compute identity + attention for matched faces
            # ----------------------
            for mpf, iff, best_iou in matched:
                mx1, my1, mx2, my2 = mpf["bbox"]
                pts = mpf["pts"]

                # Compute attention features first (works even if identity fails)
                ear_val = 0.5 * (ear_from_landmarks(pts, LEFT_EYE) + ear_from_landmarks(pts, RIGHT_EYE))
                eyes_closed = ear_val < EAR_CLOSED_THRESH

                if eyes_closed:
                    closed_long_flag = True  # will become true only after timer per person id exists
                else:
                    closed_long_flag = False

                head_ratio = simple_head_forward_ratio(pts)
                head_forward = head_ratio < NOSE_CENTER_RATIO_THRESH

                # Identity (if we have InsightFace embedding for this MP face)
                if iff is None:
                    pid = None
                else:
                    pid, next_id, sim, was_new = match_or_create_identity(iff["emb"])

                attentive = None

                if pid is not None:
                    if pid not in state:
                        state[pid] = {"closed_since": None, "last_attentive": None, "last_update": 0.0}
                    st = state[pid]

                    # Update closed timer
                    if eyes_closed:
                        if st["closed_since"] is None:
                            st["closed_since"] = now
                    else:
                        st["closed_since"] = None

                    eyes_closed_long = (st["closed_since"] is not None) and ((now - st["closed_since"]) >= EYES_CLOSED_LONG_SEC)

                    # Binary decision
                    attentive = 1 if (head_forward and not eyes_closed_long) else 0

                    st["last_attentive"] = attentive
                    st["last_update"] = now
                else:
                    eyes_closed_long = False  # unknown without a stable identity timer

                # Draw face box and labels
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (40, 200, 40), 2)

                if pid is None:
                    draw_label(frame, "Person: ?", mx1, my1)
                    draw_label(frame, "Attentive: --", mx1, my1 + 28)
                else:
                    draw_label(frame, f"Person {pid}", mx1, my1)
                    draw_label(frame, f"Attentive: {attentive}", mx1, my1 + 28)

                # Debug overlay
                if SHOW_DEBUG_OVERLAY:
                    draw_label(frame, f"EAR: {ear_val:.3f} closed_long:{int(eyes_closed_long)}", mx1, my1 + 56)
                    draw_label(frame, f"nose_ratio:{head_ratio:.3f} forward:{int(head_forward)}", mx1, my1 + 84)
                    if iff is not None:
                        draw_label(frame, f"IoU(IF/MP): {best_iou:.2f}", mx1, my1 + 112)

                # Terminal debug prints (optional)
                if PRINT_DEBUG_EVERY_N_FRAMES and frame_i % PRINT_DEBUG_EVERY_N_FRAMES == 0 and pid is not None:
                    print(
                        f"[Person {pid}] ear={ear_val:.3f} "
                        f"head_ratio={head_ratio:.3f} head_forward={head_forward} "
                        f"attentive={attentive}"
                    )

            # ----------------------
            # Draw YOLO person boxes (counting)
            # ----------------------
            for (x1, y1, x2, y2) in person_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 40), 2)

            # ----------------------
            # HUD
            # ----------------------
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / (dt + 1e-9)

            draw_label(frame, f"People in frame: {len(person_boxes)}", 10, 30)
            draw_label(frame, f"FPS: {fps:.1f}", 10, 60)
            draw_label(frame, f"Saved identities: {len(people)}", 10, 90)
            draw_label(frame, "q: quit | c: clear DB", 10, 120)

            cv2.imshow("People + Identity + Binary Attention (Fixed)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                # Clear identity DB
                people = []
                next_id = 1
                state = {}
                if os.path.exists(DB_PATH):
                    os.remove(DB_PATH)
                print("Cleared saved identities (deleted faces_db.npz).")

    finally:
        save_db(DB_PATH, people, next_id)
        print(f"Saved {len(people)} identities to {DB_PATH}.")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
