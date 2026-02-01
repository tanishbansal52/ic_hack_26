import time
import cv2
import numpy as np
import sqlite3

from ultralytics import YOLO
from insightface.app import FaceAnalysis
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp


# ======================
# CONFIG
# ======================
CAM_INDEX = 0

# People detection
YOLO_MODEL = "yolov8n.pt"
PERSON_CONF = 0.35

# Face recognition
FACE_DET_SIZE = (640, 640)
COSINE_THRESH = 0.50          # stricter => fewer false matches (0.45–0.60 typical)
MAX_EMBS_PER_PERSON = 12      # cap samples per person (keeps DB small)

# SQLite
SQLITE_PATH = "../imperial_students.db"  # Use root directory database

# Face / landmark reliability
MIN_FACE_AREA = 40 * 40

# Attention (binary)
EAR_CLOSED_THRESH = 0.20
EYES_CLOSED_LONG_SEC = 1.0
NOSE_CENTER_RATIO_THRESH = 0.17  # 0.12–0.22 typical

# Pairing IF (InsightFace) faces to MP (MediaPipe) faces
MIN_IOU_MATCH = 0.10

# Debug
SHOW_DEBUG_OVERLAY = True
PRINT_DEBUG_EVERY_N_FRAMES = 30  # 0 disables


# ======================
# DB + Embedding Store
# ======================
def init_db(db: sqlite3.Connection):
    cur = db.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at REAL DEFAULT (strftime('%s','now'))
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS people_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        created_at REAL DEFAULT (strftime('%s','now')),
        FOREIGN KEY(person_id) REFERENCES people(id)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS attention_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        module_name TEXT NOT NULL,
        session_start REAL NOT NULL,
        session_end REAL,
        total_frames INTEGER DEFAULT 0,
        attentive_frames INTEGER DEFAULT 0,
        eyes_closed_frames INTEGER DEFAULT 0,
        head_forward_frames INTEGER DEFAULT 0,
        attentiveness_score REAL DEFAULT 0.0,
        UNIQUE(person_id, module_name),
        FOREIGN KEY(person_id) REFERENCES people(id)
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_people_embeddings_person ON people_embeddings(person_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_attention_records_person ON attention_records(person_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_attention_records_module ON attention_records(module_name)")
    db.commit()


def count_people(db: sqlite3.Connection) -> int:
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM people")
    return int(cur.fetchone()[0])


def clear_people(db: sqlite3.Connection):
    cur = db.cursor()
    cur.execute("DELETE FROM people_embeddings")
    cur.execute("DELETE FROM people")
    db.commit()


class EmbeddingStore:
    """
    Keeps a RAM cache of normalized embeddings for fast matching,
    while persisting everything to SQLite.
    """

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.person_ids: list[int] = []
        self.embs: list[np.ndarray] = []  # normalized float32 (512,)
        self._load_all()

    @staticmethod
    def _l2norm(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def _load_all(self):
        self.person_ids.clear()
        self.embs.clear()

        cur = self.db.cursor()
        cur.execute("SELECT person_id, embedding FROM people_embeddings")
        for pid, blob in cur.fetchall():
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.size == 0:
                continue
            self.person_ids.append(int(pid))
            # assume stored already normalized; if not, normalize again safely:
            self.embs.append(self._l2norm(emb))

    def _insert_embedding(self, person_id: int, emb_norm: np.ndarray):
        blob = emb_norm.astype(np.float32).tobytes()
        cur = self.db.cursor()
        cur.execute(
            "INSERT INTO people_embeddings(person_id, embedding) VALUES (?, ?)",
            (int(person_id), sqlite3.Binary(blob)),
        )
        self.db.commit()

        # RAM cache update
        self.person_ids.append(int(person_id))
        self.embs.append(emb_norm)

    def _trim_person_embeddings(self, person_id: int, max_keep: int):
        # Keep only newest N embeddings per person
        cur = self.db.cursor()
        cur.execute("""
            SELECT id FROM people_embeddings
            WHERE person_id = ?
            ORDER BY id DESC
        """, (int(person_id),))
        ids = [row[0] for row in cur.fetchall()]
        if len(ids) <= max_keep:
            return
        to_delete = ids[max_keep:]
        cur.executemany("DELETE FROM people_embeddings WHERE id = ?", [(i,) for i in to_delete])
        self.db.commit()

        # Refresh RAM cache (simple + safe)
        self._load_all()

    def match_or_create(self, face_emb: np.ndarray) -> tuple[int, float, bool]:
        """
        Returns: (person_id, best_similarity, was_new)
        """
        emb_norm = self._l2norm(face_emb)

        if len(self.embs) == 0:
            # new person
            pid = self._create_new_person()
            self._insert_embedding(pid, emb_norm)
            return pid, -1.0, True

        # Cosine similarity since everything is normalized:
        # cos(a,b) = dot(a,b)
        dots = [float(np.dot(emb_norm, e)) for e in self.embs]
        best_idx = int(np.argmax(dots))
        best_s = float(dots[best_idx])
        best_pid = int(self.person_ids[best_idx])

        if best_s >= COSINE_THRESH:
            # add sample if not near-duplicate vs that person's existing embeddings
            # quick check: if any same-person embedding is extremely close, skip
            same_idxs = [i for i, pid in enumerate(self.person_ids) if pid == best_pid]
            too_close = any(float(np.dot(emb_norm, self.embs[i])) > 0.98 for i in same_idxs)
            if not too_close:
                self._insert_embedding(best_pid, emb_norm)
                self._trim_person_embeddings(best_pid, MAX_EMBS_PER_PERSON)
            return best_pid, best_s, False

        # create new person
        pid = self._create_new_person()
        self._insert_embedding(pid, emb_norm)
        return pid, best_s, True

    def _create_new_person(self) -> int:
        cur = self.db.cursor()
        cur.execute("INSERT INTO people DEFAULT VALUES")
        self.db.commit()
        return int(cur.lastrowid)


class AttentionTracker:
    """
    Tracks attention metrics for each person across a session.
    Stores data in imperial_students.db with module information.
    """
    
    def __init__(self, db: sqlite3.Connection, module_name: str):
        self.db = db
        self.module_name = module_name
        self.sessions = {}  # pid -> session data
        
    def update(self, person_id: int, is_attentive: bool, eyes_closed_long: bool, head_forward: bool, timestamp: float):
        """Update attention metrics for a person."""
        if person_id not in self.sessions:
            self.sessions[person_id] = {
                'session_start': timestamp,
                'last_seen': timestamp,
                'total_frames': 0,
                'attentive_frames': 0,
                'eyes_closed_frames': 0,
                'head_forward_frames': 0,
            }
        
        session = self.sessions[person_id]
        session['last_seen'] = timestamp
        session['total_frames'] += 1
        
        if is_attentive:
            session['attentive_frames'] += 1
        if eyes_closed_long:
            session['eyes_closed_frames'] += 1
        if head_forward:
            session['head_forward_frames'] += 1
    
    def get_attentiveness_score(self, person_id: int) -> float:
        """
        Calculate attentiveness score (0-100).
        Based on: attentive frames / total frames
        Penalized by excessive eye closure.
        """
        if person_id not in self.sessions:
            return 0.0
        
        session = self.sessions[person_id]
        total = session['total_frames']
        
        if total == 0:
            return 0.0
        
        # Base score: percentage of attentive frames
        base_score = (session['attentive_frames'] / total) * 100
        
        # Penalty for excessive eye closure (excluding normal blinking)
        # If eyes closed > 15% of frames, apply penalty
        eyes_closed_ratio = session['eyes_closed_frames'] / total
        penalty = max(0, (eyes_closed_ratio - 0.15) * 100)
        
        return max(0.0, min(100.0, base_score - penalty))
    
    def save_session(self, person_id: int):
        """Save or update session data in database."""
        if person_id not in self.sessions:
            return
        
        session = self.sessions[person_id]
        score = self.get_attentiveness_score(person_id)
        
        cur = self.db.cursor()
        
        # Check if record already exists for this person and module
        cur.execute("""
            SELECT id, total_frames, attentive_frames, eyes_closed_frames, head_forward_frames 
            FROM attention_records 
            WHERE person_id = ? AND module_name = ?
        """, (person_id, self.module_name))
        
        existing = cur.fetchone()
        
        if existing:
            # Update existing record: accumulate frames
            cur.execute("""
                UPDATE attention_records 
                SET session_end = ?,
                    total_frames = ?,
                    attentive_frames = ?,
                    eyes_closed_frames = ?,
                    head_forward_frames = ?,
                    attentiveness_score = ?
                WHERE person_id = ? AND module_name = ?
            """, (
                session['last_seen'],
                existing[1] + session['total_frames'],
                existing[2] + session['attentive_frames'],
                existing[3] + session['eyes_closed_frames'],
                existing[4] + session['head_forward_frames'],
                score,
                person_id,
                self.module_name
            ))
            print(f"✅ Updated attention record ID {existing[0]} for Person {person_id} in '{self.module_name}' (Score: {score:.1f}%)")
        else:
            # Insert new record
            cur.execute("""
                INSERT INTO attention_records 
                (person_id, module_name, session_start, session_end, total_frames, 
                 attentive_frames, eyes_closed_frames, head_forward_frames, 
                 attentiveness_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                person_id,
                self.module_name,
                session['session_start'],
                session['last_seen'],
                session['total_frames'],
                session['attentive_frames'],
                session['eyes_closed_frames'],
                session['head_forward_frames'],
                score
            ))
            record_id = cur.lastrowid
            print(f"✅ Created NEW attention record ID {record_id} for Person {person_id} in '{self.module_name}' (Score: {score:.1f}%)")
        
        self.db.commit()
    
    def cleanup_inactive_sessions(self, current_time: float, timeout: float = 30.0):
        """Save and remove sessions that haven't been seen recently."""
        to_remove = []
        for pid, session in self.sessions.items():
            if current_time - session['last_seen'] > timeout:
                self.save_session(pid)
                to_remove.append(pid)
        
        for pid in to_remove:
            del self.sessions[pid]
    
    def save_all_sessions(self):
        """Save all active sessions (called on exit)."""
        for pid in list(self.sessions.keys()):
            self.save_session(pid)
        self.sessions.clear()


# ======================
# Geometry + Drawing
# ======================
def bbox_area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou_xyxy(a, b) -> float:
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
# Attention features
# ======================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def ear_from_landmarks(pts, idx):
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-9)


def nose_center_ratio(pts) -> float:
    """
    Nose x offset from eye-midpoint, normalized by eye distance.
    Smaller => more forward.
    """
    nose = pts[1]
    left_eye = pts[33]
    right_eye = pts[263]

    eye_mid_x = 0.5 * (left_eye[0] + right_eye[0])
    eye_dist = abs(right_eye[0] - left_eye[0]) + 1e-6
    offset = abs(nose[0] - eye_mid_x)
    return float(offset / eye_dist)


# ======================
# MAIN
# ======================
def main():
    # DB
    db = sqlite3.connect(SQLITE_PATH)
    init_db(db)
    store = EmbeddingStore(db)
    
    # Ask for module name
    print("\n" + "="*60)
    module_name = input("Enter module name for this session (e.g., 'Machine Learning'): ").strip()
    if not module_name:
        module_name = "Live Session"
    print(f"Tracking attention for: {module_name}")
    print("="*60 + "\n")
    
    attention_tracker = AttentionTracker(db, module_name)

    # Models
    print("Loading models...")
    yolo = YOLO(YOLO_MODEL)

    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=FACE_DET_SIZE)

    # MediaPipe Face Landmarker with new API
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=10,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    face_mesh = mp_vision.FaceLandmarker.create_from_options(options)

    # Attention state (timers need stable identity)
    # state[pid] = {"closed_since": float|None}
    state = {}

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try CAM_INDEX=0/1.")

    frame_i = 0
    prev_time = time.time()
    last_cleanup = time.time()

    print("Running... q=quit, c=clear identities, s=save sessions")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_i += 1
            now = time.time()
            H, W = frame.shape[:2]

            # Cleanup inactive sessions every 10 seconds
            if now - last_cleanup > 10.0:
                attention_tracker.cleanup_inactive_sessions(now)
                last_cleanup = now

            # ----------------------
            # YOLO person detection (count)
            # ----------------------
            y = yolo.predict(frame, conf=PERSON_CONF, classes=[0], verbose=False)[0]
            person_boxes = []
            if y.boxes is not None and len(y.boxes) > 0:
                for b in y.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                    person_boxes.append((x1, y1, x2, y2))

            # ----------------------
            # InsightFace faces
            # ----------------------
            faces_if = face_app.get(frame)
            if_faces = []
            for f in faces_if:
                fx1, fy1, fx2, fy2 = [int(v) for v in f.bbox]
                if bbox_area_xyxy((fx1, fy1, fx2, fy2)) < MIN_FACE_AREA:
                    continue
                if_faces.append({"bbox": (fx1, fy1, fx2, fy2), "emb": f.embedding})
            
            # ----------------------
            # MediaPipe landmarks
            # ----------------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = face_mesh.detect_for_video(mp_image, int(now * 1000))
            
            mp_faces = []
            if res.face_landmarks:
                for fm in res.face_landmarks:
                    pts = np.array([(lm.x * W, lm.y * H) for lm in fm], dtype=np.float32)
                    x1 = int(np.clip(np.min(pts[:, 0]), 0, W - 1))
                    y1 = int(np.clip(np.min(pts[:, 1]), 0, H - 1))
                    x2 = int(np.clip(np.max(pts[:, 0]), 0, W - 1))
                    y2 = int(np.clip(np.max(pts[:, 1]), 0, H - 1))
                    if bbox_area_xyxy((x1, y1, x2, y2)) < MIN_FACE_AREA:
                        continue
                    mp_faces.append({"bbox": (x1, y1, x2, y2), "pts": pts})

            # ----------------------
            # Pair MP faces to IF faces by IoU
            # ----------------------
            matched = []
            used_if = set()

            for mpf in mp_faces:
                best_j = None
                best_iou = 0.0
                for j, iff in enumerate(if_faces):
                    if j in used_if:
                        continue
                    s = iou_xyxy(mpf["bbox"], iff["bbox"])
                    if s > best_iou:
                        best_iou = s
                        best_j = j

                if best_j is not None and (best_iou >= MIN_IOU_MATCH or (len(if_faces) == 1 and len(mp_faces) == 1)):
                    used_if.add(best_j)
                    matched.append((mpf, if_faces[best_j], best_iou))
                else:
                    matched.append((mpf, None, best_iou))

            # ----------------------
            # Identity + Attention
            # ----------------------
            for mpf, iff, best_iou in matched:
                mx1, my1, mx2, my2 = mpf["bbox"]
                pts = mpf["pts"]

                # Attention features
                ear_val = 0.5 * (ear_from_landmarks(pts, LEFT_EYE) + ear_from_landmarks(pts, RIGHT_EYE))
                eyes_closed = ear_val < EAR_CLOSED_THRESH

                ratio = nose_center_ratio(pts)
                head_forward = ratio < NOSE_CENTER_RATIO_THRESH

                pid = None
                sim = None
                was_new = None

                if iff is not None:
                    pid, sim, was_new = store.match_or_create(iff["emb"])

                attentive = None
                eyes_closed_long = False

                if pid is not None:
                    if pid not in state:
                        state[pid] = {"closed_since": None}
                    st = state[pid]

                    # eyes-closed timer
                    if eyes_closed:
                        if st["closed_since"] is None:
                            st["closed_since"] = now
                    else:
                        st["closed_since"] = None

                    eyes_closed_long = (st["closed_since"] is not None) and ((now - st["closed_since"]) >= EYES_CLOSED_LONG_SEC)

                    attentive = 1 if (head_forward and not eyes_closed_long) else 0
                    
                    # Track attention
                    attention_tracker.update(pid, bool(attentive), eyes_closed_long, head_forward, now)
                    current_score = attention_tracker.get_attentiveness_score(pid)

                # Draw face bbox
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (40, 200, 40), 2)

                if pid is None:
                    draw_label(frame, "Person: ?", mx1, my1)
                    draw_label(frame, "Attentive: --", mx1, my1 + 28)
                else:
                    draw_label(frame, f"Person {pid}", mx1, my1)
                    draw_label(frame, f"Attentive: {attentive} (Score: {current_score:.1f}%)", mx1, my1 + 28)

                if SHOW_DEBUG_OVERLAY:
                    draw_label(frame, f"EAR:{ear_val:.3f} closed_long:{int(eyes_closed_long)}", mx1, my1 + 56)
                    draw_label(frame, f"nose_ratio:{ratio:.3f} forward:{int(head_forward)}", mx1, my1 + 84)
                    if sim is not None:
                        draw_label(frame, f"cos:{sim:.2f} iou:{best_iou:.2f}", mx1, my1 + 112)

                if PRINT_DEBUG_EVERY_N_FRAMES and pid is not None and frame_i % PRINT_DEBUG_EVERY_N_FRAMES == 0:
                    print(f"[Person {pid}] cos={sim:.3f} ear={ear_val:.3f} ratio={ratio:.3f} forward={head_forward} closed_long={eyes_closed_long} attentive={attentive}")

            # ----------------------
            # Draw YOLO person boxes
            # ----------------------
            for (x1, y1, x2, y2) in person_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 40), 2)

            # ----------------------
            # HUD
            # ----------------------
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / (dt + 1e-9)

            saved_count = count_people(db)

            draw_label(frame, f"People in frame: {len(person_boxes)}", 10, 30)
            draw_label(frame, f"FPS: {fps:.1f}", 10, 60)
            draw_label(frame, f"Saved identities: {saved_count}", 10, 90)
            draw_label(frame, "q: quit | c: clear DB | s: save sessions", 10, 120)

            cv2.imshow("People + Identity + Attention Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                clear_people(db)
                store._load_all()
                state = {}
                attention_tracker.sessions.clear()
                print(f"Cleared identities from {SQLITE_PATH}")
            if key == ord("s"):
                attention_tracker.save_all_sessions()
                print("Saved all sessions to database")

    finally:
        # Save all active sessions before closing
        print("\nSaving attention data...")
        attention_tracker.save_all_sessions()
        db.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
