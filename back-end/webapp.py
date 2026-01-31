import sqlite3
import os
from flask import Flask, jsonify
from flask_cors import CORS
from scipy import stats

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION: PATHS TO YOUR DATABASES ---
# We use absolute paths to be safe, or relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_DB_PATH = os.path.join(BASE_DIR, "../database.db") # The existing one
GRADES_DB_PATH = os.path.join(BASE_DIR, "../grades.db")   # The new one you made

def get_db_connection():
    try:
        # 1. Connect to the main Camera Database
        conn = sqlite3.connect(CAMERA_DB_PATH)
        conn.row_factory = sqlite3.Row
        
        # 2. ATTACH the Grades Database
        # We give it the nickname 'grades_db' so we can use it in queries
        conn.execute(f"ATTACH DATABASE '{GRADES_DB_PATH}' AS grades_db")
        
        return conn
    except Exception as e:
        print(f"Error connecting to DB: {e}")
        return None

@app.route("/")
def home():
    return jsonify({"message": "Backend running with TWO databases attached!"})

@app.route("/api/modules/<int:module_id>/roi", methods=["GET"])
def get_module_roi(module_id):
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()

    # --- STEP 1: Get Total Lectures for this Module ---
    # 'main' refers to the camera database we connected to first
    cur.execute("SELECT COUNT(*) FROM main.lectures WHERE module = ?", (module_id,))
    total_lectures = cur.fetchone()[0]

    if total_lectures == 0:
        conn.close()
        return jsonify({"module_id": module_id, "r_squared": 0, "points": []})

    # --- STEP 2: The Cross-Database Query ---
    # We fetch grades from 'grades_db.grades' 
    # and join them with 'main.camera_data'
    
    query = """
        SELECT 
            g.student_id,
            g.final_score,
            -- Count unique lectures this student attended for this module
            COUNT(DISTINCT CASE WHEN l.module = ? THEN cd.FK_lecture_id END) as attended_count
        FROM grades_db.grades g
        -- Join Camera Data (in main DB)
        LEFT JOIN main.camera_data cd ON g.student_id = cd.FK_person_id
        -- Join Lectures (in main DB)
        LEFT JOIN main.lectures l ON cd.FK_lecture_id = l.id
        WHERE g.module_id = ?
        GROUP BY g.student_id, g.final_score;
    """
    
    try:
        # Pass module_id twice (for the CASE and the WHERE)
        cur.execute(query, (module_id, module_id))
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        conn.close()
        return jsonify({"error": f"SQL Error: {e}"}), 500
    
    conn.close()

    # --- STEP 3: Format Data ---
    attendance_pct_list = []
    grade_list = []
    points = []

    for row in rows:
        attended_count = row["attended_count"]
        final_grade = row["final_score"]

        # Calculate Percentage
        attendance_pct = (attended_count / total_lectures) * 100
        
        attendance_pct_list.append(attendance_pct)
        grade_list.append(final_grade)
        
        points.append({
            "attentiveness": round(attendance_pct, 1),
            "grade": round(final_grade, 1)
        })

    # --- STEP 4: Calculate R^2 ---
    r_squared = 0
    if len(attendance_pct_list) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(attendance_pct_list, grade_list)
        r_squared = r_value ** 2

    return jsonify({
        "module_id": module_id,
        "r_squared": round(r_squared, 2),
        "points": points
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)