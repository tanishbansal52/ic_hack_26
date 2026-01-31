import sqlite3
import time
from config import DB_PATH

conn = sqlite.connect(DB_PATH)
cursor = conn.cursor()

def insertAttInstance(person_features, attScore, time):

    cursor.execute("""
        SELECT student_id FROM students
        WHERE face_encoding = ?
    """, (person_features,))
    StudentID = cursor.fetchone()[0]
    #StudentID should be unique

    cursor.execute("""
        SELECT id FROM lectures
        WHERE ? > start_time
        AND ? < end_time
    """, (time, time,))
    LecID = cursor.fetchone()[0]

    cursor.execute("""
        INSERT INTO attendance_logs (student_id, lecture_id, attentiveness_score)
        VALUES (?, ?, ?)
    """, (StudentID, LecID, attScore))

    cursor.commit()
