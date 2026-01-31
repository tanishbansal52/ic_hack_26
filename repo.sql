DROP TABLE IF EXISTS attendance_logs CASCADE;
DROP TABLE IF EXISTS grades CASCADE;
DROP TABLE IF EXISTS lectures CASCADE;
DROP TABLE IF EXISTS modules CASCADE;
DROP TABLE IF EXISTS students CASCADE;

-- 1. Students (Biometric Data Only)
CREATE TABLE students (
    student_id INT PRIMARY KEY, -- CID
    face_encoding FLOAT[]       -- The 128-float vector from facial recognition
);

-- 2. Modules
CREATE TABLE modules (
    module_id SERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE,
    name VARCHAR(100)
);

-- 3. Lectures
CREATE TABLE lectures (
    lecture_id SERIAL PRIMARY KEY,
    module_id INT REFERENCES modules(module_id),
    start_time TIMESTAMP,
    end_time TIMESTAMP
);

-- 4. Attendance Logs (Links Face ID -> Lecture)
CREATE TABLE attendance_logs (
    log_id SERIAL PRIMARY KEY,
    student_id INT REFERENCES students(student_id),
    lecture_id INT REFERENCES lectures(lecture_id),
    attentiveness_score FLOAT, 
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- timestamp tells you exactly when during that class they were scanned. can be ignored for now.

-- 5. Grades
CREATE TABLE grades (
    grade_id SERIAL PRIMARY KEY,
    student_id INT REFERENCES students(student_id),
    module_id INT REFERENCES modules(module_id),
    final_score FLOAT
);