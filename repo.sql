-- 1. Users (Students)
CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    year_group INT
);

-- 2. Modules (The hardcoded subjects)
CREATE TABLE modules (
    module_id SERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE, -- e.g., 'DOC101'
    name VARCHAR(100),
    description TEXT
);

-- 3. Lectures (Specific sessions for a module)
CREATE TABLE lectures (
    lecture_id SERIAL PRIMARY KEY,
    module_id INT REFERENCES modules(module_id),
    date DATETIME,
    topic VARCHAR(100)
);

-- 4. Attendance & Attentiveness (The Facial Rec Output)
-- This is where your AI writes data: "Person X was at Lecture Y and was 80% attentive"
CREATE TABLE attendance_logs (
    log_id SERIAL PRIMARY KEY,
    student_id INT REFERENCES students(student_id),
    lecture_id INT REFERENCES lectures(lecture_id),
    is_present BOOLEAN DEFAULT TRUE,
    attentiveness_score FLOAT, -- 0.0 to 1.0 (Derived from facial rec: awake vs sleeping)
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 5. Grades (To calculate the correlation)
CREATE TABLE grades (
    grade_id SERIAL PRIMARY KEY,
    student_id INT REFERENCES students(student_id),
    module_id INT REFERENCES modules(module_id),
    final_score FLOAT -- 0 to 100
);