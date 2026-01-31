-- 1. Insert Students (Using dummy zeros for face vectors for now)
-- We use array literal syntax '{0,0...}' for the float[] column
INSERT INTO students (student_id, face_encoding) VALUES 
(101, '{0.1, 0.2, 0.3}'), -- The "Good" Student
(102, '{0.4, 0.5, 0.6}'), -- The "Average" Student
(103, '{0.7, 0.8, 0.9}'); -- The "Slacker" Student
-- need to replace with actual 128-d float arrays input from facial recognition system

-- 2. Insert Modules
INSERT INTO modules (module_id, code, name) VALUES 
(1, 'DOC601', 'Machine Learning'),
(2, 'DOC602', 'Computer Graphics'),
(3, 'DOC404', 'Professional Skills');

-- 3. Insert Lectures (Timestamps)
-- Machine Learning Lectures (Hard Module)
INSERT INTO lectures (lecture_id, module_id, start_time, end_time) VALUES 
(10, 1, '2023-10-01 09:00:00', '2023-10-01 11:00:00'), -- ML Lecture 1
(11, 1, '2023-10-08 09:00:00', '2023-10-08 11:00:00'), -- ML Lecture 2
(12, 1, '2023-10-15 09:00:00', '2023-10-15 11:00:00'); -- ML Lecture 3

-- Professional Skills Lectures (Easy Module)
INSERT INTO lectures (lecture_id, module_id, start_time, end_time) VALUES 
(20, 3, '2023-10-02 14:00:00', '2023-10-02 15:00:00');

-- 4. Insert Grades (The Outcome)
-- ML: Correlation is high (Good grades = Good attendance)
INSERT INTO grades (student_id, module_id, final_score) VALUES 
(101, 1, 92.5), -- Student 101 Aced ML
(102, 1, 65.0), -- Student 102 Scraped by
(103, 1, 38.0); -- Student 103 Failed

-- Pro Skills: Correlation is low (Everyone does okay)
INSERT INTO grades (student_id, module_id, final_score) VALUES 
(101, 3, 85.0),
(103, 3, 82.0); -- Even the slacker got an 82

-- 5. Insert Attendance Logs (The Input)
-- Student 101: Attended everything, High Attention (0.9)
INSERT INTO attendance_logs (student_id, lecture_id, attentiveness_score) VALUES 
(101, 10, 0.95),
(101, 11, 0.90),
(101, 12, 0.88);

-- Student 102: Missed one lecture, Medium Attention (0.5)
INSERT INTO attendance_logs (student_id, lecture_id, attentiveness_score) VALUES 
(102, 10, 0.50),
(102, 12, 0.45); -- Missed lecture 11

-- Student 103: Attended only one, Sleeping (0.1)
INSERT INTO attendance_logs (student_id, lecture_id, attentiveness_score) VALUES 
(103, 10, 0.15);