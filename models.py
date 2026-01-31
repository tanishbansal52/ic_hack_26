from datetime import datetime

class AttendanceRecord:
    def __init__(self, student_id, attentiveness_score):
        self.student_id = student_id
        self.attentiveness_score = attentiveness_score  # 0.0 to 1.0
        # In the future, this is where facial recognition data lands

class Lecture:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.attendance_log = []  # List of AttendanceRecord objects

    def add_attendance(self, student_id, attentiveness_score):
        record = AttendanceRecord(student_id, attentiveness_score)
        self.attendance_log.append(record)

    def get_duration_minutes(self):
        delta = self.end_time - self.start_time
        return delta.total_seconds() / 60

class Module:
    def __init__(self, module_id, name):
        self.module_id = module_id
        self.name = name
        self.lectures = []  # List of Lecture objects
        self.grades = {}    # Dictionary mapping student_id -> final_grade

    def add_lecture(self, start_time, end_time):
        new_lecture = Lecture(start_time, end_time)
        self.lectures.append(new_lecture)

    def add_grade(self, student_id, grade):
        self.grades[student_id] = grade

    # This is the method your Frontend will call later
    def calculate_roi(self):
        # Placeholder logic: Returns a mock correlation score
        # Later, we replace this with actual math using self.lectures and self.grades
        return {
            "module": self.name,
            "correlation_score": 0.75, 
            "message": "High attendance correlates with +15% grade boost."
        }