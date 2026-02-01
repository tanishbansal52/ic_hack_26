"""
Tools for the AI agent to access user and module information.
This uses fake data for now - replace with real database calls later.
"""

from typing import Dict, List, Optional


# Fake database - replace with real database later
FAKE_USER_DB = {
    "user001": {
        "id": "user001",
        "name": "Alice Johnson",
        "email": "alice.johnson@university.edu",
        "year": 2,
        "program": "Computer Science",
        "gpa": 3.7,
        "grades": {
            "CS101": {"grade": "A", "credits": 4, "semester": "Fall 2024"},
            "CS102": {"grade": "A-", "credits": 4, "semester": "Fall 2024"},
            "MATH201": {"grade": "B+", "credits": 3, "semester": "Fall 2024"},
            "CS201": {"grade": "A", "credits": 4, "semester": "Spring 2025"},
            "CS202": {"grade": "B+", "credits": 4, "semester": "Spring 2025"},
        },
        "module_history": ["CS101", "CS102", "MATH201", "CS201", "CS202"]
    },
    "user002": {
        "id": "user002",
        "name": "Bob Smith",
        "email": "bob.smith@university.edu",
        "year": 3,
        "program": "Data Science",
        "gpa": 3.9,
        "grades": {
            "CS101": {"grade": "A", "credits": 4, "semester": "Fall 2023"},
            "MATH201": {"grade": "A", "credits": 3, "semester": "Fall 2023"},
            "DS301": {"grade": "A-", "credits": 4, "semester": "Spring 2024"},
            "CS301": {"grade": "A", "credits": 4, "semester": "Fall 2024"},
            "ML401": {"grade": "A-", "credits": 4, "semester": "Fall 2024"},
        },
        "module_history": ["CS101", "MATH201", "DS301", "CS301", "ML401"]
    }
}

FAKE_MODULE_DB = {
    "CS301": {
        "code": "CS301",
        "name": "Advanced Algorithms",
        "credits": 4,
        "description": "Advanced data structures and algorithm design techniques",
        "prerequisites": ["CS201", "CS202"],
        "available_semesters": ["Fall", "Spring"],
        "capacity": 50,
        "instructor": "Dr. Williams",
        "past_attendance": {
            "average_rate": 0.85,
            "trend": "stable"
        },
        "past_attentiveness": {
            "average_rating": 4.2,
            "engagement_score": 0.78,
            "difficulty_rating": 4.5
        },
        "student_feedback": {
            "overall_rating": 4.3,
            "would_recommend": 0.88
        }
    },
    "CS302": {
        "code": "CS302",
        "name": "Database Systems",
        "credits": 4,
        "description": "Design and implementation of database management systems",
        "prerequisites": ["CS201"],
        "available_semesters": ["Fall", "Spring"],
        "capacity": 45,
        "instructor": "Dr. Chen",
        "past_attendance": {
            "average_rate": 0.92,
            "trend": "increasing"
        },
        "past_attentiveness": {
            "average_rating": 4.5,
            "engagement_score": 0.85,
            "difficulty_rating": 3.8
        },
        "student_feedback": {
            "overall_rating": 4.6,
            "would_recommend": 0.92
        }
    },
    "ML401": {
        "code": "ML401",
        "name": "Machine Learning",
        "credits": 4,
        "description": "Introduction to machine learning algorithms and applications",
        "prerequisites": ["CS201", "MATH201"],
        "available_semesters": ["Fall", "Spring"],
        "capacity": 60,
        "instructor": "Dr. Patel",
        "past_attendance": {
            "average_rate": 0.88,
            "trend": "stable"
        },
        "past_attentiveness": {
            "average_rating": 4.7,
            "engagement_score": 0.90,
            "difficulty_rating": 4.8
        },
        "student_feedback": {
            "overall_rating": 4.8,
            "would_recommend": 0.95
        }
    },
    "DS301": {
        "code": "DS301",
        "name": "Data Analytics",
        "credits": 4,
        "description": "Statistical analysis and visualization of large datasets",
        "prerequisites": ["MATH201"],
        "available_semesters": ["Fall", "Spring"],
        "capacity": 40,
        "instructor": "Dr. Garcia",
        "past_attendance": {
            "average_rate": 0.90,
            "trend": "stable"
        },
        "past_attentiveness": {
            "average_rating": 4.4,
            "engagement_score": 0.82,
            "difficulty_rating": 4.0
        },
        "student_feedback": {
            "overall_rating": 4.5,
            "would_recommend": 0.90
        }
    },
    "CS401": {
        "code": "CS401",
        "name": "Software Engineering",
        "credits": 4,
        "description": "Principles and practices of large-scale software development",
        "prerequisites": ["CS201", "CS202"],
        "available_semesters": ["Fall", "Spring"],
        "capacity": 55,
        "instructor": "Dr. Thompson",
        "past_attendance": {
            "average_rate": 0.87,
            "trend": "stable"
        },
        "past_attentiveness": {
            "average_rating": 4.3,
            "engagement_score": 0.80,
            "difficulty_rating": 4.2
        },
        "student_feedback": {
            "overall_rating": 4.4,
            "would_recommend": 0.89
        }
    }
}


def get_user_information(name: str, year: int, course: str) -> Optional[Dict]:
    """
    Retrieve comprehensive user information including profile, grades, and module history.
    
    Args:
        name: The student's name
        year: The student's year of study (1, 2, 3, or 4)
        course: The student's course/program (e.g., "Computer Science", "Data Science")
        
    Returns:
        Dictionary containing user profile, grades, and module history, or None if user not found
    """
    # Find user by matching name and program
    user_data = None
    for user in FAKE_USER_DB.values():
        if user["name"].lower() == name.lower() and user["program"].lower() == course.lower():
            user_data = user
            break
    
    if not user_data:
        return None
    
    return {
        "profile": {
            "id": user_data["id"],
            "name": user_data["name"],
            "email": user_data["email"],
            "year": user_data["year"],
            "program": user_data["program"],
            "gpa": user_data["gpa"]
        },
        "grades": user_data["grades"],
        "module_history": user_data["module_history"]
    }


def get_user_grades(name: str, year: int, course: str) -> Optional[Dict]:
    """
    Get just the grades for a specific user.
    
    Args:
        name: The student's name
        year: The student's year of study (1, 2, 3, or 4)
        course: The student's course/program (e.g., "Computer Science", "Data Science")
        
    Returns:
        Dictionary of grades or None if user not found
    """
    # Find user by matching name and program
    user_data = None
    for user in FAKE_USER_DB.values():
        if user["name"].lower() == name.lower() and user["program"].lower() == course.lower():
            user_data = user
            break
    return user_data["grades"] if user_data else None


def get_user_profile(name: str, year: int, course: str) -> Optional[Dict]:
    """
    Get just the profile information for a specific user.
    
    Args:
        name: The student's name
        year: The student's year of study (1, 2, 3, or 4)
        course: The student's course/program (e.g., "Computer Science", "Data Science")
        
    Returns:
        Dictionary of profile information or None if user not found
    """
    # Find user by matching name and program
    user_data = None
    for user in FAKE_USER_DB.values():
        if user["name"].lower() == name.lower() and user["program"].lower() == course.lower():
            user_data = user
            break
    
    if not user_data:
        return None
        
    return {
        "id": user_data["id"],
        "name": user_data["name"],
        "email": user_data["email"],
        "year": user_data["year"],
        "program": user_data["program"],
        "gpa": user_data["gpa"]
    }


def get_user_module_history(name: str, year: int, course: str) -> Optional[List[str]]:
    """
    Get the module history for a specific user.
    
    Args:
        name: The student's name
        year: The student's year of study (1, 2, 3, or 4)
        course: The student's course/program (e.g., "Computer Science", "Data Science")
        
    Returns:
        List of module codes or None if user not found
    """
    # Find user by matching name and program
    user_data = None
    for user in FAKE_USER_DB.values():
        if user["name"].lower() == name.lower() and user["program"].lower() == course.lower():
            user_data = user
            break
    return user_data["module_history"] if user_data else None


def get_all_module_choices() -> Dict:
    """
    Get all available module choices with comprehensive information.
    
    Returns:
        Dictionary of all modules with their details including:
        - Basic info (code, name, credits, description)
        - Prerequisites
        - Availability
        - Past attendance statistics
        - Past attentiveness metrics
        - Student feedback
    """
    return FAKE_MODULE_DB


def get_module_info(module_code: str) -> Optional[Dict]:
    """
    Get detailed information about a specific module.
    
    Args:
        module_code: The module code (e.g., "CS301")
        
    Returns:
        Dictionary containing module information or None if module not found
    """
    return FAKE_MODULE_DB.get(module_code)


def get_module_attendance_data(module_code: str) -> Optional[Dict]:
    """
    Get attendance statistics for a specific module.
    
    Args:
        module_code: The module code (e.g., "CS301")
        
    Returns:
        Dictionary containing attendance data or None if module not found
    """
    module = FAKE_MODULE_DB.get(module_code)
    return module["past_attendance"] if module else None


def get_module_attentiveness_data(module_code: str) -> Optional[Dict]:
    """
    Get attentiveness metrics for a specific module.
    
    Args:
        module_code: The module code (e.g., "CS301")
        
    Returns:
        Dictionary containing attentiveness data or None if module not found
    """
    module = FAKE_MODULE_DB.get(module_code)
    return module["past_attentiveness"] if module else None


def get_eligible_modules(name: str, year: int, course: str) -> List[Dict]:
    """
    Get modules that a user is eligible to take based on their module history.
    
    Args:
        name: The student's name
        year: The student's year of study (1, 2, 3, or 4)
        course: The student's course/program (e.g., "Computer Science", "Data Science")
        
    Returns:
        List of eligible modules with their information
    """
    # Find user by matching name and program
    user_data = None
    for user in FAKE_USER_DB.values():
        if user["name"].lower() == name.lower() and user["program"].lower() == course.lower():
            user_data = user
            break
    
    if not user_data:
        return []
    
    completed_modules = set(user_data["module_history"])
    eligible_modules = []
    
    for module_code, module_info in FAKE_MODULE_DB.items():
        # Check if user has already taken this module
        if module_code in completed_modules:
            continue
            
        # Check if user meets prerequisites
        prerequisites = set(module_info.get("prerequisites", []))
        if prerequisites.issubset(completed_modules):
            eligible_modules.append({
                "code": module_code,
                **module_info
            })
    
    return eligible_modules


def search_modules(
    query: str = None,
    min_rating: float = None,
    max_difficulty: float = None,
    instructor: str = None
) -> List[Dict]:
    """
    Search and filter modules based on various criteria.
    
    Args:
        query: Search term to match against module name or description
        min_rating: Minimum overall rating
        max_difficulty: Maximum difficulty rating
        instructor: Filter by instructor name
        
    Returns:
        List of modules matching the criteria
    """
    results = []
    
    for module_code, module_info in FAKE_MODULE_DB.items():
        # Text search
        if query:
            query_lower = query.lower()
            if not (query_lower in module_info["name"].lower() or 
                   query_lower in module_info["description"].lower()):
                continue
        
        # Rating filter
        if min_rating and module_info["student_feedback"]["overall_rating"] < min_rating:
            continue
            
        # Difficulty filter
        if max_difficulty and module_info["past_attentiveness"]["difficulty_rating"] > max_difficulty:
            continue
            
        # Instructor filter
        if instructor and instructor.lower() not in module_info["instructor"].lower():
            continue
        
        results.append({
            "code": module_code,
            **module_info
        })
    
    return results