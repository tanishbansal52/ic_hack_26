import sqlite3
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import json

# Database configuration
DB_PATH = "imperial_students.db"  # Adjust path if needed


def get_modules_list(db_path: str) -> List[str]:
    """Get list of all unique modules in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT module_name FROM module_records WHERE module_name IS NOT NULL")
    modules = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return modules


def get_module_data(db_path: str, module_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch lecture_attendance and grade data for a specific module.
    
    Returns:
        Tuple of (attendance_array, grade_array)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
    SELECT lecture_attendance, grade 
    FROM module_records 
    WHERE module_name = ?
      AND lecture_attendance IS NOT NULL 
      AND grade IS NOT NULL
    """
    
    cursor.execute(query, (module_name,))
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        return np.array([]), np.array([])
    
    attendance = np.array([row[0] for row in results])
    grades = np.array([row[1] for row in results])
    
    return attendance, grades

def calculate_effectiveness_score(attendance: np.ndarray, grades: np.ndarray) -> Dict:
    """
    Calculate effectiveness score for a module based on attendance-grade correlation.
    
    Effectiveness Score Components:
    1. Pearson correlation coefficient (strength of linear relationship)
    2. R-squared value (variance explained)
    3. Statistical significance (p-value)
    4. Effect size (slope of regression)
    
    Returns:
        Dictionary with detailed metrics and overall effectiveness score (0-100)
    """
    if len(attendance) < 3:  # Need at least 3 data points
        return {
            'effectiveness_score': 0,
            'sample_size': len(attendance),
            'error': 'Insufficient data (n < 3)'
        }
    
    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(attendance, grades)
    spearman_r, spearman_p = stats.spearmanr(attendance, grades)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(attendance, grades)
    r_squared = r_value ** 2
    
    # Normalize attendance and grades to 0-1 scale for comparable slope
    att_normalized = (attendance - attendance.min()) / (attendance.max() - attendance.min() + 1e-10)
    grade_normalized = (grades - grades.min()) / (grades.max() - grades.min() + 1e-10)
    
    if len(att_normalized) >= 3:
        slope_norm, _, _, _, _ = stats.linregress(att_normalized, grade_normalized)
    else:
        slope_norm = 0
    
    # Calculate effectiveness score (0-100)
    # Components:
    # 1. Correlation strength (0-40 points): |r| * 40
    # 2. R-squared (0-30 points): R² * 30
    # 3. Statistical significance (0-20 points): 20 if p < 0.05, 10 if p < 0.1, else 0
    # 4. Effect size (0-10 points): normalized slope * 10
    
    correlation_score = abs(pearson_r) * 40
    r_squared_score = r_squared * 30
    
    if pearson_p < 0.05:
        significance_score = 20
    elif pearson_p < 0.1:
        significance_score = 10
    else:
        significance_score = 0
    
    effect_size_score = min(abs(slope_norm) * 10, 10)  # Cap at 10
    
    effectiveness_score = correlation_score + r_squared_score + significance_score + effect_size_score
    
    # Determine effectiveness category
    if effectiveness_score >= 80:
        category = "Highly Effective"
    elif effectiveness_score >= 60:
        category = "Effective"
    elif effectiveness_score >= 40:
        category = "Moderately Effective"
    elif effectiveness_score >= 20:
        category = "Slightly Effective"
    else:
        category = "Not Effective"
    
    return {
        'effectiveness_score': round(effectiveness_score, 2),
        'category': category,
        'pearson_r': round(pearson_r, 4),
        'pearson_p': round(pearson_p, 6),
        'spearman_r': round(spearman_r, 4),
        'r_squared': round(r_squared, 4),
        'slope': round(slope, 4),
        'normalized_slope': round(slope_norm, 4),
        'intercept': round(intercept, 4),
        'sample_size': int(len(attendance)),  # Convert to int
        'is_significant': bool(pearson_p < 0.05),  # Convert to Python bool
        'attendance_mean': round(np.mean(attendance), 2),
        'attendance_std': round(np.std(attendance), 2),
        'grade_mean': round(np.mean(grades), 2),
        'grade_std': round(np.std(grades), 2),
        'score_breakdown': {
            'correlation_strength': round(correlation_score, 2),
            'variance_explained': round(r_squared_score, 2),
            'statistical_significance': round(significance_score, 2),
            'effect_size': round(effect_size_score, 2)
        }
    }

def analyze_all_modules(db_path: str) -> Dict[str, Dict]:
    """Analyze effectiveness for all modules in the database."""
    modules = get_modules_list(db_path)
    results = {}
    
    for module in modules:
        attendance, grades = get_module_data(db_path, module)
        if len(attendance) > 0:
            results[module] = calculate_effectiveness_score(attendance, grades)
    
    return results


def print_effectiveness_report(results: Dict[str, Dict]):
    """Print formatted effectiveness report for all modules."""
    print("\n" + "="*80)
    print("MODULE LECTURE EFFECTIVENESS ANALYSIS")
    print("="*80)
    
    # Sort modules by effectiveness score
    sorted_modules = sorted(results.items(), key=lambda x: x[1].get('effectiveness_score', 0), reverse=True)
    
    print(f"\n📚 Total Modules Analyzed: {len(sorted_modules)}")
    print("\n" + "-"*80)
    
    for rank, (module_name, stats) in enumerate(sorted_modules, 1):
        if 'error' in stats:
            print(f"\n#{rank}. {module_name}")
            print(f"   ⚠️  {stats['error']}")
            continue
        
        score = stats['effectiveness_score']
        category = stats['category']
        
        # Determine emoji based on score
        if score >= 80:
            emoji = "🌟"
        elif score >= 60:
            emoji = "✅"
        elif score >= 40:
            emoji = "⚡"
        elif score >= 20:
            emoji = "⚠️"
        else:
            emoji = "❌"
        
        print(f"\n#{rank}. {emoji} {module_name}")
        print(f"   {'─'*76}")
        print(f"   Effectiveness Score: {score:.2f}/100 ({category})")
        print(f"   Sample Size: {stats['sample_size']} students")
        print(f"   Correlation: r = {stats['pearson_r']:.4f} " + 
              f"({'significant' if stats['is_significant'] else 'not significant'}, p = {stats['pearson_p']:.4f})")
        print(f"   R-squared: {stats['r_squared']:.4f} ({stats['r_squared']*100:.1f}% of grade variance explained)")
        print(f"   Effect: {stats['slope']:.4f} grade points per unit attendance")
        
        print(f"\n   Score Breakdown:")
        print(f"     • Correlation Strength:      {stats['score_breakdown']['correlation_strength']:.2f}/40")
        print(f"     • Variance Explained (R²):   {stats['score_breakdown']['variance_explained']:.2f}/30")
        print(f"     • Statistical Significance:  {stats['score_breakdown']['statistical_significance']:.2f}/20")
        print(f"     • Effect Size:               {stats['score_breakdown']['effect_size']:.2f}/10")
        
        print(f"\n   Descriptive Stats:")
        print(f"     • Avg Attendance: {stats['attendance_mean']:.2f} ± {stats['attendance_std']:.2f}")
        print(f"     • Avg Grade:      {stats['grade_mean']:.2f} ± {stats['grade_std']:.2f}")
    
    print("\n" + "="*80)
    print("\n📊 SUMMARY STATISTICS:")
    
    valid_scores = [s['effectiveness_score'] for s in results.values() if 'effectiveness_score' in s]
    if valid_scores:
        print(f"   Average Effectiveness Score: {np.mean(valid_scores):.2f}/100")
        print(f"   Median Effectiveness Score:  {np.median(valid_scores):.2f}/100")
        print(f"   Highest Score:               {max(valid_scores):.2f}/100")
        print(f"   Lowest Score:                {min(valid_scores):.2f}/100")
        
        # Count by category
        categories = {}
        for stats in results.values():
            if 'category' in stats:
                cat = stats['category']
                categories[cat] = categories.get(cat, 0) + 1
        
        print("\n   Distribution by Category:")
        for cat in ["Highly Effective", "Effective", "Moderately Effective", "Slightly Effective", "Not Effective"]:
            if cat in categories:
                print(f"     • {cat}: {categories[cat]} module(s)")
    
    print("\n" + "="*80 + "\n")


def save_results_json(results: Dict[str, Dict], output_path: str = "module_effectiveness.json"):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {output_path}")


def main():
    """Main analysis function."""
    try:
        print("Analyzing module lecture effectiveness...")
        results = analyze_all_modules(DB_PATH)
        
        # Print report
        print_effectiveness_report(results)
        
        # Save to JSON
        save_results_json(results)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()