import json
from typing import Dict, Optional, List

EFFECTIVENESS_DB_PATH = "module_effectiveness.json"


def load_module_effectiveness() -> Dict:
    """Load module effectiveness data from JSON file."""
    try:
        with open(EFFECTIVENESS_DB_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Module effectiveness database not found at {EFFECTIVENESS_DB_PATH}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in module effectiveness database")


def get_module_info(module_name: str) -> Optional[Dict]:
    """Get effectiveness information for a specific module."""
    data = load_module_effectiveness()
    return data.get(module_name)


def get_all_modules() -> List[str]:
    """Get list of all module names."""
    data = load_module_effectiveness()
    return list(data.keys())


def get_top_effective_modules(n: int = 5) -> List[tuple]:
    """
    Get top N most effective modules.
    
    Returns:
        List of tuples (module_name, effectiveness_score)
    """
    data = load_module_effectiveness()
    sorted_modules = sorted(
        data.items(),
        key=lambda x: x[1].get('effectiveness_score', 0),
        reverse=True
    )
    return [(name, info['effectiveness_score']) for name, info in sorted_modules[:n]]


def get_modules_by_category(category: str) -> List[str]:
    """
    Get modules by effectiveness category.
    
    Categories: "Highly Effective", "Effective", "Moderately Effective", 
                "Slightly Effective", "Not Effective"
    """
    data = load_module_effectiveness()
    return [
        name for name, info in data.items()
        if info.get('category') == category
    ]


def get_module_summary(module_name: str) -> str:
    """Get a formatted summary of a module's effectiveness."""
    info = get_module_info(module_name)
    
    if not info:
        return f"Module '{module_name}' not found in database."
    
    if 'error' in info:
        return f"{module_name}: {info['error']}"
    
    return f"""
📚 {module_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Effectiveness Score: {info['effectiveness_score']:.2f}/100
Category: {info['category']}
Sample Size: {info['sample_size']} students

📊 Key Metrics:
• Correlation (r): {info['pearson_r']:.4f}
• Variance Explained (R²): {info['r_squared']:.4f} ({info['r_squared']*100:.1f}%)
• Statistical Significance: {'Yes' if info['is_significant'] else 'No'} (p={info['pearson_p']:.4f})
• Effect: {info['slope']:.4f} grade points per attendance unit

📈 Statistics:
• Avg Attendance: {info['attendance_mean']:.2f}% ± {info['attendance_std']:.2f}%
• Avg Grade: {info['grade_mean']:.2f} ± {info['grade_std']:.2f}

💡 Interpretation:
For every 10% increase in lecture attendance, the grade changes by approximately {info['slope']*10:.2f} points.
"""


def compare_modules(module_names: List[str]) -> str:
    """Compare effectiveness of multiple modules."""
    data = load_module_effectiveness()
    
    comparisons = []
    for name in module_names:
        if name in data:
            info = data[name]
            comparisons.append({
                'name': name,
                'score': info.get('effectiveness_score', 0),
                'correlation': info.get('pearson_r', 0),
                'sample_size': info.get('sample_size', 0)
            })
    
    if not comparisons:
        return "No valid modules found for comparison."
    
    # Sort by effectiveness score
    comparisons.sort(key=lambda x: x['score'], reverse=True)
    
    result = "\n📊 MODULE COMPARISON\n" + "="*60 + "\n"
    for i, module in enumerate(comparisons, 1):
        result += f"\n{i}. {module['name']}\n"
        result += f"   Effectiveness: {module['score']:.2f}/100\n"
        result += f"   Correlation: {module['correlation']:.4f}\n"
        result += f"   Students: {module['sample_size']}\n"
    
    return result


def get_effectiveness_statistics() -> Dict:
    """Get overall statistics across all modules."""
    data = load_module_effectiveness()
    
    scores = [info['effectiveness_score'] for info in data.values() if 'effectiveness_score' in info]
    correlations = [info['pearson_r'] for info in data.values() if 'pearson_r' in info]
    
    categories = {}
    for info in data.values():
        if 'category' in info:
            cat = info['category']
            categories[cat] = categories.get(cat, 0) + 1
    
    return {
        'total_modules': len(data),
        'avg_effectiveness': sum(scores) / len(scores) if scores else 0,
        'median_effectiveness': sorted(scores)[len(scores)//2] if scores else 0,
        'avg_correlation': sum(correlations) / len(correlations) if correlations else 0,
        'category_distribution': categories
    }


# Example usage
if __name__ == "__main__":
    # Test the functions
    print("=== All Modules ===")
    print(get_all_modules())
    
    print("\n=== Top 5 Most Effective ===")
    for name, score in get_top_effective_modules(5):
        print(f"{name}: {score:.2f}")
    
    print("\n=== Module Summary: Mathematical Methods ===")
    print(get_module_summary("Mathematical Methods"))
    
    print("\n=== Overall Statistics ===")
    stats = get_effectiveness_statistics()
    print(f"Total Modules: {stats['total_modules']}")
    print(f"Avg Effectiveness: {stats['avg_effectiveness']:.2f}")
    print(f"Avg Correlation: {stats['avg_correlation']:.4f}")
    print(f"Category Distribution: {stats['category_distribution']}")