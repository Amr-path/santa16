"""
io_utils.py - I/O utilities for Santa 2025

Matches EXACT format from official Kaggle starter:
- id format: NNN_I (3-digit n, tree index)
- values: sX.XXXXXX (s prefix, 6 decimals)
- coordinates shifted so min x,y at origin
"""
import os
from typing import List, Tuple, Dict, Optional

from shapely.ops import unary_union
from .geometry import make_tree_polygon


def find_data_path() -> str:
    """Find competition data directory."""
    candidates = [
        "/kaggle/input/santa-2025",
        "./data",
        "../data",
        "."
    ]
    for path in candidates:
        if os.path.isdir(path):
            sample = os.path.join(path, "sample_submission.csv")
            if os.path.isfile(sample):
                return path
    return "."


def get_output_path(filename: str = "submission.csv") -> str:
    """Get output path."""
    if os.path.isdir("/kaggle/working"):
        return f"/kaggle/working/{filename}"
    return filename


def create_submission(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    output_path: Optional[str] = None,
    decimals: int = 6
) -> str:
    """
    Create submission CSV matching official format EXACTLY.
    
    Format from official code:
    - Header: id,x,y,deg
    - id: NNN_I format (3-digit padded n, underscore, tree index)
    - Values: 's' + float with 6 decimals
    - Coordinates shifted so polygon bounds start at (0, 0)
    """
    if output_path is None:
        output_path = get_output_path()
    
    with open(output_path, "w") as f:
        f.write("id,x,y,deg\n")
        
        for n in range(1, 201):
            if n not in solutions:
                raise ValueError(f"Missing solution for n={n}")
            
            positions = solutions[n]
            if len(positions) != n:
                raise ValueError(f"Wrong count for n={n}")
            
            # Get bounds to shift coordinates (like official code)
            polys = [make_tree_polygon(x, y, d) for x, y, d in positions]
            bounds = unary_union(polys).bounds
            min_x, min_y = bounds[0], bounds[1]
            
            for idx, (x, y, deg) in enumerate(positions):
                # Shift coordinates so min is at 0
                X = x - min_x
                Y = y - min_y
                
                # Format: s prefix + 6 decimals (matching official)
                f.write(f"{n:03d}_{idx},s{X:.{decimals}f},s{Y:.{decimals}f},s{deg:.{decimals}f}\n")
    
    return output_path


def validate_submission_format(path: str) -> Tuple[bool, Optional[str]]:
    """Validate submission CSV format."""
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        return False, f"Cannot read: {e}"
    
    if not lines:
        return False, "Empty file"
    
    # Check header
    if lines[0].strip() != "id,x,y,deg":
        return False, f"Wrong header: {lines[0].strip()}"
    
    # Check row count (sum 1..200 = 20100)
    expected = 200 * 201 // 2
    actual = len(lines) - 1
    if actual != expected:
        return False, f"Wrong rows: expected {expected}, got {actual}"
    
    return True, None


def print_solution_summary(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200
):
    """Print solution summary."""
    from .validate import compute_score
    from .geometry import compute_bounding_square_side
    
    print("=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)
    
    n_solutions = len([n for n in range(1, max_n + 1) if n in solutions])
    print(f"Solutions: {n_solutions}/{max_n}")
    
    if n_solutions > 0:
        sides = {n: compute_bounding_square_side(solutions[n]) 
                 for n in range(1, max_n + 1) if n in solutions}
        
        print(f"Side range: {min(sides.values()):.4f} - {max(sides.values()):.4f}")
        
        total = compute_score(solutions, max_n)
        print(f"Total score: {total:.4f}")
        print(f"Baseline: 157.08")
        if total < 157.08:
            print(f"Improvement: {(157.08 - total) / 157.08 * 100:.1f}%")
    
    print("=" * 60)
