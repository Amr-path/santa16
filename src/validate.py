"""
validate.py - Solution validation for Santa 2025
"""
from typing import List, Tuple, Dict
from dataclasses import dataclass

from .geometry import (
    make_tree_polygon, check_all_overlaps,
    compute_bounding_square_side, bounding_square_side_from_polys
)


def validate_solution(
    solution: List[Tuple[float, float, float]],
    n: int
) -> Tuple[bool, str]:
    """Validate a single solution."""
    
    if len(solution) != n:
        return False, f"Expected {n} trees, got {len(solution)}"
    
    if n == 0:
        return True, "OK"
    
    overlaps = check_all_overlaps(solution)
    if overlaps:
        return False, f"{len(overlaps)} overlapping pair(s)"
    
    return True, "OK"


def validate_all_solutions(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200,
    verbose: bool = True
) -> bool:
    """Validate all solutions."""
    
    all_valid = True
    invalid = []
    
    for n in range(1, max_n + 1):
        if n not in solutions:
            all_valid = False
            invalid.append((n, "Missing"))
            continue
        
        valid, msg = validate_solution(solutions[n], n)
        if not valid:
            all_valid = False
            invalid.append((n, msg))
    
    if verbose:
        if all_valid:
            print(f"✓ All {max_n} solutions are valid")
        else:
            print(f"✗ {len(invalid)} invalid solution(s)")
            for n, msg in invalid[:10]:
                print(f"  n={n}: {msg}")
    
    return all_valid


def compute_score(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200
) -> float:
    """Compute total competition score: sum of (side^2 / n)."""
    total = 0.0
    for n in range(1, max_n + 1):
        if n not in solutions:
            continue
        side = compute_bounding_square_side(solutions[n])
        total += (side ** 2) / n
    return total


def print_score_summary(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200
):
    """Print detailed score summary."""
    
    print("=" * 60)
    print("SCORE SUMMARY")
    print("=" * 60)
    
    sides = {}
    contributions = {}
    
    for n in range(1, max_n + 1):
        if n not in solutions:
            continue
        side = compute_bounding_square_side(solutions[n])
        sides[n] = side
        contributions[n] = (side ** 2) / n
    
    total = sum(contributions.values())
    baseline = 157.08  # Official baseline score
    
    print(f"Total score: {total:.4f}")
    print(f"Baseline: {baseline}")
    print(f"Improvement: {(baseline - total) / baseline * 100:.1f}%")
    print()
    
    # Worst contributors
    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 worst score contributions:")
    for n, contrib in sorted_contrib[:10]:
        print(f"  n={n:3d}: side={sides[n]:.4f}, contrib={contrib:.4f}")
    
    print()
    print(f"Side length range: {min(sides.values()):.4f} - {max(sides.values()):.4f}")
    print("=" * 60)
