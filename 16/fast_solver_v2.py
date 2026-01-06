#!/usr/bin/env python3
"""
Santa 2025 - FAST Solver
========================
Simple and reliable solver that actually works.
"""

import math
import time
import random
import argparse
from typing import List, Tuple, Dict

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# Tree coordinates
TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

BASE_POLYGON = Polygon(TREE_COORDS)

# Pre-compute rotated polygons
ROTATED = {}
for deg in range(0, 360, 5):
    ROTATED[deg] = affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))


def make_tree(x: float, y: float, deg: float) -> Polygon:
    snapped = int(round(deg / 5) * 5) % 360
    poly = ROTATED[snapped]
    if x != 0 or y != 0:
        return affinity.translate(poly, xoff=x, yoff=y)
    return poly


def bbox_side(placements: List[Tuple[float, float, float]]) -> float:
    if not placements:
        return 0.0
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def has_collision(poly: Polygon, others: List[Polygon], buf: float = 0.01) -> bool:
    for o in others:
        if poly.distance(o) < buf:
            return True
    return False


def center_solution(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    if not placements:
        return placements
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]


def place_tree(base_poly: Polygon, existing: List[Polygon], attempts: int = 30) -> Tuple[float, float]:
    """Simple radial placement."""
    if not existing:
        return 0.0, 0.0

    best_x, best_y = 0.0, 0.0
    best_r = float('inf')

    for _ in range(attempts):
        # Random angle (weighted toward corners)
        while True:
            angle = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * angle)):
                break

        vx, vy = math.cos(angle), math.sin(angle)

        # Start far, move in
        for r in np.arange(15.0, 0, -0.2):
            px, py = r * vx, r * vy
            cand = affinity.translate(base_poly, xoff=px, yoff=py)

            if has_collision(cand, existing):
                # Back up
                for r2 in np.arange(r, r + 2, 0.02):
                    px2, py2 = r2 * vx, r2 * vy
                    cand2 = affinity.translate(base_poly, xoff=px2, yoff=py2)
                    if not has_collision(cand2, existing):
                        if r2 < best_r:
                            best_r = r2
                            best_x, best_y = px2, py2
                        break
                break
        else:
            # No collision found - can place at center
            if 0 < best_r:
                best_r = 0
                best_x, best_y = 0, 0

    return best_x, best_y


def simulated_annealing(placements: List[Tuple[float, float, float]],
                        iterations: int = 2000) -> List[Tuple[float, float, float]]:
    """Simple SA refinement."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]

    cur_score = bbox_side(current)
    best = list(current)
    best_score = cur_score

    T = 1.0

    for it in range(iterations):
        i = random.randrange(n)
        x, y, d = current[i]

        # Random move
        if random.random() < 0.7:
            nx = x + random.gauss(0, 0.1 * cur_score)
            ny = y + random.gauss(0, 0.1 * cur_score)
            nd = d
        else:
            nx, ny = x, y
            nd = (d + random.choice([-5, 5, -10, 10, -15, 15])) % 360

        new_poly = make_tree(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if has_collision(new_poly, others):
            T *= 0.999
            continue

        # Check improvement
        old_poly = polys[i]
        polys[i] = new_poly
        new_score = bbox_side([(p[0], p[1], p[2]) if j != i else (nx, ny, nd)
                               for j, p in enumerate(current)])

        delta = new_score - cur_score
        if delta < 0 or (T > 0.01 and random.random() < math.exp(-delta / T)):
            current[i] = (nx, ny, nd)
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best = list(current)
        else:
            polys[i] = old_poly

        T *= 0.999

    return best


def solve_all(max_n: int = 200, sa_iters: int = 2000, attempts: int = 30,
              verbose: bool = True) -> Dict[int, List[Tuple[float, float, float]]]:
    """Solve all puzzles."""
    solutions = {}

    total_start = time.time()

    if verbose:
        print("=" * 60)
        print("SANTA 2025 - FAST SOLVER")
        print("=" * 60)
        print(f"SA iterations: {sa_iters}")
        print(f"Placement attempts: {attempts}")
        print()

    for n in range(1, max_n + 1):
        n_start = time.time()

        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
        else:
            prev = solutions[n - 1]
            prev_polys = [make_tree(x, y, d) for x, y, d in prev]

            # Random rotation for new tree
            angle = random.uniform(0, 360)
            snapped = int(round(angle / 5) * 5) % 360
            base = ROTATED[snapped]

            # Place new tree
            px, py = place_tree(base, prev_polys, attempts)
            sol = prev + [(px, py, float(snapped))]

            # Refine with SA
            sol = simulated_annealing(sol, sa_iters)

        # Center solution
        sol = center_solution(sol)
        solutions[n] = sol

        elapsed = time.time() - n_start

        if verbose and (n <= 10 or n % 10 == 0 or n == max_n):
            side = bbox_side(sol)
            total = sum((bbox_side(solutions[i]) ** 2) / i for i in range(1, n + 1))
            print(f"n={n:3d}: side={side:.4f}, time={elapsed:.1f}s, total={total:.2f}")

    if verbose:
        total_time = time.time() - total_start
        total_score = sum((bbox_side(s) ** 2) / n for n, s in solutions.items())
        print()
        print("=" * 60)
        print(f"DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"Final Score: {total_score:.4f}")
        print("=" * 60)

    return solutions


def save_submission(solutions: Dict[int, List[Tuple[float, float, float]]],
                    output: str = "submission.csv"):
    """Save to CSV."""
    with open(output, "w") as f:
        f.write("id,x,y,deg\n")
        for n in sorted(solutions.keys()):
            pos = solutions[n]
            polys = [make_tree(x, y, d) for x, y, d in pos]
            b = unary_union(polys).bounds
            for idx, (x, y, d) in enumerate(pos):
                f.write(f"{n:03d}_{idx},s{x - b[0]:.6f},s{y - b[1]:.6f},s{d:.6f}\n")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Santa 2025 Fast Solver")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--max-n", type=int, default=200)
    parser.add_argument("--sa-iters", type=int, default=2000, help="SA iterations per puzzle")
    parser.add_argument("--attempts", type=int, default=30, help="Placement attempts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    solutions = solve_all(
        max_n=args.max_n,
        sa_iters=args.sa_iters,
        attempts=args.attempts,
        verbose=True
    )

    save_submission(solutions, args.output)

    total_score = sum((bbox_side(s) ** 2) / n for n, s in solutions.items())
    print(f"Final Score: {total_score:.4f}")


if __name__ == "__main__":
    main()
