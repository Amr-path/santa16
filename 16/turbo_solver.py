#!/usr/bin/env python3
"""
Santa 2025 - TURBO Optimization Solver
========================================

Ultra-fast with approximations:
- Fast approximate bounding box (no unary_union in hot path)
- Minimal shapely calls
- Efficient SA (500 iterations)
- ~0.5-1 second per puzzle

Usage:
    python turbo_solver.py [--output submission.csv]
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

# =============================================================================
# GEOMETRY - Pre-computed
# =============================================================================

TREE_COORDS = np.array([
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
])

# Pre-compute rotated coordinates for all angles (every 5 degrees)
ROTATED_COORDS = {}
for deg in range(0, 360, 5):
    rad = math.radians(deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rotated = np.zeros_like(TREE_COORDS)
    for i, (x, y) in enumerate(TREE_COORDS):
        rotated[i] = (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
    ROTATED_COORDS[deg] = rotated

# Pre-compute bounds for each rotation
ROTATED_BOUNDS = {}  # {deg: (min_x, min_y, max_x, max_y)}
for deg, coords in ROTATED_COORDS.items():
    ROTATED_BOUNDS[deg] = (coords[:, 0].min(), coords[:, 1].min(),
                           coords[:, 0].max(), coords[:, 1].max())

BASE_POLYGON = Polygon(TREE_COORDS)
ROTATED_POLYGONS = {deg: Polygon(coords) for deg, coords in ROTATED_COORDS.items()}


def snap_angle(deg: float) -> int:
    return int(round(deg / 5) * 5) % 360


def get_bounds(x: float, y: float, deg: float) -> Tuple[float, float, float, float]:
    """Fast bounds calculation without creating polygon."""
    snapped = snap_angle(deg)
    b = ROTATED_BOUNDS[snapped]
    return (b[0] + x, b[1] + y, b[2] + x, b[3] + y)


def get_poly(x: float, y: float, deg: float) -> Polygon:
    """Get polygon (for collision checks only)."""
    snapped = snap_angle(deg)
    poly = ROTATED_POLYGONS[snapped]
    return affinity.translate(poly, xoff=x, yoff=y) if x != 0 or y != 0 else poly


def fast_bbox_side(placements: List[Tuple[float, float, float]]) -> float:
    """Fast bounding box calculation."""
    if not placements:
        return 0.0

    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for x, y, d in placements:
        b = get_bounds(x, y, d)
        min_x = min(min_x, b[0])
        min_y = min(min_y, b[1])
        max_x = max(max_x, b[2])
        max_y = max(max_y, b[3])

    return max(max_x - min_x, max_y - min_y)


def precise_bbox_side(placements: List[Tuple[float, float, float]]) -> float:
    """Precise bounding box using shapely."""
    if not placements:
        return 0.0
    polys = [get_poly(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def collides(poly: Polygon, others: List[Polygon], buf: float) -> bool:
    for o in others:
        if poly.distance(o) < buf:
            return True
    return False


def center(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    if not placements:
        return placements

    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    for x, y, d in placements:
        b = get_bounds(x, y, d)
        min_x = min(min_x, b[0])
        min_y = min(min_y, b[1])
        max_x = max(max_x, b[2])
        max_y = max(max_y, b[3])

    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]


def has_overlaps(placements: List[Tuple[float, float, float]], buf: float = 0.0) -> bool:
    polys = [get_poly(x, y, d) for x, y, d in placements]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].distance(polys[j]) < buf:
                return True
    return False


# =============================================================================
# PLACEMENT - Minimal
# =============================================================================

def weighted_angle() -> float:
    while True:
        a = random.uniform(0, 2 * math.pi)
        if random.random() < abs(math.sin(2 * a)):
            return a


def place(existing: List[Polygon], buf: float, base_deg: int, attempts: int = 50) -> Tuple[float, float]:
    if not existing:
        return 0.0, 0.0

    idx = STRtree(existing)
    base = ROTATED_POLYGONS[base_deg]

    best = (0.0, 0.0)
    min_r = float('inf')

    for _ in range(attempts):
        a = weighted_angle()
        vx, vy = math.cos(a), math.sin(a)

        r = 12.0
        while r >= 0:
            px, py = r * vx, r * vy
            cand = affinity.translate(base, xoff=px, yoff=py)

            # Quick spatial check
            candidates = idx.query(cand.buffer(buf))
            collision = False
            for ci in candidates:
                if cand.distance(existing[ci]) < buf:
                    collision = True
                    break

            if collision:
                break
            r -= 0.15

        # Back up
        while r < 25:
            r += 0.02
            px, py = r * vx, r * vy
            cand = affinity.translate(base, xoff=px, yoff=py)

            candidates = idx.query(cand.buffer(buf))
            collision = False
            for ci in candidates:
                if cand.distance(existing[ci]) < buf:
                    collision = True
                    break

            if not collision:
                break

        if r < min_r:
            min_r = r
            best = (px, py)

    return best


# =============================================================================
# SIMULATED ANNEALING - Ultra Fast
# =============================================================================

def sa(placements: List[Tuple[float, float, float]], buf: float, iterations: int = 500) -> List[Tuple[float, float, float]]:
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [get_poly(x, y, d) for x, y, d in current]

    cur_score = fast_bbox_side(current)
    best_score = cur_score
    best = list(current)

    T = 1.0
    shift = 0.03 * cur_score

    for it in range(iterations):
        i = random.randrange(n)
        x, y, d = current[i]

        r = random.random()
        if r < 0.5:
            nx = x + random.gauss(0, shift)
            ny = y + random.gauss(0, shift)
            nd = d
        elif r < 0.75:
            nx, ny = x, y
            nd = (d + random.choice([-5, 5, -10, 10, -30, 30])) % 360
        else:
            # Center-seeking
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 0.001
            step = random.uniform(0.01, 0.05) * cur_score
            nx = x + step * dx / dist
            ny = y + step * dy / dist
            nd = d

        new_poly = get_poly(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if collides(new_poly, others, buf):
            continue

        # Check score improvement using fast bbox
        old_placement = current[i]
        current[i] = (nx, ny, nd)
        new_score = fast_bbox_side(current)

        delta = new_score - cur_score
        if delta <= 0 or random.random() < math.exp(-delta / T):
            polys[i] = new_poly
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best = list(current)
        else:
            current[i] = old_placement

        T *= 0.995

    return best


# =============================================================================
# COMPACT - Fast
# =============================================================================

def compact(placements: List[Tuple[float, float, float]], buf: float, passes: int = 3) -> List[Tuple[float, float, float]]:
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)

    for _ in range(passes):
        polys = [get_poly(x, y, d) for x, y, d in current]

        cx = sum(p[0] for p in current) / n
        cy = sum(p[1] for p in current) / n

        for i in range(n):
            x, y, d = current[i]
            dist = math.sqrt((x-cx)**2 + (y-cy)**2)
            if dist < 0.05:
                continue

            dx, dy = cx - x, cy - y
            norm = dist + 0.001

            for mult in [0.1, 0.03]:
                move = mult * dist
                nx = x + move * dx / norm
                ny = y + move * dy / norm

                new_poly = get_poly(nx, ny, d)
                others = polys[:i] + polys[i+1:]

                if not collides(new_poly, others, buf):
                    current[i] = (nx, ny, d)
                    polys[i] = new_poly
                    break

    return current


# =============================================================================
# SOLVER
# =============================================================================

class TurboSolver:
    def __init__(self, seed: int = 42, restarts: int = 10, buf: float = 0.018,
                 sa_iters: int = 500, compact_passes: int = 3):
        self.seed = seed
        self.restarts = restarts
        self.buf = buf
        self.sa_iters = sa_iters
        self.compact_passes = compact_passes
        random.seed(seed)
        np.random.seed(seed)

        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.scores: Dict[int, float] = {}

    def solve_one(self, n: int, prev: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float, float]], float]:
        if n <= 0:
            return [], 0.0
        if n == 1:
            return [(0.0, 0.0, 0.0)], 1.0

        prev_polys = [get_poly(x, y, d) for x, y, d in prev]

        best_sol = None
        best_score = float('inf')

        for r in range(self.restarts):
            random.seed(self.seed + r * 1000 + n)

            angle = (r * 30) % 360  # Try different angles
            snapped = snap_angle(angle)

            bx, by = place(prev_polys, self.buf, snapped, attempts=30 + r * 5)
            sol = prev + [(bx, by, float(snapped))]

            # Perturb from best
            if r > 0 and best_sol is not None and r % 2 == 0:
                sol = list(best_sol)
                num_perturb = max(1, n // 15)
                for idx in random.sample(range(n), num_perturb):
                    x, y, d = sol[idx]
                    sol[idx] = (
                        x + random.gauss(0, 0.01 * best_score),
                        y + random.gauss(0, 0.01 * best_score),
                        (d + random.choice([-5, 5])) % 360
                    )

            if has_overlaps(sol, self.buf):
                continue

            sol = sa(sol, self.buf, iterations=self.sa_iters + r * 30)
            sol = compact(sol, self.buf, passes=self.compact_passes)

            score = fast_bbox_side(sol)
            if not has_overlaps(sol, self.buf) and score < best_score:
                best_score = score
                best_sol = sol

        if best_sol is None:
            angle = random.uniform(0, 360)
            snapped = snap_angle(angle)
            bx, by = place(prev_polys, self.buf, snapped, attempts=100)
            best_sol = prev + [(bx, by, float(snapped))]
            best_score = fast_bbox_side(best_sol)

        # Use precise bbox for final score
        final_score = precise_bbox_side(best_sol)
        return center(best_sol), final_score

    def solve_all(self, max_n: int = 200, verbose: bool = True):
        start = time.time()

        if verbose:
            print("=" * 60)
            print("SANTA 2025 - TURBO SOLVER")
            print("=" * 60)
            print(f"Restarts: {self.restarts}, Buffer: {self.buf}")
            print()

        for n in range(1, max_n + 1):
            ns = time.time()
            prev = self.solutions[n - 1] if n > 1 else []
            sol, score = self.solve_one(n, prev)
            self.solutions[n] = sol
            self.scores[n] = score
            et = time.time() - ns

            if verbose and (n <= 10 or n % 10 == 0 or n == max_n):
                total = sum((self.scores[i] ** 2) / i for i in range(1, n + 1))
                print(f"n={n:3d}: side={score:.4f}, time={et:.1f}s, total={total:.2f}")

        if verbose:
            tt = time.time() - start
            total = sum((self.scores[i] ** 2) / i for i in range(1, max_n + 1))
            print()
            print("=" * 60)
            print(f"COMPLETE - Time: {tt:.0f}s ({tt/60:.1f} min)")
            print(f"Final Score: {total:.4f}")
            print("=" * 60)

        return self.solutions

    def total_score(self) -> float:
        return sum((self.scores[n] ** 2) / n for n in self.solutions)


# =============================================================================
# I/O
# =============================================================================

def save(solutions: Dict[int, List[Tuple[float, float, float]]], path: str):
    with open(path, "w") as f:
        f.write("id,x,y,deg\n")
        for n in range(1, 201):
            pos = solutions[n]
            polys = [get_poly(x, y, d) for x, y, d in pos]
            b = unary_union(polys).bounds
            for idx, (x, y, d) in enumerate(pos):
                f.write(f"{n:03d}_{idx},s{x - b[0]:.6f},s{y - b[1]:.6f},s{d:.6f}\n")


def validate(solutions):
    valid = True
    for n in range(1, 201):
        if n not in solutions or len(solutions[n]) != n:
            print(f"  n={n}: Invalid")
            valid = False
        elif has_overlaps(solutions[n]):
            print(f"  n={n}: Overlaps")
            valid = False
    return valid


def summary(solutions):
    print("=" * 60)
    sides = {}
    contribs = {}
    for n, sol in solutions.items():
        side = precise_bbox_side(sol)
        sides[n] = side
        contribs[n] = (side ** 2) / n

    total = sum(contribs.values())
    print(f"Total: {total:.4f}")
    print(f"Baseline: 157.08, Improvement: {(157.08-total)/157.08*100:.1f}%")

    worst = sorted(contribs.items(), key=lambda x: -x[1])[:5]
    print("\nWorst contributions:")
    for n, c in worst:
        print(f"  n={n}: {c:.4f}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="turbo_submission.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--boost", action="store_true", help="Aggressive mode for better score")
    parser.add_argument("--extreme", action="store_true", help="Extreme mode targeting <60")
    args = parser.parse_args()

    restarts = args.restarts
    buf = 0.018
    sa_iters = 500
    compact_passes = 3

    if args.extreme:
        restarts = 100
        buf = 0.012
        sa_iters = 2000
        compact_passes = 8
        print("EXTREME MODE: 100 restarts, 2000 SA iters, tighter buffer")
    elif args.boost:
        restarts = 50
        buf = 0.015
        sa_iters = 1000
        compact_passes = 5
        print("BOOST MODE: 50 restarts, 1000 SA iters, tighter buffer")

    solver = TurboSolver(seed=args.seed, restarts=restarts, buf=buf,
                         sa_iters=sa_iters, compact_passes=compact_passes)
    solutions = solver.solve_all(max_n=200, verbose=True)

    print("\nValidating...")
    if validate(solutions):
        print("All valid!")

    summary(solutions)

    print(f"\nSaving: {args.output}")
    save(solutions, args.output)
    print(f"Final: {solver.total_score():.4f}")


if __name__ == "__main__":
    main()
