#!/usr/bin/env python3
"""
Santa 2025 - QUICK Optimization Solver
=======================================

Highly optimized for speed while maintaining good quality:
- Minimal restarts (15) with smart selection
- Efficient SA with reduced iterations (2000)
- Fast local search (100 iterations)
- Light compacting
- ~1-2 seconds per puzzle

Usage:
    python quick_solver.py [--output submission.csv]
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
# GEOMETRY - Optimized with caching
# =============================================================================

TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

BASE_POLYGON = Polygon(TREE_COORDS)
ROTATED_POLYGONS = {deg: affinity.rotate(BASE_POLYGON, deg, origin=(0, 0)) for deg in range(0, 360, 5)}


def make_tree(x: float, y: float, deg: float) -> Polygon:
    snapped = int(round(deg / 5) * 5) % 360
    poly = ROTATED_POLYGONS[snapped]
    return affinity.translate(poly, xoff=x, yoff=y) if x != 0 or y != 0 else poly


def collides_simple(poly: Polygon, others: List[Polygon], buf: float) -> bool:
    """Simple collision check - fast for small lists."""
    for o in others:
        if poly.distance(o) < buf:
            return True
    return False


def collides_indexed(poly: Polygon, idx: STRtree, polys: List[Polygon], buf: float) -> bool:
    """Indexed collision check for larger lists."""
    for i in idx.query(poly.buffer(buf)):
        if poly.distance(polys[i]) < buf:
            return True
    return False


def bbox_side(placements: List[Tuple[float, float, float]]) -> float:
    if not placements:
        return 0.0
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def bbox_polys(polys: List[Polygon]) -> float:
    if not polys:
        return 0.0
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def center(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    if not placements:
        return placements
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]


def has_overlaps(placements: List[Tuple[float, float, float]], buf: float = 0.0) -> bool:
    polys = [make_tree(x, y, d) for x, y, d in placements]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].distance(polys[j]) < buf:
                return True
    return False


# =============================================================================
# PLACEMENT
# =============================================================================

def weighted_angle() -> float:
    while True:
        a = random.uniform(0, 2 * math.pi)
        if random.random() < abs(math.sin(2 * a)):
            return a


def place(base: Polygon, existing: List[Polygon], buf: float, attempts: int = 100) -> Tuple[float, float]:
    if not existing:
        return 0.0, 0.0

    idx = STRtree(existing)
    best = (0.0, 0.0)
    min_r = float('inf')

    for _ in range(attempts):
        a = weighted_angle()
        vx, vy = math.cos(a), math.sin(a)

        r = 15.0
        while r >= 0:
            px, py = r * vx, r * vy
            cand = affinity.translate(base, xoff=px, yoff=py)
            if collides_indexed(cand, idx, existing, buf):
                break
            r -= 0.1

        # Back up
        while r < 30:
            r += 0.01
            px, py = r * vx, r * vy
            cand = affinity.translate(base, xoff=px, yoff=py)
            if not collides_indexed(cand, idx, existing, buf):
                break

        if r < min_r:
            min_r = r
            best = (px, py)

    return best


# =============================================================================
# SIMULATED ANNEALING - Streamlined
# =============================================================================

def sa(placements: List[Tuple[float, float, float]], buf: float, iterations: int = 2000) -> List[Tuple[float, float, float]]:
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]

    cur_score = bbox_polys(polys)
    best_score = cur_score
    best = list(current)

    T = 1.5
    shift = 0.05 * cur_score

    for it in range(iterations):
        i = random.randrange(n)
        x, y, d = current[i]

        r = random.random()
        if r < 0.6:
            nx = x + random.gauss(0, shift)
            ny = y + random.gauss(0, shift)
            nd = d
        elif r < 0.8:
            nx, ny = x, y
            nd = (d + random.choice([-5, 5, -10, 10, -30, 30])) % 360
        else:
            # Center-seeking
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 0.001
            step = random.uniform(0.02, 0.08) * cur_score
            nx = x + step * dx / dist
            ny = y + step * dy / dist
            nd = d

        new_poly = make_tree(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if collides_simple(new_poly, others, buf):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_score = bbox_polys(polys)

        delta = new_score - cur_score
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current[i] = (nx, ny, nd)
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best = list(current)
        else:
            polys[i] = old_poly

        T *= 0.998

    return best


# =============================================================================
# LOCAL SEARCH - Light
# =============================================================================

def local(placements: List[Tuple[float, float, float]], buf: float, iterations: int = 100) -> List[Tuple[float, float, float]]:
    n = len(placements)
    if n <= 1:
        return placements

    prec = 0.005
    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]
    cur_score = bbox_polys(polys)

    for _ in range(iterations):
        improved = False
        for i in range(n):
            x, y, d = current[i]
            for dx, dy in [(-prec, 0), (prec, 0), (0, -prec), (0, prec)]:
                nx, ny = x + dx, y + dy
                new_poly = make_tree(nx, ny, d)
                others = polys[:i] + polys[i+1:]

                if not collides_simple(new_poly, others, buf):
                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bbox_polys(polys)

                    if new_score < cur_score - 0.0001:
                        current[i] = (nx, ny, d)
                        cur_score = new_score
                        improved = True
                        break
                    polys[i] = old_poly
        if not improved:
            break

    return current


# =============================================================================
# COMPACT - Light
# =============================================================================

def compact(placements: List[Tuple[float, float, float]], buf: float, passes: int = 5) -> List[Tuple[float, float, float]]:
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)

    for _ in range(passes):
        polys = [make_tree(x, y, d) for x, y, d in current]
        cx = sum(p[0] for p in current) / n
        cy = sum(p[1] for p in current) / n

        for i in range(n):
            x, y, d = current[i]
            dist = math.sqrt((x-cx)**2 + (y-cy)**2)
            if dist < 0.1:
                continue

            dx, dy = cx - x, cy - y
            norm = dist + 0.001

            for mult in [0.1, 0.05, 0.02]:
                move = mult * dist
                nx = x + move * dx / norm
                ny = y + move * dy / norm

                new_poly = make_tree(nx, ny, d)
                others = polys[:i] + polys[i+1:]

                if not collides_simple(new_poly, others, buf):
                    current[i] = (nx, ny, d)
                    polys[i] = new_poly
                    break

    return current


# =============================================================================
# SOLVER
# =============================================================================

class QuickSolver:
    def __init__(self, seed: int = 42, restarts: int = 15, buf: float = 0.015):
        self.seed = seed
        self.restarts = restarts
        self.buf = buf
        random.seed(seed)
        np.random.seed(seed)

        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.scores: Dict[int, float] = {}

    def solve_one(self, n: int, prev: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float, float]], float]:
        if n <= 0:
            return [], 0.0
        if n == 1:
            return [(0.0, 0.0, 0.0)], 1.0

        prev_polys = [make_tree(x, y, d) for x, y, d in prev]
        angles = [i * 5.0 for i in range(72)]

        best_sol = None
        best_score = float('inf')

        for r in range(self.restarts):
            random.seed(self.seed + r * 1000 + n)

            angle = angles[r % 72]
            base = ROTATED_POLYGONS[int(angle) % 360]
            bx, by = place(base, prev_polys, self.buf, attempts=50 + r * 5)

            sol = prev + [(bx, by, angle)]

            # Perturb from best on later restarts
            if r > 0 and best_sol is not None and r % 2 == 0:
                sol = list(best_sol)
                for idx in random.sample(range(n), max(1, n // 10)):
                    x, y, d = sol[idx]
                    sol[idx] = (
                        x + random.gauss(0, 0.015 * best_score),
                        y + random.gauss(0, 0.015 * best_score),
                        (d + random.choice([-5, 5, -10, 10])) % 360
                    )

            if has_overlaps(sol, self.buf):
                continue

            sol = sa(sol, self.buf, iterations=1500 + r * 100)
            sol = local(sol, self.buf, iterations=50)
            sol = compact(sol, self.buf, passes=3)

            score = bbox_side(sol)
            if not has_overlaps(sol, self.buf) and score < best_score:
                best_score = score
                best_sol = sol

        if best_sol is None:
            # Fallback
            angle = random.uniform(0, 360)
            base = affinity.rotate(BASE_POLYGON, angle, origin=(0, 0))
            bx, by = place(base, prev_polys, self.buf, attempts=200)
            best_sol = prev + [(bx, by, angle)]
            best_score = bbox_side(best_sol)

        return center(best_sol), best_score

    def solve_all(self, max_n: int = 200, verbose: bool = True):
        start = time.time()

        if verbose:
            print("=" * 60)
            print("SANTA 2025 - QUICK SOLVER")
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
            polys = [make_tree(x, y, d) for x, y, d in pos]
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="quick_submission.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=15)
    args = parser.parse_args()

    solver = QuickSolver(seed=args.seed, restarts=args.restarts)
    solutions = solver.solve_all(max_n=200, verbose=True)

    print("\nValidating...")
    if validate(solutions):
        print("All valid!")

    print(f"\nSaving: {args.output}")
    save(solutions, args.output)
    print(f"Final: {solver.total_score():.4f}")


if __name__ == "__main__":
    main()
