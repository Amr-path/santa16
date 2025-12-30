#!/usr/bin/env python3
"""
Santa 2025 - FAST Optimization Solver
======================================

Balanced speed/quality optimization targeting score < 60:
- 50 restarts per puzzle
- Fast simulated annealing with adaptive cooling
- Efficient compacting
- ~2-3 seconds per puzzle (~10 min total)

Usage:
    python fast_solver.py [--output submission.csv] [--seed 42]
"""

import math
import time
import random
import argparse
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FastConfig:
    """Balanced speed/quality configuration."""
    num_restarts: int = 50
    num_placement_attempts: int = 200
    start_radius: float = 15.0
    step_in: float = 0.05
    step_out: float = 0.005
    collision_buffer: float = 0.012

    sa_iterations: int = 10000
    sa_temp_initial: float = 2.0
    sa_temp_final: float = 0.0001

    local_precision: float = 0.003
    local_iterations: int = 300

    compact_passes: int = 10
    compact_step: float = 0.002

    seed: int = 42


# =============================================================================
# GEOMETRY
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


def collides(poly: Polygon, others: List[Polygon], buf: float = 0.0) -> bool:
    for o in others:
        if buf > 0:
            if poly.distance(o) < buf:
                return True
        elif poly.intersects(o) and not poly.touches(o):
            return True
    return False


def collides_fast(poly: Polygon, tree_idx: STRtree, all_polys: List[Polygon], buf: float = 0.0) -> bool:
    query = poly.buffer(buf) if buf > 0 else poly
    for idx in tree_idx.query(query):
        if buf > 0:
            if poly.distance(all_polys[idx]) < buf:
                return True
        elif poly.intersects(all_polys[idx]) and not poly.touches(all_polys[idx]):
            return True
    return False


def bbox_side(placements: List[Tuple[float, float, float]]) -> float:
    if not placements:
        return 0.0
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def bbox_side_polys(polys: List[Polygon]) -> float:
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


def check_overlaps(placements: List[Tuple[float, float, float]], buf: float = 0.0) -> bool:
    polys = [make_tree(x, y, d) for x, y, d in placements]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if buf > 0:
                if polys[i].distance(polys[j]) < buf:
                    return True
            elif polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
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


def place_radial(base: Polygon, existing: List[Polygon], cfg: FastConfig) -> Tuple[float, float]:
    if not existing:
        return 0.0, 0.0

    idx = STRtree(existing)
    best = (None, None)
    min_r = float('inf')

    for _ in range(cfg.num_placement_attempts):
        a = weighted_angle()
        vx, vy = math.cos(a), math.sin(a)

        r = cfg.start_radius
        found = False

        while r >= 0:
            px, py = r * vx, r * vy
            cand = affinity.translate(base, xoff=px, yoff=py)
            if collides_fast(cand, idx, existing, cfg.collision_buffer):
                found = True
                break
            r -= cfg.step_in

        if found:
            while r < cfg.start_radius * 2:
                r += cfg.step_out
                px, py = r * vx, r * vy
                cand = affinity.translate(base, xoff=px, yoff=py)
                if not collides_fast(cand, idx, existing, cfg.collision_buffer):
                    break
        else:
            r, px, py = 0, 0.0, 0.0

        if r < min_r:
            min_r = r
            best = (px, py)

    return best[0] if best[0] is not None else 0.0, best[1] if best[1] is not None else 0.0


# =============================================================================
# SIMULATED ANNEALING
# =============================================================================

def sa_fast(placements: List[Tuple[float, float, float]], cfg: FastConfig) -> List[Tuple[float, float, float]]:
    n = len(placements)
    if n <= 1:
        return placements

    buf = cfg.collision_buffer

    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]

    cur_score = bbox_side_polys(polys)
    best_score = cur_score
    best = list(current)

    T = cfg.sa_temp_initial
    shift = 0.08 * cur_score

    for it in range(cfg.sa_iterations):
        i = random.randrange(n)
        x, y, d = current[i]

        mt = random.random()
        if mt < 0.5:
            # Position
            nx = x + random.gauss(0, shift)
            ny = y + random.gauss(0, shift)
            nd = d
        elif mt < 0.7:
            # Large position
            nx = x + random.uniform(-shift * 3, shift * 3)
            ny = y + random.uniform(-shift * 3, shift * 3)
            nd = d
        elif mt < 0.85:
            # Rotation
            nx, ny = x, y
            nd = (d + random.choice([-5, 5, -10, 10, -15, 15, -30, 30])) % 360
        else:
            # Center
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 0.001
            step = random.uniform(0.02, 0.1) * cur_score
            nx = x + step * dx / dist
            ny = y + step * dy / dist
            nd = d

        new_poly = make_tree(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if collides(new_poly, others, buf):
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_score = bbox_side_polys(polys)

        delta = new_score - cur_score
        if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current[i] = (nx, ny, nd)
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best = list(current)
        else:
            polys[i] = old_poly

        # Adaptive temperature
        progress = it / cfg.sa_iterations
        T = cfg.sa_temp_initial * ((cfg.sa_temp_final / cfg.sa_temp_initial) ** progress)

        if it % 500 == 0:
            shift = 0.08 * cur_score * (1 - 0.5 * progress)

    return best


# =============================================================================
# LOCAL SEARCH
# =============================================================================

def local_search(placements: List[Tuple[float, float, float]], cfg: FastConfig) -> List[Tuple[float, float, float]]:
    n = len(placements)
    if n <= 1:
        return placements

    buf = cfg.collision_buffer
    prec = cfg.local_precision

    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]
    cur_score = bbox_side_polys(polys)

    for _ in range(cfg.local_iterations):
        improved = False

        for i in random.sample(range(n), n):
            x, y, d = current[i]

            for dx, dy in [(-prec, 0), (prec, 0), (0, -prec), (0, prec),
                           (-prec, -prec), (-prec, prec), (prec, -prec), (prec, prec)]:
                nx, ny = x + dx, y + dy
                new_poly = make_tree(nx, ny, d)
                others = polys[:i] + polys[i+1:]

                if not collides(new_poly, others, buf):
                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bbox_side_polys(polys)

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
# COMPACTING
# =============================================================================

def compact(placements: List[Tuple[float, float, float]], cfg: FastConfig) -> List[Tuple[float, float, float]]:
    n = len(placements)
    if n <= 1:
        return placements

    buf = cfg.collision_buffer
    step = cfg.compact_step

    current = list(placements)

    for _ in range(cfg.compact_passes):
        polys = [make_tree(x, y, d) for x, y, d in current]

        cx = sum(p[0] for p in current) / n
        cy = sum(p[1] for p in current) / n

        dists = [(i, math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)) for i, p in enumerate(current)]
        dists.sort(key=lambda x: -x[1])

        for idx, dist in dists:
            if dist < 0.05:
                continue

            x, y, d = current[idx]
            dx, dy = cx - x, cy - y
            norm = math.sqrt(dx*dx + dy*dy) + 0.001

            for mult in [1.0, 0.5, 0.2]:
                move = step * mult * dist * 3
                nx = x + move * dx / norm
                ny = y + move * dy / norm

                new_poly = make_tree(nx, ny, d)
                others = polys[:idx] + polys[idx+1:]

                if not collides(new_poly, others, buf):
                    current[idx] = (nx, ny, d)
                    polys[idx] = new_poly
                    break

    return current


# =============================================================================
# SOLVER
# =============================================================================

class FastSolver:
    def __init__(self, config: Optional[FastConfig] = None, seed: int = 42):
        self.cfg = config or FastConfig()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.scores: Dict[int, float] = {}

    def solve_single(self, n: int, prev: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float, float]], float]:
        if n <= 0:
            return [], 0.0

        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            return sol, bbox_side(sol)

        prev_polys = [make_tree(x, y, d) for x, y, d in prev]

        best_sol = None
        best_score = float('inf')

        angles = [i * 5.0 for i in range(72)]

        for restart in range(self.cfg.num_restarts):
            random.seed(self.seed + restart * 1000 + n)

            angle = angles[restart % 72]
            base = ROTATED_POLYGONS[int(angle) % 360]

            bx, by = place_radial(base, prev_polys, self.cfg)
            sol = prev + [(bx, by, angle)]

            # Perturb from best
            if restart > 0 and best_sol is not None and restart % 3 != 0:
                sol = list(best_sol)
                for idx in random.sample(range(n), max(1, n // 10)):
                    x, y, d = sol[idx]
                    sol[idx] = (
                        x + random.gauss(0, 0.02 * best_score),
                        y + random.gauss(0, 0.02 * best_score),
                        (d + random.uniform(-10, 10)) % 360
                    )

            if check_overlaps(sol, self.cfg.collision_buffer):
                continue

            # Optimize
            sol = sa_fast(sol, self.cfg)
            sol = local_search(sol, self.cfg)
            sol = compact(sol, self.cfg)

            score = bbox_side(sol)
            if not check_overlaps(sol, self.cfg.collision_buffer) and score < best_score:
                best_score = score
                best_sol = sol

        if best_sol is None:
            angle = random.uniform(0, 360)
            base = affinity.rotate(BASE_POLYGON, angle, origin=(0, 0))
            bx, by = place_radial(base, prev_polys, self.cfg)
            best_sol = prev + [(bx, by, angle)]
            best_score = bbox_side(best_sol)

        return center(best_sol), best_score

    def solve_all(self, max_n: int = 200, verbose: bool = True) -> Dict[int, List[Tuple[float, float, float]]]:
        total_start = time.time()

        if verbose:
            print("=" * 70)
            print("SANTA 2025 - FAST SOLVER")
            print("=" * 70)
            print(f"Restarts: {self.cfg.num_restarts}")
            print(f"SA iterations: {self.cfg.sa_iterations}")
            print(f"Collision buffer: {self.cfg.collision_buffer}")
            print()

        for n in range(1, max_n + 1):
            n_start = time.time()

            prev = self.solutions[n - 1] if n > 1 else []
            sol, score = self.solve_single(n, prev)

            self.solutions[n] = sol
            self.scores[n] = score

            elapsed = time.time() - n_start

            if verbose and (n <= 10 or n % 10 == 0 or n == max_n):
                total = self.total_score()
                print(f"n={n:3d}: side={score:.4f}, time={elapsed:.1f}s, total={total:.2f}")

        if verbose:
            total_time = time.time() - total_start
            total = self.total_score()
            print()
            print("=" * 70)
            print(f"COMPLETE - Time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"Final Score: {total:.4f}")
            print("=" * 70)

        return self.solutions

    def total_score(self) -> float:
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, bbox_side(sol))
            total += (side ** 2) / n
        return total


# =============================================================================
# I/O
# =============================================================================

def create_submission(solutions: Dict[int, List[Tuple[float, float, float]]],
                      output: str = "submission.csv") -> str:
    with open(output, "w") as f:
        f.write("id,x,y,deg\n")
        for n in range(1, 201):
            if n not in solutions:
                raise ValueError(f"Missing n={n}")
            pos = solutions[n]
            polys = [make_tree(x, y, d) for x, y, d in pos]
            b = unary_union(polys).bounds
            for idx, (x, y, d) in enumerate(pos):
                f.write(f"{n:03d}_{idx},s{x - b[0]:.6f},s{y - b[1]:.6f},s{d:.6f}\n")
    return output


def validate_all(solutions: Dict[int, List[Tuple[float, float, float]]]) -> bool:
    valid = True
    for n in range(1, 201):
        if n not in solutions or len(solutions[n]) != n:
            print(f"  n={n}: Invalid count")
            valid = False
            continue
        if check_overlaps(solutions[n]):
            print(f"  n={n}: Has overlaps")
            valid = False
    return valid


def print_summary(solutions: Dict[int, List[Tuple[float, float, float]]]):
    print("=" * 60)
    sides = {n: bbox_side(sol) for n, sol in solutions.items()}
    contribs = {n: (s ** 2) / n for n, s in sides.items()}
    total = sum(contribs.values())

    print(f"Total score: {total:.4f}")
    print(f"Baseline: 157.08")
    print(f"Improvement: {(157.08 - total) / 157.08 * 100:.1f}%")

    worst = sorted(contribs.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 worst contributions:")
    for n, c in worst:
        print(f"  n={n:3d}: side={sides[n]:.4f}, contrib={c:.4f}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Santa 2025 Fast Solver")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-n", type=int, default=200)
    parser.add_argument("--restarts", type=int, default=50)
    args = parser.parse_args()

    cfg = FastConfig()
    cfg.seed = args.seed
    cfg.num_restarts = args.restarts

    solver = FastSolver(config=cfg, seed=args.seed)
    solutions = solver.solve_all(max_n=args.max_n, verbose=True)

    print("\nValidating...")
    if validate_all(solutions):
        print("All solutions valid!")

    print()
    print_summary(solutions)

    print(f"\nSaving: {args.output}")
    create_submission(solutions, args.output)
    print(f"Final Score: {solver.total_score():.4f}")


if __name__ == "__main__":
    main()
