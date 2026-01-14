#!/usr/bin/env python3
"""
Santa 2025 - ELITE Solver V3 (Target: Score 62)
================================================

Key features:
1. Multi-core parallel processing (uses all CPU cores)
2. Zero collision buffer (touching is OK)
3. Greedy placement minimizing bounding box
4. Multiple angle combinations tested
5. Simulated annealing + local optimization

Tree area = 0.2456
Theoretical minimum score (100% efficiency) = 49
Target score 62 = ~77% packing efficiency

Usage:
    python elite_solver_v3.py [--output submission.csv]
    python elite_solver_v3.py --quick      # Fast test (~1 hour)
    python elite_solver_v3.py --hours 24   # Custom time limit
    python elite_solver_v3.py --workers 8  # Use 8 cores
"""

import os
import sys
import math
import time
import random
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

try:
    from shapely.geometry import Polygon, Point
    from shapely import affinity
    from shapely.ops import unary_union
    from shapely.prepared import prep
    HAS_SHAPELY = True
except ImportError:
    print("ERROR: shapely required. pip install shapely")
    sys.exit(1)

try:
    import pyclipper
    HAS_PYCLIPPER = True
except ImportError:
    HAS_PYCLIPPER = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x


# =============================================================================
# TREE GEOMETRY
# =============================================================================

TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

BASE_POLYGON = Polygon(TREE_COORDS)
TREE_AREA = BASE_POLYGON.area  # 0.2456

# Cache rotated polygons
ANGLE_CACHE = {}
for angle in range(360):
    ANGLE_CACHE[angle] = affinity.rotate(BASE_POLYGON, angle, origin=(0, 0))


def make_tree(x: float, y: float, angle: float) -> Polygon:
    """Create tree polygon at position with rotation."""
    a = int(angle) % 360
    poly = ANGLE_CACHE.get(a, affinity.rotate(BASE_POLYGON, angle, origin=(0, 0)))
    return affinity.translate(poly, xoff=x, yoff=y)


# =============================================================================
# COLLISION DETECTION
# =============================================================================

def check_collision(p1: Polygon, p2: Polygon) -> bool:
    """Check if polygons collide (overlap, not just touch)."""
    return p1.intersects(p2) and not p1.touches(p2)


def has_any_collision(placements: List[Tuple[float, float, float]]) -> bool:
    """Check if any placements collide."""
    polys = [make_tree(x, y, a) for x, y, a in placements]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if check_collision(polys[i], polys[j]):
                return True
    return False


def collides_with_any(new_poly: Polygon, polys: List[Polygon]) -> bool:
    """Check if new polygon collides with existing ones."""
    for p in polys:
        if check_collision(new_poly, p):
            return True
    return False


# =============================================================================
# SCORING
# =============================================================================

def compute_side(placements: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side."""
    if not placements:
        return 0.0
    polys = [make_tree(x, y, a) for x, y, a in placements]
    bounds = unary_union(polys).bounds
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1])


def compute_bounds(placements: List[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    """Get bounding box."""
    if not placements:
        return (0, 0, 0, 0)
    polys = [make_tree(x, y, a) for x, y, a in placements]
    return unary_union(polys).bounds


# =============================================================================
# NFP CALCULATOR
# =============================================================================

class NFPCalculator:
    """No-Fit Polygon calculator for tight placement."""

    def __init__(self):
        self.cache = {}

    def compute_nfp(self, angle1: float, angle2: float) -> Optional[Polygon]:
        """Compute NFP between two tree orientations."""
        if not HAS_PYCLIPPER:
            return None

        a1, a2 = int(angle1) % 360, int(angle2) % 360
        key = (a1, a2)
        if key in self.cache:
            return self.cache[key]

        try:
            fixed = make_tree(0, 0, a1)
            moving = make_tree(0, 0, a2)
            moving_ref = affinity.scale(moving, -1, -1, origin=(0, 0))

            precision = 100000
            fixed_path = [(int(x*precision), int(y*precision))
                         for x, y in list(fixed.exterior.coords)[:-1]]
            moving_path = [(int(x*precision), int(y*precision))
                          for x, y in list(moving_ref.exterior.coords)[:-1]]

            result = pyclipper.MinkowskiSum(fixed_path, moving_path, True)
            if result and len(result[0]) >= 3:
                coords = [(x/precision, y/precision) for x, y in result[0]]
                nfp = Polygon(coords)
                self.cache[key] = nfp
                return nfp
        except:
            pass
        return None


# =============================================================================
# GREEDY PLACER
# =============================================================================

class GreedyPlacer:
    """Greedy placement minimizing bounding box."""

    def __init__(self):
        self.nfp = NFPCalculator()

    def place_greedy(self, n: int, angles: List[float]) -> List[Tuple[float, float, float]]:
        """Place trees using greedy heuristic."""
        placements = []
        polys = []

        for i in range(n):
            angle = angles[i % len(angles)]

            if i == 0:
                placements.append((0.0, 0.0, angle))
                polys.append(make_tree(0, 0, angle))
                continue

            # Find best position (minimize bounding box)
            best_pos = self.find_best_position(placements, polys, angle)
            placements.append((best_pos[0], best_pos[1], angle))
            polys.append(make_tree(best_pos[0], best_pos[1], angle))

        return placements

    def find_best_position(self, placements: List[Tuple[float, float, float]],
                            polys: List[Polygon], angle: float) -> Tuple[float, float]:
        """Find position that minimizes bounding box."""
        candidates = []

        # Current bounds
        if polys:
            current_bounds = unary_union(polys).bounds
            cx = (current_bounds[0] + current_bounds[2]) / 2
            cy = (current_bounds[1] + current_bounds[3]) / 2
        else:
            cx, cy = 0, 0
            current_bounds = (0, 0, 0, 0)

        # NFP-based candidates (tight positions)
        for idx, (fx, fy, fa) in enumerate(placements):
            nfp = self.nfp.compute_nfp(fa, angle)
            if nfp:
                nfp_t = affinity.translate(nfp, fx, fy)
                boundary = nfp_t.exterior
                for t in np.linspace(0, 1, 80, endpoint=False):
                    pt = boundary.interpolate(t, normalized=True)
                    candidates.append((pt.x, pt.y))

        # Grid candidates near existing trees
        if polys:
            step = 0.08
            for x in np.arange(current_bounds[0] - 0.5, current_bounds[2] + 1, step):
                for y in np.arange(current_bounds[1] - 0.5, current_bounds[3] + 1, step):
                    candidates.append((x, y))

        # Evaluate: minimize new bounding box
        best_pos = None
        best_side = float('inf')

        for x, y in candidates:
            new_poly = make_tree(x, y, angle)
            if collides_with_any(new_poly, polys):
                continue

            # Compute new bounding box
            new_bounds = new_poly.bounds
            minx = min(current_bounds[0], new_bounds[0])
            miny = min(current_bounds[1], new_bounds[1])
            maxx = max(current_bounds[2], new_bounds[2])
            maxy = max(current_bounds[3], new_bounds[3])
            new_side = max(maxx - minx, maxy - miny)

            if new_side < best_side:
                best_side = new_side
                best_pos = (x, y)

        if best_pos is None:
            # Fallback: place adjacent to rightmost tree
            if polys:
                best_pos = (current_bounds[2] + 0.5, cy)
            else:
                best_pos = (0.5, 0)

        return best_pos


# =============================================================================
# LOCAL OPTIMIZER
# =============================================================================

class LocalOptimizer:
    """Local search to refine placements."""

    def optimize(self, placements: List[Tuple[float, float, float]],
                time_limit: float) -> List[Tuple[float, float, float]]:
        """Optimize using local moves."""
        start = time.time()
        n = len(placements)
        if n == 0:
            return placements

        current = list(placements)
        best = list(current)
        best_side = compute_side(best)

        step = 0.01
        iteration = 0

        while time.time() - start < time_limit:
            iteration += 1
            improved = False

            # Try moving each tree toward center and in all directions
            bounds = compute_bounds(current)
            cx = (bounds[0] + bounds[2]) / 2
            cy = (bounds[1] + bounds[3]) / 2

            for i in range(n):
                x, y, a = current[i]

                # Try moves toward center
                dx, dy = cx - x, cy - y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0.01:
                    moves = [
                        (x + dx/dist*step, y + dy/dist*step, a),  # toward center
                    ]
                else:
                    moves = []

                # Also try cardinal directions
                moves.extend([
                    (x + step, y, a), (x - step, y, a),
                    (x, y + step, a), (x, y - step, a),
                ])

                for nx, ny, na in moves:
                    test = current[:i] + [(nx, ny, na)] + current[i+1:]
                    if not has_any_collision(test):
                        side = compute_side(test)
                        if side < best_side:
                            best = list(test)
                            best_side = side
                            current = list(test)
                            improved = True
                            break

            if not improved:
                step *= 0.8
                if step < 0.0005:
                    break

        return best


# =============================================================================
# SIMULATED ANNEALING
# =============================================================================

class SimulatedAnnealing:
    """SA for global optimization."""

    def optimize(self, placements: List[Tuple[float, float, float]],
                time_limit: float, temp: float = 0.5, seed: int = 42) -> List[Tuple[float, float, float]]:
        """Run simulated annealing."""
        rng = random.Random(seed)
        start = time.time()
        n = len(placements)
        if n == 0:
            return placements

        current = list(placements)
        current_side = compute_side(current)
        best = list(current)
        best_side = current_side

        cooling = 0.9995

        while time.time() - start < time_limit and temp > 1e-6:
            # Generate neighbor
            idx = rng.randint(0, n - 1)
            x, y, a = current[idx]

            move = rng.choice(['pos', 'angle', 'swap'])
            if move == 'pos':
                nx = x + rng.gauss(0, 0.03)
                ny = y + rng.gauss(0, 0.03)
                neighbor = current[:idx] + [(nx, ny, a)] + current[idx+1:]
            elif move == 'angle':
                na = (a + rng.choice([15, 30, 45, 90, 180])) % 360
                neighbor = current[:idx] + [(x, y, na)] + current[idx+1:]
            else:  # swap
                j = rng.randint(0, n - 1)
                neighbor = list(current)
                neighbor[idx], neighbor[j] = neighbor[j], neighbor[idx]

            if has_any_collision(neighbor):
                temp *= cooling
                continue

            neighbor_side = compute_side(neighbor)
            delta = neighbor_side - current_side

            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current = neighbor
                current_side = neighbor_side
                if current_side < best_side:
                    best = list(current)
                    best_side = current_side

            temp *= cooling

        return best


# =============================================================================
# WORKER FUNCTION FOR PARALLEL PROCESSING
# =============================================================================

def solve_puzzle_worker(args: Tuple[int, float, int]) -> Tuple[int, List[Tuple[float, float, float]], float]:
    """Worker function to solve a single puzzle (for parallel execution)."""
    n, time_budget, seed = args

    if n == 0:
        return (n, [], 0.0)

    rng = random.Random(seed)
    np.random.seed(seed)

    placer = GreedyPlacer()
    local = LocalOptimizer()
    sa = SimulatedAnnealing()

    start = time.time()
    best = None
    best_side = float('inf')

    # Angle configurations to try
    angle_configs = [
        [45],  # All 45°
        [0, 180],  # Alternating up/down
        [45, 225],  # Alternating diagonal
        [0, 90, 180, 270],  # All cardinal
        [30, 210],
        [60, 240],
    ]

    # 1. Try greedy placement with different angles
    for config in angle_configs:
        if time.time() - start > time_budget * 0.3:
            break
        try:
            placements = placer.place_greedy(n, config)
            if not has_any_collision(placements):
                side = compute_side(placements)
                if side < best_side:
                    best = placements
                    best_side = side
        except:
            continue

    if best is None:
        best = placer.place_greedy(n, [45])

    # 2. Simulated annealing
    remaining = time_budget - (time.time() - start)
    if remaining > 0.5:
        sa_result = sa.optimize(best, remaining * 0.5, seed=seed)
        if not has_any_collision(sa_result):
            side = compute_side(sa_result)
            if side < best_side:
                best = sa_result
                best_side = side

    # 3. Local optimization
    remaining = time_budget - (time.time() - start)
    if remaining > 0.1:
        local_result = local.optimize(best, remaining * 0.9)
        if not has_any_collision(local_result):
            side = compute_side(local_result)
            if side < best_side:
                best = local_result
                best_side = side

    contrib = (best_side ** 2) / n
    return (n, best, contrib)


# =============================================================================
# MAIN SOLVER
# =============================================================================

@dataclass
class Config:
    max_hours: float = 24.0
    workers: int = 0  # 0 = auto-detect
    seed: int = 42


class EliteSolverV3:
    """Main solver with parallel processing."""

    def __init__(self, config: Config):
        self.config = config

        # Auto-detect workers
        if config.workers <= 0:
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = config.workers

        random.seed(config.seed)
        np.random.seed(config.seed)

    def solve_all(self, output: str = "submission_v3.csv") -> Dict[int, List[Tuple[float, float, float]]]:
        """Solve all puzzles using parallel processing."""
        solutions = {}
        total_score = 0.0

        start = time.time()
        max_time = self.config.max_hours * 3600

        # Priority: small n gets more time
        priorities = {n: 1.0 + 14.0 * (200 - n) / 199 for n in range(1, 201)}
        total_priority = sum(priorities.values())

        # Calculate time budgets for each puzzle
        time_budgets = {}
        remaining_time = max_time
        for n in range(1, 201):
            rem_priority = sum(priorities[k] for k in range(n, 201))
            budget = min(remaining_time * priorities[n] / rem_priority, 600)
            time_budgets[n] = budget
            remaining_time -= budget

        print(f"Using {self.num_workers} workers")
        print(f"Total time budget: {max_time/3600:.1f} hours")
        print()

        # Create tasks: (n, time_budget, seed)
        tasks = [(n, time_budgets[n], self.config.seed + n) for n in range(1, 201)]

        # Process in parallel with progress tracking
        if self.num_workers > 1:
            # Parallel mode
            completed = 0
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(solve_puzzle_worker, task): task[0] for task in tasks}

                pbar = tqdm(total=200, desc="Solving") if HAS_TQDM else None

                for future in as_completed(futures):
                    n, placements, contrib = future.result()
                    solutions[n] = placements
                    total_score += contrib
                    completed += 1

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({'score': f'{total_score:.2f}', 'done': completed})

                if pbar:
                    pbar.close()
        else:
            # Sequential mode (single worker)
            pbar = tqdm(tasks, desc="Solving") if HAS_TQDM else tasks
            for task in pbar:
                n, placements, contrib = solve_puzzle_worker(task)
                solutions[n] = placements
                total_score += contrib

                if HAS_TQDM:
                    pbar.set_postfix({'score': f'{total_score:.2f}', 'n': n})

        # Save results
        self.save(solutions, output)
        return solutions

    def save(self, solutions: Dict, path: str):
        """Save submission."""
        with open(path, 'w') as f:
            f.write("id,x,y,deg\n")
            for n in sorted(solutions.keys()):
                placements = solutions[n]
                if placements:
                    polys = [make_tree(x, y, a) for x, y, a in placements]
                    bounds = unary_union(polys).bounds
                    ox, oy = bounds[0], bounds[1]
                else:
                    ox, oy = 0, 0
                for idx, (x, y, a) in enumerate(placements):
                    f.write(f"{n:03d}_{idx},{x-ox:.6f},{y-oy:.6f},{a:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Santa 2025 Elite Solver V3 (Multi-core)')
    parser.add_argument('--output', '-o', default='submission_v3.csv', help='Output file')
    parser.add_argument('--quick', action='store_true', help='Quick mode (~1 hour)')
    parser.add_argument('--hours', type=float, default=24.0, help='Time limit in hours')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers (0=auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    config = Config(
        max_hours=1.0 if args.quick else args.hours,
        workers=args.workers,
        seed=args.seed,
    )

    print("=" * 50)
    print("ELITE SOLVER V3 (Multi-core)")
    print("=" * 50)
    print(f"Hours: {config.max_hours}")
    print(f"Workers: {config.workers if config.workers > 0 else 'auto'}")
    print(f"NFP: {'YES' if HAS_PYCLIPPER else 'NO'}")
    print(f"Seed: {config.seed}")
    print("=" * 50)

    solver = EliteSolverV3(config)
    solutions = solver.solve_all(args.output)

    score = sum((compute_side(solutions[n])**2)/n for n in solutions)
    print()
    print("=" * 50)
    print(f"FINAL SCORE: {score:.4f}")
    print(f"Saved: {args.output}")
    print("=" * 50)


if __name__ == '__main__':
    main()
