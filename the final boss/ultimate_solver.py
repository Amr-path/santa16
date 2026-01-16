#!/usr/bin/env python3
"""
Santa 2025 - ULTIMATE Solver (Fixed - Independent & Parallel)
==============================================================

FIXED VERSION: Each puzzle solved independently with multi-core parallelism.

Key improvements over previous version:
- Independent puzzle solving (NOT incremental)
- 15-core parallel processing by default
- Better initial placement strategies
- More aggressive optimization for small n

Usage:
    python ultimate_solver.py --output submission.csv --cores 15
    python ultimate_solver.py --quick --cores 15     # Fast test (~1-2 hours)
    python ultimate_solver.py --standard --cores 15  # Standard (~6-12 hours)
    python ultimate_solver.py --ultra --cores 15     # Maximum (~24-48 hours)

Requirements:
    pip install numpy shapely tqdm
"""

import os
import sys
import math
import time
import random
import argparse
import signal
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from shapely.ops import unary_union

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Solver configuration."""
    # Parallelism
    num_cores: int = 15

    # Time budgets (will be distributed based on priority)
    total_time_hours: float = 12.0

    # Multi-restart per puzzle
    restarts_small: int = 200    # n=1-10
    restarts_medium: int = 100   # n=11-50
    restarts_large: int = 50     # n=51-200

    # Placement
    placement_attempts: int = 200

    # Collision
    collision_buffer: float = 0.01

    # Simulated Annealing
    sa_temp_initial: float = 2.5
    sa_temp_final: float = 1e-9

    # Local search
    local_precision: float = 0.0003
    local_iterations: int = 2000

    # Compacting
    compact_passes: int = 40
    compact_step: float = 0.0002

    seed: int = 42


def quick_config() -> Config:
    """Quick test mode (~1-2 hours)."""
    return Config(
        total_time_hours=2.0,
        restarts_small=50,
        restarts_medium=25,
        restarts_large=15,
        placement_attempts=80,
        local_iterations=800,
        compact_passes=20,
    )


def standard_config() -> Config:
    """Standard mode (~6-12 hours)."""
    return Config(
        total_time_hours=12.0,
        restarts_small=200,
        restarts_medium=100,
        restarts_large=50,
    )


def ultra_config() -> Config:
    """Ultra quality mode (~24-48 hours)."""
    return Config(
        total_time_hours=48.0,
        restarts_small=500,
        restarts_medium=250,
        restarts_large=100,
        placement_attempts=400,
        collision_buffer=0.008,
        local_precision=0.0002,
        local_iterations=4000,
        compact_passes=80,
        compact_step=0.0001,
    )


# =============================================================================
# GEOMETRY
# =============================================================================

TREE_COORDS = [
    (0.0, 0.8),
    (0.125, 0.5),
    (0.0625, 0.5),
    (0.2, 0.25),
    (0.1, 0.25),
    (0.35, 0.0),
    (0.075, 0.0),
    (0.075, -0.2),
    (-0.075, -0.2),
    (-0.075, 0.0),
    (-0.35, 0.0),
    (-0.1, 0.25),
    (-0.2, 0.25),
    (-0.0625, 0.5),
    (-0.125, 0.5),
]

BASE_POLYGON = Polygon(TREE_COORDS)
TREE_AREA = BASE_POLYGON.area

# Pre-compute rotated polygons
ROTATED_POLYGONS = {}
for deg in range(360):
    ROTATED_POLYGONS[deg] = affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))


def make_tree(x: float, y: float, deg: float) -> Polygon:
    """Create tree polygon at position with rotation."""
    snapped = int(round(deg)) % 360
    poly = ROTATED_POLYGONS[snapped]
    if x != 0 or y != 0:
        return affinity.translate(poly, xoff=x, yoff=y)
    return poly


def collides(poly: Polygon, others: List[Polygon], buf: float = 0.0) -> bool:
    """Check collision with buffer."""
    for o in others:
        if buf > 0:
            if poly.distance(o) < buf:
                return True
        elif poly.intersects(o) and not poly.touches(o):
            return True
    return False


def bbox_side(placements: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side."""
    if not placements:
        return 0.0
    polys = [make_tree(x, y, d) for x, y, d in placements]
    return bbox_side_polys(polys)


def bbox_side_polys(polys: List[Polygon]) -> float:
    """Compute bounding square side from polygons."""
    if not polys:
        return 0.0
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def center_placements(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Center placements around origin."""
    if not placements:
        return placements
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]


def check_overlaps(placements: List[Tuple[float, float, float]], buf: float = 0.0) -> bool:
    """Check if any trees overlap."""
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
# INITIAL PLACEMENT STRATEGIES
# =============================================================================

def greedy_radial_placement(n: int, cfg: Config, seed: int) -> List[Tuple[float, float, float]]:
    """Greedy placement using radial approach."""
    random.seed(seed)

    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, random.randint(0, 359))]

    buf = cfg.collision_buffer
    placements = []
    polys = []

    # First tree at origin
    deg = random.randint(0, 359)
    placements.append((0.0, 0.0, deg))
    polys.append(make_tree(0.0, 0.0, deg))

    for i in range(1, n):
        best_pos = None
        best_dist = float('inf')

        for _ in range(cfg.placement_attempts):
            angle = random.uniform(0, 2 * math.pi)
            deg = random.randint(0, 359)

            # Binary search for closest valid position
            r_min, r_max = 0.0, 10.0

            for _ in range(20):
                r = (r_min + r_max) / 2
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                poly = make_tree(x, y, deg)

                if collides(poly, polys, buf):
                    r_min = r
                else:
                    r_max = r

            # Fine-tune outward
            r = r_max
            for _ in range(10):
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                poly = make_tree(x, y, deg)
                if not collides(poly, polys, buf):
                    break
                r += 0.01

            if not collides(poly, polys, buf):
                test_placements = placements + [(x, y, deg)]
                test_polys = polys + [poly]
                score = bbox_side_polys(test_polys)

                if score < best_dist:
                    best_dist = score
                    best_pos = (x, y, deg, poly)

        if best_pos:
            placements.append((best_pos[0], best_pos[1], best_pos[2]))
            polys.append(best_pos[3])
        else:
            # Fallback
            angle = random.uniform(0, 2 * math.pi)
            r = 0.5 + i * 0.3
            x, y = r * math.cos(angle), r * math.sin(angle)
            deg = random.randint(0, 359)
            placements.append((x, y, deg))
            polys.append(make_tree(x, y, deg))

    return placements


def hexagonal_placement(n: int, cfg: Config, seed: int) -> List[Tuple[float, float, float]]:
    """Place trees in hexagonal grid pattern."""
    random.seed(seed)

    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, random.randint(0, 359))]

    buf = cfg.collision_buffer
    placements = []
    polys = []

    dx = 0.72
    dy = dx * math.sqrt(3) / 2

    # Generate grid positions sorted by distance from center
    positions = []
    for ring in range(20):
        for i in range(-ring, ring + 1):
            for j in range(-ring, ring + 1):
                x = i * dx + (j % 2) * dx / 2
                y = j * dy
                dist = x*x + y*y
                positions.append((dist, x, y))

    positions.sort()

    for dist, x, y in positions:
        if len(placements) >= n:
            break

        deg = random.randint(0, 359)
        poly = make_tree(x, y, deg)

        if not collides(poly, polys, buf):
            placements.append((x, y, deg))
            polys.append(poly)

    # Fill remaining with radial if needed
    while len(placements) < n:
        angle = random.uniform(0, 2 * math.pi)
        r = 1.0 + len(placements) * 0.2
        x, y = r * math.cos(angle), r * math.sin(angle)
        deg = random.randint(0, 359)
        poly = make_tree(x, y, deg)

        if not collides(poly, polys, buf):
            placements.append((x, y, deg))
            polys.append(poly)

    return placements[:n]


def spiral_placement(n: int, cfg: Config, seed: int) -> List[Tuple[float, float, float]]:
    """Place trees in spiral pattern."""
    random.seed(seed)

    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, random.randint(0, 359))]

    buf = cfg.collision_buffer
    placements = []
    polys = []

    angle = 0
    radius = 0.0
    golden_angle = math.pi * (3 - math.sqrt(5))  # ~137.5 degrees

    for i in range(n * 3):  # Try more positions than needed
        if len(placements) >= n:
            break

        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        deg = random.randint(0, 359)
        poly = make_tree(x, y, deg)

        if not collides(poly, polys, buf):
            placements.append((x, y, deg))
            polys.append(poly)

        angle += golden_angle
        radius += 0.08

    return placements[:n]


# =============================================================================
# SIMULATED ANNEALING
# =============================================================================

def simulated_annealing(placements: List[Tuple[float, float, float]],
                        cfg: Config, time_limit: float,
                        seed: int) -> List[Tuple[float, float, float]]:
    """Simulated annealing optimization."""
    n = len(placements)
    if n <= 1:
        return placements

    random.seed(seed)
    buf = cfg.collision_buffer
    start = time.time()

    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]

    cur_score = bbox_side_polys(polys)
    best_score = cur_score
    best = list(current)

    T = cfg.sa_temp_initial
    shift = 0.1 * cur_score
    iters = 0
    accepted = 0

    while time.time() - start < time_limit:
        iters += 1

        i = random.randrange(n)
        x, y, d = current[i]

        move = random.random()

        if move < 0.12:
            # Ultra-fine
            nx = x + random.gauss(0, shift * 0.03)
            ny = y + random.gauss(0, shift * 0.03)
            nd = d
        elif move < 0.28:
            # Fine
            nx = x + random.gauss(0, shift * 0.1)
            ny = y + random.gauss(0, shift * 0.1)
            nd = d
        elif move < 0.45:
            # Medium
            nx = x + random.gauss(0, shift * 0.3)
            ny = y + random.gauss(0, shift * 0.3)
            nd = d
        elif move < 0.55:
            # Large
            nx = x + random.uniform(-shift * 1.5, shift * 1.5)
            ny = y + random.uniform(-shift * 1.5, shift * 1.5)
            nd = d
        elif move < 0.65:
            # Fine rotation
            nx, ny = x, y
            nd = (d + random.choice([-1, 1, -2, 2, -3, 3])) % 360
        elif move < 0.75:
            # Large rotation
            nx, ny = x, y
            nd = (d + random.choice([-10, 10, -15, 15, -30, 30, -45, 45])) % 360
        elif move < 0.88:
            # Center seeking
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6
            step = random.uniform(0.005, 0.05) * cur_score
            nx = x + step * dx / dist
            ny = y + step * dy / dist
            nd = d
        else:
            # Swap
            if n >= 2:
                j = random.randrange(n)
                while j == i:
                    j = random.randrange(n)
                ox, oy, od = current[j]

                new_poly_i = make_tree(ox, oy, d)
                new_poly_j = make_tree(x, y, od)
                others = [p for k, p in enumerate(polys) if k != i and k != j]

                if not collides(new_poly_i, others + [new_poly_j], buf) and \
                   not collides(new_poly_j, others + [new_poly_i], buf):
                    old_polys = list(polys)
                    polys[i] = new_poly_i
                    polys[j] = new_poly_j
                    new_score = bbox_side_polys(polys)

                    delta = new_score - cur_score
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                        current[i] = (ox, oy, d)
                        current[j] = (x, y, od)
                        cur_score = new_score
                        accepted += 1
                        if cur_score < best_score:
                            best_score = cur_score
                            best = list(current)
                    else:
                        polys = old_polys

                progress = (time.time() - start) / time_limit
                T = cfg.sa_temp_initial * ((cfg.sa_temp_final / cfg.sa_temp_initial) ** progress)
                continue

            nx, ny, nd = x, y, d

        new_poly = make_tree(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if collides(new_poly, others, buf):
            progress = (time.time() - start) / time_limit
            T = cfg.sa_temp_initial * ((cfg.sa_temp_final / cfg.sa_temp_initial) ** progress)
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_score = bbox_side_polys(polys)

        delta = new_score - cur_score
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current[i] = (nx, ny, nd)
            cur_score = new_score
            accepted += 1
            if cur_score < best_score:
                best_score = cur_score
                best = list(current)
        else:
            polys[i] = old_poly

        progress = (time.time() - start) / time_limit
        T = cfg.sa_temp_initial * ((cfg.sa_temp_final / cfg.sa_temp_initial) ** progress)

        if iters % 2000 == 0:
            rate = accepted / iters
            if rate > 0.35:
                shift = min(cur_score * 0.25, shift * 1.08)
            elif rate < 0.1:
                shift = max(0.0003, shift * 0.92)

    return best


# =============================================================================
# LOCAL SEARCH
# =============================================================================

def local_search(placements: List[Tuple[float, float, float]],
                 cfg: Config) -> List[Tuple[float, float, float]]:
    """Fine-grained local search."""
    n = len(placements)
    if n <= 1:
        return placements

    buf = cfg.collision_buffer
    prec = cfg.local_precision

    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]
    cur_score = bbox_side_polys(polys)

    improved = True
    iters = 0

    while improved and iters < cfg.local_iterations:
        improved = False
        iters += 1

        for i in random.sample(range(n), n):
            x, y, d = current[i]
            best_move = None
            best_improvement = 0

            # Multi-scale moves
            moves = []
            for scale in [0.25, 0.5, 1.0, 1.5, 2.5]:
                p = prec * scale
                moves.extend([
                    (-p, 0), (p, 0), (0, -p), (0, p),
                    (-p, -p), (-p, p), (p, -p), (p, p),
                ])

            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                new_poly = make_tree(nx, ny, d)
                others = polys[:i] + polys[i+1:]

                if not collides(new_poly, others, buf):
                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bbox_side_polys(polys)
                    improvement = cur_score - new_score

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (nx, ny, d, new_poly)

                    polys[i] = old_poly

            # Rotations
            for dd in [-1, 1, -2, 2, -3, 3, -5, 5]:
                nd = (d + dd) % 360
                new_poly = make_tree(x, y, nd)
                others = polys[:i] + polys[i+1:]

                if not collides(new_poly, others, buf):
                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bbox_side_polys(polys)
                    improvement = cur_score - new_score

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (x, y, nd, new_poly)

                    polys[i] = old_poly

            if best_move and best_improvement > 1e-9:
                current[i] = (best_move[0], best_move[1], best_move[2])
                polys[i] = best_move[3]
                cur_score -= best_improvement
                improved = True

    return current


# =============================================================================
# COMPACTING
# =============================================================================

def compact_toward_center(placements: List[Tuple[float, float, float]],
                          cfg: Config) -> List[Tuple[float, float, float]]:
    """Compact trees toward center."""
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

        # Sort by distance from center (furthest first)
        dists = [(i, math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)) for i, p in enumerate(current)]
        dists.sort(key=lambda x: -x[1])

        for idx, dist in dists:
            if dist < 0.01:
                continue

            x, y, d = current[idx]
            dx, dy = cx - x, cy - y
            norm = math.sqrt(dx*dx + dy*dy) + 1e-6

            for mult in [1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015]:
                move = step * mult * dist * 15
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
# SOLVE SINGLE PUZZLE (Independent)
# =============================================================================

def solve_puzzle(n: int, cfg: Config, time_limit: float,
                 seed: int) -> Tuple[int, List[Tuple[float, float, float]], float]:
    """
    Solve a single puzzle INDEPENDENTLY (not incrementally).
    This is the key fix - each puzzle is solved from scratch.
    """
    if n == 0:
        return n, [], 0.0
    if n == 1:
        return n, [(0.0, 0.0, 0.0)], bbox_side([(0.0, 0.0, 0.0)])

    random.seed(seed)
    np.random.seed(seed)

    # Determine restarts based on n
    if n <= 10:
        num_restarts = cfg.restarts_small
    elif n <= 50:
        num_restarts = cfg.restarts_medium
    else:
        num_restarts = cfg.restarts_large

    best_placements = None
    best_score = float('inf')

    start_time = time.time()
    time_per_restart = time_limit / max(num_restarts, 1)

    for restart in range(num_restarts):
        if time.time() - start_time >= time_limit * 0.95:
            break

        restart_seed = seed + restart * 1000

        # Choose placement strategy
        strategy = restart % 3
        if strategy == 0:
            placements = greedy_radial_placement(n, cfg, restart_seed)
        elif strategy == 1:
            placements = hexagonal_placement(n, cfg, restart_seed)
        else:
            placements = spiral_placement(n, cfg, restart_seed)

        # Validate initial placement
        if len(placements) != n or check_overlaps(placements, cfg.collision_buffer):
            continue

        # Optimization pipeline
        remaining = min(time_per_restart * 0.8, time_limit - (time.time() - start_time))

        if remaining > 0.3:
            # SA optimization
            placements = simulated_annealing(placements, cfg, remaining * 0.5, restart_seed)

            # Local search
            placements = local_search(placements, cfg)

            # Compact
            placements = compact_toward_center(placements, cfg)

            # Final local search
            placements = local_search(placements, cfg)

        # Validate and score
        if not check_overlaps(placements, cfg.collision_buffer):
            score = bbox_side(placements)
            if score < best_score:
                best_score = score
                best_placements = center_placements(placements)

    if best_placements is None:
        # Fallback
        best_placements = greedy_radial_placement(n, cfg, seed)
        best_placements = center_placements(best_placements)
        best_score = bbox_side(best_placements)

    return n, best_placements, best_score


# =============================================================================
# MAIN SOLVER
# =============================================================================

class UltimateSolver:
    """Ultimate solver with parallel independent puzzle solving."""

    def __init__(self, config: Optional[Config] = None, verbose: bool = True):
        self.cfg = config or Config()
        self.verbose = verbose
        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.running = True

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n\nReceived stop signal. Saving progress...")
        self.running = False

    def compute_time_allocation(self, total_time: float) -> Dict[int, float]:
        """Allocate more time to small n (higher score contribution)."""
        allocations = {}
        total_weight = 0

        for n in range(1, 201):
            # Weight inversely proportional to n (small n = more time)
            weight = 10.0 / n
            allocations[n] = weight
            total_weight += weight

        # Normalize to total time (reserve 10% for overhead)
        usable_time = total_time * 0.9
        for n in range(1, 201):
            allocations[n] = (allocations[n] / total_weight) * usable_time
            allocations[n] = max(allocations[n], 3.0)  # Minimum 3 seconds

        return allocations

    def solve_all(self, output_path: str = "submission.csv") -> float:
        """Solve all puzzles in parallel."""
        total_time = self.cfg.total_time_hours * 3600
        start_time = time.time()

        if self.verbose:
            print("=" * 70)
            print("SANTA 2025 - ULTIMATE SOLVER (Fixed - Independent & Parallel)")
            print("=" * 70)
            print(f"Cores: {self.cfg.num_cores}")
            print(f"Total time budget: {self.cfg.total_time_hours:.1f} hours")
            print(f"Restarts: small={self.cfg.restarts_small}, med={self.cfg.restarts_medium}, large={self.cfg.restarts_large}")
            print(f"Collision buffer: {self.cfg.collision_buffer}")
            print("=" * 70)
            print()

        # Compute time per puzzle
        time_alloc = self.compute_time_allocation(total_time)

        # Prepare work items
        work_items = []
        for n in range(1, 201):
            work_items.append((n, self.cfg, time_alloc[n], self.cfg.seed + n))

        # Progress tracking
        completed = 0
        if HAS_TQDM:
            pbar = tqdm(total=200, desc="Solving",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
        else:
            print("Solving puzzles...")

        # Parallel execution
        with ProcessPoolExecutor(max_workers=self.cfg.num_cores) as executor:
            futures = {executor.submit(solve_puzzle, *item): item[0] for item in work_items}

            for future in as_completed(futures):
                if not self.running:
                    break

                try:
                    n, placements, score = future.result()
                    self.solutions[n] = placements
                    completed += 1

                    if self.verbose:
                        total_score = self.compute_score()
                        if HAS_TQDM:
                            pbar.update(1)
                            pbar.set_postfix({'n': n, 'side': f'{score:.3f}', 'total': f'{total_score:.2f}'})
                        elif completed % 20 == 0:
                            print(f"  Progress: {completed}/200, Score: {total_score:.2f}")
                except Exception as e:
                    print(f"Error solving n={futures[future]}: {e}")

        if HAS_TQDM:
            pbar.close()

        # Save results
        self.save_submission(output_path)

        total_time_taken = time.time() - start_time
        final_score = self.compute_score()

        if self.verbose:
            print()
            print("=" * 70)
            print("COMPLETE")
            print(f"  Puzzles solved: {len(self.solutions)}")
            print(f"  Final score: {final_score:.4f}")
            print(f"  Time taken: {total_time_taken/3600:.2f} hours")
            print(f"  Baseline: 157.08")
            print(f"  Improvement: {(157.08 - final_score) / 157.08 * 100:.1f}%")
            print(f"  Output: {output_path}")
            print("=" * 70)

        return final_score

    def compute_score(self) -> float:
        """Compute total competition score."""
        total = 0.0
        for n, sol in self.solutions.items():
            if sol:
                side = bbox_side(sol)
                total += (side ** 2) / n
        return total

    def save_submission(self, output_path: str):
        """Save solutions to CSV."""
        with open(output_path, "w") as f:
            f.write("id,x,y,deg\n")
            for n in sorted(self.solutions.keys()):
                pos = self.solutions[n]
                if not pos:
                    continue
                polys = [make_tree(x, y, d) for x, y, d in pos]
                b = unary_union(polys).bounds
                for idx, (x, y, d) in enumerate(pos):
                    f.write(f"{n:03d}_{idx},s{x - b[0]:.6f},s{y - b[1]:.6f},s{d:.6f}\n")


def print_summary(solutions: Dict[int, List[Tuple[float, float, float]]]):
    """Print score summary."""
    print("=" * 60)
    print("SCORE SUMMARY")
    print("=" * 60)

    sides = {n: bbox_side(sol) for n, sol in solutions.items()}
    contribs = {n: (s ** 2) / n for n, s in sides.items()}
    total = sum(contribs.values())

    print(f"Total score: {total:.4f}")
    print(f"Baseline: 157.08")
    print(f"Improvement: {(157.08 - total) / 157.08 * 100:.1f}%")

    worst = sorted(contribs.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 worst contributors:")
    for n, c in worst:
        print(f"  n={n:3d}: side={sides[n]:.4f}, contrib={c:.4f}")

    print("\nScore by range:")
    for start, end in [(1, 10), (11, 50), (51, 100), (101, 150), (151, 200)]:
        range_score = sum(contribs.get(n, 0) for n in range(start, end + 1))
        print(f"  n={start:3d}-{end:3d}: {range_score:.4f}")

    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Santa 2025 Ultimate Solver (Fixed - Independent & Parallel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ultimate_solver.py --output submission.csv --cores 15
  python ultimate_solver.py --quick --cores 15
  python ultimate_solver.py --standard --cores 15
  python ultimate_solver.py --ultra --cores 15
"""
    )
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--cores", type=int, default=15, help="Number of CPU cores")
    parser.add_argument("--quick", action="store_true", help="Quick mode (~1-2 hours)")
    parser.add_argument("--standard", action="store_true", help="Standard mode (~6-12 hours)")
    parser.add_argument("--ultra", action="store_true", help="Ultra mode (~24-48 hours)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Select config
    if args.quick:
        cfg = quick_config()
        print("Using QUICK mode (~1-2 hours)")
    elif args.ultra:
        cfg = ultra_config()
        print("Using ULTRA mode (~24-48 hours)")
    else:
        cfg = standard_config()
        print("Using STANDARD mode (~6-12 hours)")

    cfg.num_cores = args.cores
    cfg.seed = args.seed

    print()
    if not HAS_TQDM:
        print("Note: Install tqdm for progress bars: pip install tqdm")
        print()

    # Solve
    solver = UltimateSolver(config=cfg, verbose=True)
    solver.solve_all(output_path=args.output)

    # Summary
    print()
    print_summary(solver.solutions)


if __name__ == "__main__":
    main()
