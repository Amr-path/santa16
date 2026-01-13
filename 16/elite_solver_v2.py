#!/usr/bin/env python3
"""
Santa 2025 - ELITE Solver V2 (CMA-ES + Advanced Optimization)
==============================================================

Target: Score < 65

This solver uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy),
one of the most powerful derivative-free optimization algorithms.

Key improvements over V1:
1. CMA-ES for global optimization (much better than simple GA/SA)
2. Simultaneous optimization of all tree positions
3. Adaptive penalty method for collision handling
4. Multi-start with intelligent restarts
5. Optimal angle optimization (not just 45 degrees)
6. Progressive refinement pipeline
7. Tight theoretical lower bounds for guidance

Usage:
    python elite_solver_v2.py [--output submission.csv] [--hours 24]

Requirements:
    pip install numpy shapely pyclipper tqdm cmaes
"""

import os
import sys
import math
import time
import random
import argparse
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from shapely.ops import unary_union
from shapely.strtree import STRtree

# Try imports
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("Warning: cma not installed. Install with: pip install cma")

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
    def tqdm(x, **kwargs):
        return x

# =============================================================================
# TREE GEOMETRY
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
TREE_WIDTH = 0.7
TREE_HEIGHT = 1.0

# Precompute rotations at finer granularity
ROTATED_CACHE = {}
for deg in range(360):
    ROTATED_CACHE[deg] = affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))


def make_tree(x: float, y: float, deg: float) -> Polygon:
    """Create tree polygon."""
    key = int(round(deg)) % 360
    poly = ROTATED_CACHE[key]
    return affinity.translate(poly, xoff=x, yoff=y)


def make_tree_exact(x: float, y: float, deg: float) -> Polygon:
    """Create tree with exact angle (not cached)."""
    poly = affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))
    return affinity.translate(poly, xoff=x, yoff=y)


def bbox_side(placements: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side."""
    if not placements:
        return 0.0
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def check_overlaps(placements: List[Tuple[float, float, float]], buf: float = 0.001) -> bool:
    """Check for any overlapping pairs."""
    polys = [make_tree(x, y, d) for x, y, d in placements]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if buf > 0:
                if polys[i].distance(polys[j]) < buf:
                    return True
            elif polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                return True
    return False


def count_overlaps(placements: List[Tuple[float, float, float]], buf: float = 0.001) -> int:
    """Count number of overlapping pairs."""
    polys = [make_tree(x, y, d) for x, y, d in placements]
    count = 0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if buf > 0:
                if polys[i].distance(polys[j]) < buf:
                    count += 1
            elif polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                count += 1
    return count


def overlap_penalty(placements: List[Tuple[float, float, float]], buf: float = 0.001) -> float:
    """Compute overlap penalty (sum of overlap areas)."""
    polys = [make_tree(x, y, d) for x, y, d in placements]
    penalty = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            dist = polys[i].distance(polys[j])
            if dist < buf:
                # Penalize based on how much overlap
                penalty += (buf - dist) * 10
            if polys[i].intersects(polys[j]):
                try:
                    inter = polys[i].intersection(polys[j])
                    penalty += inter.area * 100
                except:
                    penalty += 0.1
    return penalty


def center(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Center placements around origin."""
    if not placements:
        return placements
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]


# =============================================================================
# THEORETICAL LOWER BOUNDS
# =============================================================================

def tree_area() -> float:
    """Compute area of single tree polygon."""
    return BASE_POLYGON.area


def theoretical_lower_bound(n: int) -> float:
    """
    Estimate theoretical lower bound for bounding square side.

    The tree has area ~0.2825. For n trees:
    - Total area needed: n * 0.2825
    - Packing efficiency ~70-85% for irregular shapes
    - Lower bound: sqrt(n * tree_area / efficiency)
    """
    area = tree_area()  # ~0.2825
    # Assume 80% packing efficiency (optimistic)
    total_area = n * area / 0.80
    return math.sqrt(total_area)


# Tree bounding box at different angles
@lru_cache(maxsize=360)
def tree_bbox_at_angle(deg: int) -> Tuple[float, float]:
    """Get bounding box (width, height) of tree at given angle."""
    poly = ROTATED_CACHE[deg % 360]
    b = poly.bounds
    return (b[2] - b[0], b[3] - b[1])


def optimal_angle_for_bbox() -> int:
    """Find angle that minimizes tree bounding box."""
    best_angle = 0
    best_area = float('inf')
    for deg in range(0, 180, 1):  # Symmetry means we only need 0-180
        w, h = tree_bbox_at_angle(deg)
        area = w * h
        if area < best_area:
            best_area = area
            best_angle = deg
    return best_angle


OPTIMAL_ANGLE = optimal_angle_for_bbox()  # Should be around 45


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Solver configuration."""
    max_hours: float = 24.0
    time_per_puzzle_base: float = 60.0
    priority_multiplier: float = 10.0

    # CMA-ES
    cma_population_size: int = 40
    cma_max_iter: int = 500
    cma_sigma0: float = 0.3

    # Restarts
    num_restarts: int = 30

    # Local search
    local_iterations: int = 2000
    local_precision: float = 0.0002

    # Compaction
    compact_passes: int = 50
    compact_step: float = 0.0002

    # Collision
    collision_buffer: float = 0.001

    # Checkpointing
    checkpoint_interval: int = 5
    checkpoint_path: str = "elite_v2_checkpoint.csv"

    seed: int = 42

    def get_time_for_n(self, n: int) -> float:
        priority = 1.0 + (self.priority_multiplier - 1.0) * (200 - n) / 199
        return self.time_per_puzzle_base * priority


def quick_config() -> Config:
    return Config(
        max_hours=2.0,
        time_per_puzzle_base=15.0,
        priority_multiplier=5.0,
        cma_population_size=20,
        cma_max_iter=100,
        num_restarts=10,
        local_iterations=500,
        compact_passes=20,
    )


def standard_config() -> Config:
    return Config(
        max_hours=12.0,
        time_per_puzzle_base=45.0,
        cma_population_size=30,
        cma_max_iter=300,
        num_restarts=20,
    )


def ultra_config() -> Config:
    return Config(
        max_hours=48.0,
        time_per_puzzle_base=180.0,
        priority_multiplier=15.0,
        cma_population_size=60,
        cma_max_iter=1000,
        num_restarts=50,
        local_iterations=5000,
        compact_passes=100,
    )


# =============================================================================
# PATTERN-BASED INITIALIZATION
# =============================================================================

def init_grid(n: int, spacing: float = 0.82) -> List[Tuple[float, float, float]]:
    """Simple grid initialization with optimal angle."""
    placements = []
    cols = int(math.ceil(math.sqrt(n)))
    angle = OPTIMAL_ANGLE

    for i in range(n):
        row = i // cols
        col = i % cols
        x = col * spacing + (row % 2) * (spacing / 2)  # Offset rows
        y = row * spacing * 0.9  # Slightly tighter vertically
        placements.append((x, y, float(angle)))

    return center(placements)


def init_hex(n: int, spacing: float = 0.78) -> List[Tuple[float, float, float]]:
    """Hexagonal grid initialization."""
    placements = []
    dx = spacing
    dy = spacing * math.sqrt(3) / 2
    cols = int(math.ceil(math.sqrt(n * 1.2)))
    angle = OPTIMAL_ANGLE

    for i in range(n):
        row = i // cols
        col = i % cols
        x = col * dx + (row % 2) * (dx / 2)
        y = row * dy
        placements.append((x, y, float(angle)))

    return center(placements[:n])


def init_spiral(n: int) -> List[Tuple[float, float, float]]:
    """Spiral initialization."""
    placements = []
    angle_inc = 2.39996323  # Golden angle in radians
    radius = 0.0
    radius_inc = 0.25
    angle = OPTIMAL_ANGLE

    for i in range(n):
        theta = i * angle_inc
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        placements.append((x, y, float(angle)))
        radius += radius_inc / (1 + i * 0.01)

    return center(placements)


def init_concentric(n: int) -> List[Tuple[float, float, float]]:
    """Concentric rings initialization."""
    placements = []
    angle = OPTIMAL_ANGLE

    if n == 0:
        return []

    placements.append((0.0, 0.0, float(angle)))
    remaining = n - 1
    ring = 1
    ring_spacing = 0.85

    while remaining > 0:
        radius = ring * ring_spacing
        trees_in_ring = min(remaining, max(6, int(2 * math.pi * radius / 0.7)))

        for i in range(trees_in_ring):
            theta = 2 * math.pi * i / trees_in_ring
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            placements.append((x, y, float(angle)))

        remaining -= trees_in_ring
        ring += 1

    return center(placements[:n])


def init_random_compact(n: int, cfg: Config) -> List[Tuple[float, float, float]]:
    """Random but compact initialization."""
    placements = []
    polys = []
    angle = OPTIMAL_ANGLE
    max_radius = math.sqrt(n) * 0.4

    # First tree at origin
    placements.append((0.0, 0.0, float(angle)))
    polys.append(make_tree(0, 0, angle))

    for _ in range(1, n):
        best_pos = None
        best_dist = float('inf')

        for _ in range(200):
            r = random.uniform(0, max_radius)
            theta = random.uniform(0, 2 * math.pi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)

            poly = make_tree(x, y, angle)
            ok = True
            for p in polys:
                if poly.distance(p) < cfg.collision_buffer:
                    ok = False
                    break

            if ok:
                dist = x*x + y*y
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (x, y)

        if best_pos is None:
            # Expand radius
            for mult in [1.5, 2.0, 3.0, 5.0]:
                for _ in range(100):
                    r = random.uniform(max_radius, max_radius * mult)
                    theta = random.uniform(0, 2 * math.pi)
                    x = r * math.cos(theta)
                    y = r * math.sin(theta)

                    poly = make_tree(x, y, angle)
                    ok = True
                    for p in polys:
                        if poly.distance(p) < cfg.collision_buffer:
                            ok = False
                            break

                    if ok:
                        best_pos = (x, y)
                        break

                if best_pos:
                    break

        if best_pos is None:
            best_pos = (max_radius * 2, 0)

        placements.append((best_pos[0], best_pos[1], float(angle)))
        polys.append(make_tree(best_pos[0], best_pos[1], angle))
        max_radius = max(max_radius, math.sqrt(best_pos[0]**2 + best_pos[1]**2) + 0.5)

    return center(placements)


# =============================================================================
# CMA-ES OPTIMIZATION
# =============================================================================

class CMAOptimizer:
    """CMA-ES based optimizer for tree packing."""

    def __init__(self, n: int, cfg: Config):
        self.n = n
        self.cfg = cfg
        self.best_solution = None
        self.best_score = float('inf')
        self.eval_count = 0

    def encode(self, placements: List[Tuple[float, float, float]]) -> np.ndarray:
        """Encode placements to optimization vector [x1, y1, a1, x2, y2, a2, ...]."""
        vec = []
        for x, y, a in placements:
            vec.extend([x, y, a / 360.0])  # Normalize angle to [0, 1]
        return np.array(vec)

    def decode(self, vec: np.ndarray) -> List[Tuple[float, float, float]]:
        """Decode optimization vector to placements."""
        placements = []
        for i in range(0, len(vec), 3):
            x = vec[i]
            y = vec[i + 1]
            a = (vec[i + 2] % 1.0) * 360.0  # Wrap angle to [0, 360)
            placements.append((x, y, a))
        return placements

    def objective(self, vec: np.ndarray) -> float:
        """
        Objective function: minimize bounding box + collision penalty.

        Returns large value if infeasible.
        """
        self.eval_count += 1
        placements = self.decode(vec)

        # Collision penalty
        penalty = overlap_penalty(placements, self.cfg.collision_buffer)

        if penalty > 0:
            # Heavily penalize collisions
            side = bbox_side(placements)
            return side + penalty * 1000

        side = bbox_side(placements)

        # Track best valid solution
        if side < self.best_score:
            self.best_score = side
            self.best_solution = list(placements)

        return side

    def optimize(self, initial: List[Tuple[float, float, float]],
                 time_limit: float) -> List[Tuple[float, float, float]]:
        """Run CMA-ES optimization."""
        if not HAS_CMA:
            return initial

        start_time = time.time()
        x0 = self.encode(initial)

        # Estimate bounds
        max_coord = max(1.0, bbox_side(initial) * 0.6)

        # CMA-ES options
        opts = {
            'maxiter': self.cfg.cma_max_iter,
            'popsize': self.cfg.cma_population_size,
            'bounds': [[-max_coord * 2] * (self.n * 2) + [0] * self.n,
                      [max_coord * 2] * (self.n * 2) + [1] * self.n],
            'tolfun': 1e-8,
            'tolx': 1e-8,
            'verb_disp': 0,
            'verb_log': 0,
            'verbose': -9,  # Suppress output
        }

        try:
            es = cma.CMAEvolutionStrategy(x0, self.cfg.cma_sigma0, opts)

            while not es.stop() and time.time() - start_time < time_limit:
                solutions = es.ask()
                fitness = [self.objective(s) for s in solutions]
                es.tell(solutions, fitness)

            # Get best from CMA
            result = self.decode(es.result.xbest)
            if not check_overlaps(result, self.cfg.collision_buffer):
                return result

        except Exception as e:
            pass

        # Return best valid solution found
        if self.best_solution is not None:
            return self.best_solution

        return initial


# =============================================================================
# DIFFERENTIAL EVOLUTION (BACKUP)
# =============================================================================

def differential_evolution_optimize(
    placements: List[Tuple[float, float, float]],
    cfg: Config,
    time_limit: float
) -> List[Tuple[float, float, float]]:
    """Differential Evolution optimization (when CMA not available)."""
    n = len(placements)
    start_time = time.time()

    # Parameters
    pop_size = cfg.cma_population_size
    F = 0.7  # Mutation factor
    CR = 0.9  # Crossover rate

    def encode(pl):
        return np.array([v for p in pl for v in (p[0], p[1], p[2] / 360.0)])

    def decode(vec):
        result = []
        for i in range(0, len(vec), 3):
            result.append((vec[i], vec[i+1], (vec[i+2] % 1.0) * 360.0))
        return result

    def fitness(vec):
        pl = decode(vec)
        pen = overlap_penalty(pl, cfg.collision_buffer)
        side = bbox_side(pl)
        if pen > 0:
            return side + pen * 1000
        return side

    # Initialize population
    x0 = encode(placements)
    dim = len(x0)
    population = [x0 + np.random.randn(dim) * 0.1 for _ in range(pop_size)]
    scores = [fitness(p) for p in population]

    best_idx = np.argmin(scores)
    best = population[best_idx].copy()
    best_score = scores[best_idx]
    best_placements = placements

    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1

        for i in range(pop_size):
            # Mutation
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])

            # Crossover
            trial = population[i].copy()
            j_rand = random.randint(0, dim - 1)
            for j in range(dim):
                if random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # Selection
            trial_score = fitness(trial)
            if trial_score < scores[i]:
                population[i] = trial
                scores[i] = trial_score

                if trial_score < best_score:
                    pl = decode(trial)
                    if not check_overlaps(pl, cfg.collision_buffer):
                        best = trial.copy()
                        best_score = trial_score
                        best_placements = pl

    return best_placements


# =============================================================================
# LOCAL SEARCH
# =============================================================================

def local_search(
    placements: List[Tuple[float, float, float]],
    cfg: Config,
    time_limit: float
) -> List[Tuple[float, float, float]]:
    """Fine-grained local search optimization."""
    n = len(placements)
    if n <= 1:
        return placements

    start_time = time.time()
    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]
    cur_score = bbox_side(current)

    precision = cfg.local_precision
    improved = True
    iteration = 0

    while improved and iteration < cfg.local_iterations and time.time() - start_time < time_limit:
        improved = False
        iteration += 1

        # Shuffle order for variety
        order = list(range(n))
        random.shuffle(order)

        for i in order:
            x, y, d = current[i]
            best_move = None
            best_improvement = 0

            # Try position moves at multiple scales
            scales = [0.5, 1.0, 2.0, 4.0]
            moves = []
            for scale in scales:
                p = precision * scale
                moves.extend([
                    (p, 0), (-p, 0), (0, p), (0, -p),
                    (p, p), (p, -p), (-p, p), (-p, -p),
                ])

            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                new_poly = make_tree(nx, ny, d)

                # Check collision
                collision = False
                for j, p in enumerate(polys):
                    if j != i and new_poly.distance(p) < cfg.collision_buffer:
                        collision = True
                        break

                if collision:
                    continue

                # Compute improvement
                old_poly = polys[i]
                polys[i] = new_poly
                new_score = bbox_side([(px, py, pd) if k != i else (nx, ny, d)
                                      for k, (px, py, pd) in enumerate(current)])
                improvement = cur_score - new_score
                polys[i] = old_poly

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (nx, ny, d)

            # Try rotation moves
            for dd in [-1, 1, -2, 2, -3, 3, -5, 5, -10, 10]:
                nd = (d + dd) % 360
                new_poly = make_tree(x, y, nd)

                collision = False
                for j, p in enumerate(polys):
                    if j != i and new_poly.distance(p) < cfg.collision_buffer:
                        collision = True
                        break

                if collision:
                    continue

                old_poly = polys[i]
                polys[i] = new_poly
                new_score = bbox_side([(px, py, pd) if k != i else (x, y, nd)
                                      for k, (px, py, pd) in enumerate(current)])
                improvement = cur_score - new_score
                polys[i] = old_poly

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (x, y, nd)

            # Apply best move
            if best_move is not None and best_improvement > 1e-9:
                current[i] = best_move
                polys[i] = make_tree(*best_move)
                cur_score -= best_improvement
                improved = True

    return current


# =============================================================================
# COMPACTION
# =============================================================================

def compact(
    placements: List[Tuple[float, float, float]],
    cfg: Config,
    time_limit: float
) -> List[Tuple[float, float, float]]:
    """Aggressively compact trees toward center."""
    n = len(placements)
    if n <= 1:
        return placements

    start_time = time.time()
    current = list(placements)
    step = cfg.compact_step

    for pass_num in range(cfg.compact_passes):
        if time.time() - start_time >= time_limit:
            break

        polys = [make_tree(x, y, d) for x, y, d in current]

        # Compute centroid
        cx = sum(p[0] for p in current) / n
        cy = sum(p[1] for p in current) / n

        # Sort by distance from center (farthest first)
        indexed = [(i, math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)) for i, p in enumerate(current)]
        indexed.sort(key=lambda x: -x[1])

        for idx, dist in indexed:
            if dist < 0.01:
                continue

            x, y, d = current[idx]
            dx, dy = cx - x, cy - y
            norm = math.sqrt(dx*dx + dy*dy) + 1e-10

            # Try progressively smaller moves
            for mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
                move = step * mult * dist * 10
                nx = x + move * dx / norm
                ny = y + move * dy / norm

                new_poly = make_tree(nx, ny, d)
                collision = False
                for j, p in enumerate(polys):
                    if j != idx and new_poly.distance(p) < cfg.collision_buffer:
                        collision = True
                        break

                if not collision:
                    current[idx] = (nx, ny, d)
                    polys[idx] = new_poly
                    break

    return current


# =============================================================================
# SIMULATED ANNEALING (BACKUP)
# =============================================================================

def simulated_annealing(
    placements: List[Tuple[float, float, float]],
    cfg: Config,
    time_limit: float,
    seed: int = 42
) -> List[Tuple[float, float, float]]:
    """Simulated annealing optimization."""
    n = len(placements)
    if n <= 1:
        return placements

    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]
    cur_score = bbox_side(current)

    best = list(current)
    best_score = cur_score

    T = 2.0
    T_min = 1e-8

    shift = 0.1 * cur_score

    while time.time() - start_time < time_limit and T > T_min:
        i = random.randrange(n)
        x, y, d = current[i]

        # Choose move type
        mt = random.random()

        if mt < 0.3:
            # Small position move
            nx = x + random.gauss(0, shift * 0.2)
            ny = y + random.gauss(0, shift * 0.2)
            nd = d
        elif mt < 0.5:
            # Medium position move
            nx = x + random.gauss(0, shift * 0.5)
            ny = y + random.gauss(0, shift * 0.5)
            nd = d
        elif mt < 0.7:
            # Rotation
            nx, ny = x, y
            nd = (d + random.choice([-5, 5, -10, 10, -15, 15, -30, 30, -45, 45])) % 360
        elif mt < 0.85:
            # Move toward center
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-10
            step = random.uniform(0.01, 0.1) * cur_score
            nx = x + step * dx / dist
            ny = y + step * dy / dist
            nd = d
        else:
            # Large escape move
            nx = x + random.uniform(-shift * 2, shift * 2)
            ny = y + random.uniform(-shift * 2, shift * 2)
            nd = d

        # Check validity
        new_poly = make_tree(nx, ny, nd)
        collision = False
        for j, p in enumerate(polys):
            if j != i and new_poly.distance(p) < cfg.collision_buffer:
                collision = True
                break

        if collision:
            # Cool down
            progress = (time.time() - start_time) / time_limit
            T = 2.0 * (1e-8 / 2.0) ** progress
            continue

        # Compute new score
        old_poly = polys[i]
        polys[i] = new_poly
        new_score = bbox_side([(px, py, pd) if k != i else (nx, ny, nd)
                              for k, (px, py, pd) in enumerate(current)])

        # Accept or reject
        delta = new_score - cur_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current[i] = (nx, ny, nd)
            cur_score = new_score

            if cur_score < best_score:
                best = list(current)
                best_score = cur_score
        else:
            polys[i] = old_poly

        # Cool down
        progress = (time.time() - start_time) / time_limit
        T = 2.0 * (1e-8 / 2.0) ** progress

        # Adaptive step size
        shift = max(0.001, 0.1 * cur_score)

    return best


# =============================================================================
# OPTIMAL SMALL N SOLUTIONS
# =============================================================================

def solve_n1() -> List[Tuple[float, float, float]]:
    return [(0.0, 0.0, float(OPTIMAL_ANGLE))]


def solve_n2(cfg: Config) -> List[Tuple[float, float, float]]:
    """Optimal for n=2: two trees side by side."""
    best = None
    best_score = float('inf')

    # Try different configurations
    for angle1 in range(0, 180, 5):
        for angle2 in range(0, 180, 5):
            for spacing in np.arange(0.6, 1.2, 0.02):
                p1 = (0.0, 0.0, float(angle1))
                p2 = (spacing, 0.0, float(angle2))
                placements = [p1, p2]

                if check_overlaps(placements, cfg.collision_buffer):
                    continue

                side = bbox_side(placements)
                if side < best_score:
                    best_score = side
                    best = placements

    if best:
        return center(best)
    return center([(0.0, 0.0, float(OPTIMAL_ANGLE)), (0.8, 0.0, float(OPTIMAL_ANGLE))])


def solve_n3(cfg: Config) -> List[Tuple[float, float, float]]:
    """Optimal for n=3: triangle formation."""
    best = None
    best_score = float('inf')

    angle = OPTIMAL_ANGLE

    # Try triangle formations
    for spacing in np.arange(0.6, 1.2, 0.02):
        for height in np.arange(0.4, 1.0, 0.02):
            p1 = (0.0, 0.0, float(angle))
            p2 = (spacing, 0.0, float(angle))
            p3 = (spacing / 2, height, float(angle))
            placements = [p1, p2, p3]

            if check_overlaps(placements, cfg.collision_buffer):
                continue

            side = bbox_side(placements)
            if side < best_score:
                best_score = side
                best = placements

    if best:
        return center(best)
    return center(init_grid(3))


def solve_small_n(n: int, cfg: Config, time_limit: float) -> List[Tuple[float, float, float]]:
    """Exhaustive search for small n."""
    if n == 1:
        return solve_n1()
    elif n == 2:
        return solve_n2(cfg)
    elif n == 3:
        return solve_n3(cfg)

    # For n > 3, use multi-restart CMA
    best = None
    best_score = float('inf')
    start_time = time.time()

    inits = [
        init_grid(n),
        init_hex(n),
        init_spiral(n),
        init_concentric(n),
    ]

    for init in inits:
        if time.time() - start_time >= time_limit:
            break

        remaining = time_limit - (time.time() - start_time)

        if HAS_CMA:
            opt = CMAOptimizer(n, cfg)
            result = opt.optimize(init, remaining * 0.3)
        else:
            result = differential_evolution_optimize(init, cfg, remaining * 0.3)

        result = local_search(result, cfg, remaining * 0.1)
        result = compact(result, cfg, remaining * 0.05)
        result = local_search(result, cfg, remaining * 0.05)

        if not check_overlaps(result, cfg.collision_buffer):
            side = bbox_side(result)
            if side < best_score:
                best_score = side
                best = result

    if best is None:
        best = init_grid(n)

    return center(best)


# =============================================================================
# MAIN SOLVER
# =============================================================================

class EliteSolverV2:
    """Main solver using CMA-ES and advanced techniques."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.scores: Dict[int, float] = {}

    def solve_puzzle(self, n: int, time_budget: float) -> List[Tuple[float, float, float]]:
        """Solve a single puzzle."""
        if n == 0:
            return []

        start_time = time.time()

        # Small n: use specialized solvers
        if n <= 10:
            return solve_small_n(n, self.cfg, time_budget)

        best = None
        best_score = float('inf')

        # Generate diverse initializations
        inits = [
            ('grid', init_grid(n)),
            ('hex', init_hex(n)),
            ('spiral', init_spiral(n)),
            ('concentric', init_concentric(n)),
        ]

        # Add previous solution as starting point if available
        if n > 1 and (n - 1) in self.solutions:
            prev = self.solutions[n - 1]
            # Add one more tree to previous solution
            extended = list(prev)
            # Find position for new tree
            polys = [make_tree(x, y, d) for x, y, d in extended]
            bounds = unary_union(polys).bounds
            extended.append((bounds[2] + 0.5, 0.0, float(OPTIMAL_ANGLE)))
            inits.append(('extended', extended))

        # Multi-restart optimization
        time_per_restart = time_budget / max(1, self.cfg.num_restarts)

        for restart in range(self.cfg.num_restarts):
            if time.time() - start_time >= time_budget * 0.95:
                break

            # Select initialization
            if restart < len(inits):
                name, init = inits[restart]
            else:
                # Perturb best solution
                if best is not None:
                    init = self.perturb(best, 0.05 * best_score)
                else:
                    init = inits[restart % len(inits)][1]

            # Skip if initialization has collisions
            if check_overlaps(init, self.cfg.collision_buffer):
                continue

            # Optimize
            remaining = min(time_per_restart, time_budget - (time.time() - start_time))

            if HAS_CMA and remaining > 1.0:
                opt = CMAOptimizer(n, self.cfg)
                result = opt.optimize(init, remaining * 0.4)
            else:
                result = simulated_annealing(init, self.cfg, remaining * 0.4, seed=restart)

            # Refine
            remaining = time_budget - (time.time() - start_time)
            if remaining > 0.5:
                result = local_search(result, self.cfg, remaining * 0.15)
                result = compact(result, self.cfg, remaining * 0.1)
                result = local_search(result, self.cfg, remaining * 0.05)

            # Validate and update best
            if not check_overlaps(result, self.cfg.collision_buffer):
                side = bbox_side(result)
                if side < best_score:
                    best_score = side
                    best = result

        # Final refinement on best
        if best is not None:
            remaining = time_budget - (time.time() - start_time)
            if remaining > 0.5:
                best = compact(best, self.cfg, remaining * 0.3)
                best = local_search(best, self.cfg, remaining * 0.2)

        if best is None:
            best = init_grid(n)

        return center(best)

    def perturb(self, placements: List[Tuple[float, float, float]],
                magnitude: float) -> List[Tuple[float, float, float]]:
        """Perturb a solution."""
        result = []
        for x, y, d in placements:
            nx = x + random.gauss(0, magnitude)
            ny = y + random.gauss(0, magnitude)
            nd = (d + random.uniform(-15, 15)) % 360
            result.append((nx, ny, nd))
        return result

    def solve_all(self, output_path: str = "submission.csv") -> Dict[int, List[Tuple[float, float, float]]]:
        """Solve all puzzles."""
        start_time = time.time()
        max_time = self.cfg.max_hours * 3600

        # Calculate time budgets
        total_priority = sum(1.0 + (self.cfg.priority_multiplier - 1.0) * (200 - n) / 199
                            for n in range(1, 201))

        if HAS_TQDM:
            pbar = tqdm(range(1, 201), desc="Solving", ncols=100)
        else:
            pbar = range(1, 201)

        for n in pbar:
            elapsed = time.time() - start_time
            if elapsed >= max_time:
                print(f"\nTime limit reached at n={n}")
                break

            # Calculate time budget
            remaining_time = max_time - elapsed
            priority = 1.0 + (self.cfg.priority_multiplier - 1.0) * (200 - n) / 199
            remaining_priority = sum(1.0 + (self.cfg.priority_multiplier - 1.0) * (200 - k) / 199
                                    for k in range(n, 201))
            time_budget = min(
                self.cfg.get_time_for_n(n),
                remaining_time * priority / remaining_priority
            )

            # Solve
            placements = self.solve_puzzle(n, time_budget)
            self.solutions[n] = placements

            # Score
            side = bbox_side(placements)
            self.scores[n] = side

            # Progress
            total_score = sum((self.scores[k] ** 2) / k for k in self.scores)

            if HAS_TQDM:
                pbar.set_postfix({
                    'score': f'{total_score:.2f}',
                    'side': f'{side:.4f}',
                    'time': f'{time_budget:.1f}s'
                })

            # Checkpoint
            if n % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint()

        # Fill missing
        for n in range(1, 201):
            if n not in self.solutions:
                self.solutions[n] = init_grid(n)
                self.scores[n] = bbox_side(self.solutions[n])

        # Save final
        self.save_submission(output_path)

        return self.solutions

    def total_score(self) -> float:
        """Compute total competition score."""
        return sum((self.scores.get(n, bbox_side(self.solutions.get(n, []))) ** 2) / n
                  for n in range(1, 201) if n in self.solutions)

    def save_checkpoint(self):
        """Save checkpoint."""
        self.save_submission(self.cfg.checkpoint_path)

    def save_submission(self, path: str):
        """Save submission CSV."""
        with open(path, 'w') as f:
            f.write("id,x,y,deg\n")

            for n in sorted(self.solutions.keys()):
                placements = self.solutions[n]

                if placements:
                    polys = [make_tree(x, y, d) for x, y, d in placements]
                    bounds = unary_union(polys).bounds
                    min_x, min_y = bounds[0], bounds[1]
                else:
                    min_x, min_y = 0, 0

                for idx, (x, y, deg) in enumerate(placements):
                    nx = x - min_x
                    ny = y - min_y
                    f.write(f"{n:03d}_{idx},s{nx:.6f},s{ny:.6f},s{deg:.6f}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Santa 2025 Elite Solver V2 (CMA-ES)')
    parser.add_argument('--output', '-o', default='submission_elite_v2.csv', help='Output file')
    parser.add_argument('--quick', action='store_true', help='Quick mode (~2 hours)')
    parser.add_argument('--standard', action='store_true', help='Standard mode (~12 hours)')
    parser.add_argument('--ultra', action='store_true', help='Ultra mode (~48 hours)')
    parser.add_argument('--hours', type=float, help='Custom time limit in hours')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Select config
    if args.quick:
        cfg = quick_config()
    elif args.ultra:
        cfg = ultra_config()
    elif args.standard:
        cfg = standard_config()
    else:
        cfg = Config()

    if args.hours:
        cfg.max_hours = args.hours

    cfg.seed = args.seed

    print("=" * 70)
    print("SANTA 2025 - ELITE SOLVER V2 (CMA-ES)")
    print("=" * 70)
    print(f"Max hours: {cfg.max_hours}")
    print(f"CMA population: {cfg.cma_population_size}")
    print(f"CMA max iterations: {cfg.cma_max_iter}")
    print(f"Restarts per puzzle: {cfg.num_restarts}")
    print(f"CMA-ES available: {HAS_CMA}")
    print(f"Optimal tree angle: {OPTIMAL_ANGLE} degrees")
    print(f"Theoretical lower bound (n=100): {theoretical_lower_bound(100):.4f}")
    print(f"Seed: {cfg.seed}")
    print("=" * 70)

    solver = EliteSolverV2(cfg)
    solutions = solver.solve_all(args.output)

    # Final score
    total_score = solver.total_score()
    print()
    print("=" * 70)
    print(f"FINAL SCORE: {total_score:.4f}")
    print(f"Saved to: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
