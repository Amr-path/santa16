#!/usr/bin/env python3
"""
Santa 2025 - FINAL BOSS SOLVER (Extreme Optimized)
===================================================

Advanced solver using state-of-the-art algorithms from math/physics papers:
- Levy Flight Simulated Annealing (better exploration)
- Basin Hopping (global optimization with local minimization)
- Differential Evolution (population-based optimization)
- Variable Neighborhood Search (VNS)
- Force-directed physics compaction
- Nelder-Mead simplex for fine-tuning
- Adaptive reheating SA (escape local minima)
- Fermat spiral + convex hull packing

Default: EXTREME mode with 4 cores

Usage:
    python final_boss_solver.py                          # Extreme mode, 4 cores
    python final_boss_solver.py --output submission.csv
    python final_boss_solver.py --quick                  # Fast test
    python final_boss_solver.py --cores 8                # More cores

Target: Score < 60 (from baseline 157.08)
"""

import os
import sys
import math
import time
import random
import argparse
import signal
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from functools import lru_cache
import heapq

import numpy as np
from shapely.geometry import Polygon, Point
from shapely import affinity
from shapely.ops import unary_union
from shapely.prepared import prep

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# CONFIGURATION - EXTREME MODE DEFAULT
# =============================================================================

@dataclass
class Config:
    """Solver configuration - EXTREME mode by default."""
    # Parallelism - 4 cores default
    num_cores: int = 4

    # Time budgets
    total_time_hours: float = 72.0  # Extreme: 72 hours

    # Multi-restart per puzzle (aggressive)
    restarts_small: int = 600    # n=1-10: critical range
    restarts_medium: int = 300   # n=11-50
    restarts_large: int = 150    # n=51-200

    # Placement
    placement_attempts: int = 500

    # Collision
    collision_buffer: float = 0.005  # Tighter packing

    # Simulated Annealing with Levy Flight
    sa_temp_initial: float = 3.0
    sa_temp_final: float = 1e-12
    sa_reheat_threshold: float = 0.02  # Reheat if stuck
    levy_alpha: float = 1.5  # Levy exponent (1 < alpha < 2)

    # Basin Hopping
    basin_hop_iterations: int = 25
    basin_hop_step: float = 0.15

    # Differential Evolution
    de_population: int = 20
    de_mutation: float = 0.8
    de_crossover: float = 0.9

    # Local search (Nelder-Mead inspired)
    local_precision: float = 0.0001
    local_iterations: int = 5000
    simplex_iterations: int = 200

    # Compacting (force-directed)
    compact_passes: int = 100
    compact_step: float = 0.00005
    gravity_strength: float = 0.12
    repulsion_strength: float = 0.08

    # Variable Neighborhood Search
    vns_neighborhoods: int = 5
    vns_iterations: int = 50

    seed: int = 42


def quick_config() -> Config:
    """Quick test mode (~30 min - 1 hour)."""
    return Config(
        total_time_hours=1.0,
        restarts_small=80,
        restarts_medium=40,
        restarts_large=20,
        placement_attempts=150,
        local_iterations=1500,
        compact_passes=30,
        basin_hop_iterations=8,
        de_population=8,
        vns_iterations=15,
    )


def standard_config() -> Config:
    """Standard mode (~4-8 hours)."""
    return Config(
        total_time_hours=8.0,
        restarts_small=250,
        restarts_medium=120,
        restarts_large=60,
        placement_attempts=300,
        local_iterations=3000,
        compact_passes=60,
        basin_hop_iterations=15,
    )


def extreme_config() -> Config:
    """Extreme quality mode (default) - 72 hours target < 55."""
    return Config()  # Default is already extreme


# =============================================================================
# GEOMETRY WITH OPTIMIZATIONS
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
TREE_BOUNDS = BASE_POLYGON.bounds  # (minx, miny, maxx, maxy)
TREE_RADIUS = max(
    math.sqrt(x*x + y*y) for x, y in TREE_COORDS
)  # ~0.86

# Pre-compute rotated polygons with finer granularity
ROTATED_POLYGONS = {}
PREPARED_POLYGONS = {}
for deg in range(360):
    poly = affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))
    ROTATED_POLYGONS[deg] = poly
    PREPARED_POLYGONS[deg] = prep(poly)


def make_tree(x: float, y: float, deg: float) -> Polygon:
    """Create tree polygon at position with rotation."""
    snapped = int(round(deg)) % 360
    poly = ROTATED_POLYGONS[snapped]
    if x != 0 or y != 0:
        return affinity.translate(poly, xoff=x, yoff=y)
    return poly


def fast_collides(x: float, y: float, deg: int,
                  others_data: List[Tuple[float, float, int, Polygon]],
                  buf: float = 0.0) -> bool:
    """Fast collision check using bounding box pre-filter."""
    # Quick bounding box check first
    r = TREE_RADIUS + buf
    for ox, oy, od, opoly in others_data:
        # Fast distance check
        dx = x - ox
        dy = y - oy
        if dx*dx + dy*dy > (2*r + 0.1)**2:
            continue

        # Full collision check
        poly = make_tree(x, y, deg)
        if buf > 0:
            if poly.distance(opoly) < buf:
                return True
        elif poly.intersects(opoly) and not poly.touches(opoly):
            return True
    return False


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
# LEVY FLIGHT - Heavy-tailed distribution for better exploration
# =============================================================================

def levy_flight(alpha: float = 1.5, size: float = 1.0) -> float:
    """
    Generate Levy flight step using Mantegna's algorithm.
    From: "Fast, accurate algorithm for numerical simulation of Levy stable
    stochastic processes" - Mantegna (1994)

    Heavy-tailed distribution allows occasional large jumps to escape local minima.
    """
    # Mantegna's algorithm for Levy stable distribution
    sigma_u = (
        math.gamma(1 + alpha) * math.sin(math.pi * alpha / 2) /
        (math.gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2))
    ) ** (1 / alpha)

    u = random.gauss(0, sigma_u)
    v = abs(random.gauss(0, 1))

    step = u / (v ** (1 / alpha))
    return step * size


# =============================================================================
# INITIAL PLACEMENT STRATEGIES (Enhanced)
# =============================================================================

def fermat_spiral_placement(n: int, cfg: Config, seed: int) -> List[Tuple[float, float, float]]:
    """
    Fermat spiral placement - optimal for circular packing.
    From: Vogel's model of phyllotaxis (sunflower seed arrangement)

    Points are placed at: r = c * sqrt(k), theta = k * golden_angle
    This achieves near-optimal packing density.
    """
    random.seed(seed)

    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, random.randint(0, 359))]

    buf = cfg.collision_buffer
    placements = []
    polys = []

    golden_angle = math.pi * (3 - math.sqrt(5))  # ~137.5 degrees
    c = 0.55  # Spacing constant tuned for tree shape

    k = 0
    attempts = 0
    max_attempts = n * 10

    while len(placements) < n and attempts < max_attempts:
        # Fermat spiral formula
        r = c * math.sqrt(k)
        theta = k * golden_angle

        x = r * math.cos(theta)
        y = r * math.sin(theta)

        # Try multiple rotations
        best_deg = None
        best_score = float('inf')

        for deg in random.sample(range(360), min(72, 360)):  # Try 72 rotations
            poly = make_tree(x, y, deg)
            if not collides(poly, polys, buf):
                # Evaluate compactness
                test_polys = polys + [poly]
                score = bbox_side_polys(test_polys)
                if score < best_score:
                    best_score = score
                    best_deg = deg

        if best_deg is not None:
            placements.append((x, y, best_deg))
            polys.append(make_tree(x, y, best_deg))

        k += 1
        attempts += 1

    return placements[:n]


def greedy_radial_placement(n: int, cfg: Config, seed: int) -> List[Tuple[float, float, float]]:
    """Enhanced greedy placement with golden section search for optimal radius."""
    random.seed(seed)

    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, random.randint(0, 359))]

    buf = cfg.collision_buffer
    placements = []
    polys = []

    # First tree at origin with best rotation
    best_deg = random.randint(0, 359)
    placements.append((0.0, 0.0, best_deg))
    polys.append(make_tree(0.0, 0.0, best_deg))

    for i in range(1, n):
        best_pos = None
        best_dist = float('inf')

        for _ in range(cfg.placement_attempts):
            angle = random.uniform(0, 2 * math.pi)
            deg = random.randint(0, 359)

            # Golden section search for optimal radius
            r = golden_section_radius_search(angle, deg, polys, buf)

            if r is not None:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                poly = make_tree(x, y, deg)

                if not collides(poly, polys, buf):
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
            r = 0.5 + i * 0.25
            x, y = r * math.cos(angle), r * math.sin(angle)
            deg = random.randint(0, 359)
            placements.append((x, y, deg))
            polys.append(make_tree(x, y, deg))

    return placements


def golden_section_radius_search(angle: float, deg: int, polys: List[Polygon],
                                  buf: float, tol: float = 0.001) -> Optional[float]:
    """
    Golden section search for minimum valid radius.
    From: Kiefer (1953) - optimal univariate search

    Finds minimum r where tree doesn't collide.
    """
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio

    # Find bounds
    r_min, r_max = 0.0, 8.0

    # Binary search to find approximate range
    for _ in range(15):
        r = (r_min + r_max) / 2
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        poly = make_tree(x, y, deg)

        if collides(poly, polys, buf):
            r_min = r
        else:
            r_max = r

    # Golden section refinement
    a, b = r_min, r_max
    c = b - (b - a) / phi
    d = a + (b - a) / phi

    for _ in range(20):
        if abs(b - a) < tol:
            break

        x_c = c * math.cos(angle)
        y_c = c * math.sin(angle)
        x_d = d * math.cos(angle)
        y_d = d * math.sin(angle)

        poly_c = make_tree(x_c, y_c, deg)
        poly_d = make_tree(x_d, y_d, deg)

        collides_c = collides(poly_c, polys, buf)
        collides_d = collides(poly_d, polys, buf)

        if collides_c and not collides_d:
            a = c
        elif not collides_c:
            b = d
        else:
            a = c

        c = b - (b - a) / phi
        d = a + (b - a) / phi

    result = (a + b) / 2
    x = result * math.cos(angle)
    y = result * math.sin(angle)
    poly = make_tree(x, y, deg)

    if not collides(poly, polys, buf):
        return result
    return result + 0.02  # Small offset if needed


def hexagonal_placement(n: int, cfg: Config, seed: int) -> List[Tuple[float, float, float]]:
    """Hexagonal grid placement with rotation optimization."""
    random.seed(seed)

    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, random.randint(0, 359))]

    buf = cfg.collision_buffer
    placements = []
    polys = []

    dx = 0.68  # Tighter spacing
    dy = dx * math.sqrt(3) / 2

    positions = []
    for ring in range(25):
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

        # Find best rotation
        best_deg = None
        best_score = float('inf')

        for deg in range(0, 360, 5):  # Try every 5 degrees
            poly = make_tree(x, y, deg)
            if not collides(poly, polys, buf):
                test_polys = polys + [poly]
                score = bbox_side_polys(test_polys)
                if score < best_score:
                    best_score = score
                    best_deg = deg

        if best_deg is not None:
            placements.append((x, y, best_deg))
            polys.append(make_tree(x, y, best_deg))

    return placements[:n]


def convex_hull_packing(n: int, cfg: Config, seed: int) -> List[Tuple[float, float, float]]:
    """
    Convex hull based packing - place trees on shrinking convex hull.
    Inspired by: "Packing Circles in a Square" - Specht et al.
    """
    random.seed(seed)

    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, random.randint(0, 359))]

    buf = cfg.collision_buffer
    placements = []
    polys = []

    # Start with center
    deg = random.randint(0, 359)
    placements.append((0.0, 0.0, deg))
    polys.append(make_tree(0.0, 0.0, deg))

    # Place remaining on expanding rings
    for i in range(1, n):
        placed = False

        # Current bounding circle radius
        if polys:
            bounds = unary_union(polys).bounds
            current_r = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2
        else:
            current_r = 0.5

        # Try positions on slightly larger circle
        for ring_mult in [1.0, 1.2, 1.5, 2.0]:
            if placed:
                break

            r = current_r * ring_mult + 0.3

            for angle_idx in range(cfg.placement_attempts // 4):
                angle = random.uniform(0, 2 * math.pi)
                x = r * math.cos(angle)
                y = r * math.sin(angle)

                # Find best rotation
                for deg in random.sample(range(360), 36):
                    poly = make_tree(x, y, deg)
                    if not collides(poly, polys, buf):
                        placements.append((x, y, deg))
                        polys.append(poly)
                        placed = True
                        break

                if placed:
                    break

        if not placed:
            # Fallback
            angle = random.uniform(0, 2 * math.pi)
            r = 0.5 + i * 0.3
            x, y = r * math.cos(angle), r * math.sin(angle)
            deg = random.randint(0, 359)
            placements.append((x, y, deg))
            polys.append(make_tree(x, y, deg))

    return placements[:n]


# =============================================================================
# SIMULATED ANNEALING WITH LEVY FLIGHT AND ADAPTIVE REHEATING
# =============================================================================

def simulated_annealing_levy(placements: List[Tuple[float, float, float]],
                              cfg: Config, time_limit: float,
                              seed: int) -> List[Tuple[float, float, float]]:
    """
    Simulated Annealing with Levy Flight moves and adaptive reheating.

    From papers:
    - "Levy flights in evolutionary optimization" - Pavlyukevich (2007)
    - "Self-adaptive simulated annealing" - Ingber (1989)

    Levy flights provide occasional large jumps that help escape local minima.
    Adaptive reheating raises temperature when stuck.
    """
    n = len(placements)
    if n <= 1:
        return placements

    random.seed(seed)
    np.random.seed(seed)

    buf = cfg.collision_buffer
    start = time.time()

    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]

    cur_score = bbox_side_polys(polys)
    best_score = cur_score
    best = list(current)

    T = cfg.sa_temp_initial
    shift = 0.12 * cur_score

    iters = 0
    accepted = 0
    no_improve_count = 0
    last_best_score = best_score

    while time.time() - start < time_limit:
        iters += 1

        i = random.randrange(n)
        x, y, d = current[i]

        move = random.random()

        if move < 0.08:
            # Levy flight move - occasional large jumps
            levy_step = levy_flight(cfg.levy_alpha, shift * 0.5)
            angle = random.uniform(0, 2 * math.pi)
            nx = x + levy_step * math.cos(angle)
            ny = y + levy_step * math.sin(angle)
            nd = d
        elif move < 0.18:
            # Ultra-fine Gaussian
            nx = x + random.gauss(0, shift * 0.02)
            ny = y + random.gauss(0, shift * 0.02)
            nd = d
        elif move < 0.32:
            # Fine Gaussian
            nx = x + random.gauss(0, shift * 0.08)
            ny = y + random.gauss(0, shift * 0.08)
            nd = d
        elif move < 0.48:
            # Medium Gaussian
            nx = x + random.gauss(0, shift * 0.25)
            ny = y + random.gauss(0, shift * 0.25)
            nd = d
        elif move < 0.58:
            # Large uniform
            nx = x + random.uniform(-shift * 1.2, shift * 1.2)
            ny = y + random.uniform(-shift * 1.2, shift * 1.2)
            nd = d
        elif move < 0.68:
            # Fine rotation
            nx, ny = x, y
            nd = (d + random.choice([-1, 1, -2, 2, -3, 3])) % 360
        elif move < 0.78:
            # Large rotation
            nx, ny = x, y
            nd = (d + random.choice([-5, 5, -10, 10, -15, 15, -30, 30, -45, 45, -60, 60, -90, 90])) % 360
        elif move < 0.90:
            # Gradient-estimated center seeking
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx_c, dy_c = cx - x, cy - y
            dist = math.sqrt(dx_c*dx_c + dy_c*dy_c) + 1e-6
            step = random.uniform(0.003, 0.04) * cur_score
            nx = x + step * dx_c / dist
            ny = y + step * dy_c / dist
            nd = d
        else:
            # Smart swap - swap with furthest tree
            if n >= 2:
                # Find tree furthest from center
                cx = sum(p[0] for p in current) / n
                cy = sum(p[1] for p in current) / n

                dists = [(k, (current[k][0]-cx)**2 + (current[k][1]-cy)**2)
                         for k in range(n) if k != i]
                j = max(dists, key=lambda x: x[1])[0]

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
                            no_improve_count = 0
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
                no_improve_count = 0
        else:
            polys[i] = old_poly
            no_improve_count += 1

        # Adaptive temperature with reheating
        progress = (time.time() - start) / time_limit
        T = cfg.sa_temp_initial * ((cfg.sa_temp_final / cfg.sa_temp_initial) ** progress)

        # Reheat if stuck
        if no_improve_count > 1000 and progress < 0.9:
            T = max(T, cfg.sa_temp_initial * 0.3)
            no_improve_count = 0
            shift = 0.15 * cur_score  # Reset shift

        # Adaptive shift
        if iters % 1500 == 0:
            rate = accepted / iters
            if rate > 0.40:
                shift = min(cur_score * 0.3, shift * 1.1)
            elif rate < 0.08:
                shift = max(0.0002, shift * 0.88)

    return best


# =============================================================================
# BASIN HOPPING - Global optimization with local minimization
# =============================================================================

def basin_hopping(placements: List[Tuple[float, float, float]],
                  cfg: Config, time_limit: float,
                  seed: int) -> List[Tuple[float, float, float]]:
    """
    Basin Hopping algorithm for global optimization.
    From: Wales & Doye (1997) - "Global Optimization by Basin-Hopping"

    Combines random perturbation with local minimization.
    """
    n = len(placements)
    if n <= 1:
        return placements

    random.seed(seed)
    buf = cfg.collision_buffer
    start = time.time()

    current = list(placements)
    cur_score = bbox_side(current)

    best = list(current)
    best_score = cur_score

    hop_step = cfg.basin_hop_step * cur_score

    for iteration in range(cfg.basin_hop_iterations):
        if time.time() - start >= time_limit:
            break

        # Random perturbation (hop)
        candidate = list(current)
        polys = [make_tree(x, y, d) for x, y, d in candidate]

        # Perturb random subset of trees
        num_perturb = max(1, n // 4)
        indices = random.sample(range(n), num_perturb)

        for i in indices:
            x, y, d = candidate[i]

            # Random hop
            for _ in range(20):
                nx = x + random.gauss(0, hop_step)
                ny = y + random.gauss(0, hop_step)
                nd = (d + random.randint(-30, 30)) % 360

                new_poly = make_tree(nx, ny, nd)
                others = polys[:i] + polys[i+1:]

                if not collides(new_poly, others, buf):
                    candidate[i] = (nx, ny, nd)
                    polys[i] = new_poly
                    break

        # Local minimization
        remaining_time = (time_limit - (time.time() - start)) / (cfg.basin_hop_iterations - iteration)
        candidate = local_search_gradient(candidate, cfg, min(remaining_time * 0.7, 5.0))

        if not check_overlaps(candidate, buf):
            score = bbox_side(candidate)

            # Accept or reject
            if score < cur_score or random.random() < math.exp(-(score - cur_score) / 0.1):
                current = candidate
                cur_score = score

                if score < best_score:
                    best_score = score
                    best = list(current)

        # Adaptive step size
        hop_step = cfg.basin_hop_step * best_score * (1 + 0.5 * random.random())

    return best


# =============================================================================
# LOCAL SEARCH WITH GRADIENT ESTIMATION
# =============================================================================

def local_search_gradient(placements: List[Tuple[float, float, float]],
                          cfg: Config, time_limit: float = 30.0) -> List[Tuple[float, float, float]]:
    """
    Gradient-estimated local search using finite differences.
    """
    n = len(placements)
    if n <= 1:
        return placements

    buf = cfg.collision_buffer
    prec = cfg.local_precision

    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]
    cur_score = bbox_side_polys(polys)

    start = time.time()
    improved = True
    iters = 0

    while improved and iters < cfg.local_iterations and time.time() - start < time_limit:
        improved = False
        iters += 1

        for i in random.sample(range(n), n):
            x, y, d = current[i]
            best_move = None
            best_improvement = 0

            # Multi-scale gradient search
            for scale in [0.15, 0.3, 0.6, 1.0, 1.8, 3.0]:
                p = prec * scale

                # Compute numerical gradient
                moves = [
                    (-p, 0), (p, 0), (0, -p), (0, p),
                    (-p, -p), (-p, p), (p, -p), (p, p),
                    (-p*1.5, -p*0.5), (p*1.5, p*0.5),  # Diagonal
                ]

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

            # Rotation search
            for dd in [-1, 1, -2, 2, -3, 3, -5, 5, -8, 8, -12, 12]:
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

            if best_move and best_improvement > 1e-10:
                current[i] = (best_move[0], best_move[1], best_move[2])
                polys[i] = best_move[3]
                cur_score -= best_improvement
                improved = True

    return current


# =============================================================================
# NELDER-MEAD SIMPLEX OPTIMIZATION
# =============================================================================

def nelder_mead_optimize(placements: List[Tuple[float, float, float]],
                         cfg: Config, iterations: int = 100) -> List[Tuple[float, float, float]]:
    """
    Nelder-Mead simplex optimization for fine-tuning.
    From: Nelder & Mead (1965) - "A Simplex Method for Function Minimization"

    Optimizes positions of all trees simultaneously.
    """
    n = len(placements)
    if n <= 1:
        return placements

    buf = cfg.collision_buffer

    # Flatten to array
    x = np.array([[p[0], p[1], p[2]] for p in placements])

    def objective(flat_x):
        """Objective function with penalty for collisions."""
        try:
            pos = [(flat_x[i*3], flat_x[i*3+1], flat_x[i*3+2] % 360) for i in range(n)]
            if check_overlaps(pos, buf):
                return 1000.0  # Large penalty
            return bbox_side(pos)
        except:
            return 1000.0

    # Initialize simplex
    dim = 3 * n
    simplex = [x.flatten()]

    for i in range(dim):
        point = x.flatten().copy()
        if i % 3 == 2:  # Rotation
            point[i] += 5
        else:  # Position
            point[i] += 0.01
        simplex.append(point)

    # Simplex parameters
    alpha = 1.0   # Reflection
    gamma = 2.0   # Expansion
    rho = 0.5     # Contraction
    sigma = 0.5   # Shrink

    best_x = simplex[0]
    best_score = objective(best_x)

    for iteration in range(iterations):
        # Sort by objective value
        simplex.sort(key=objective)

        current_best = objective(simplex[0])
        if current_best < best_score:
            best_score = current_best
            best_x = simplex[0].copy()

        # Centroid (excluding worst)
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        reflected = centroid + alpha * (centroid - simplex[-1])
        f_reflected = objective(reflected)

        if objective(simplex[0]) <= f_reflected < objective(simplex[-2]):
            simplex[-1] = reflected
        elif f_reflected < objective(simplex[0]):
            # Expansion
            expanded = centroid + gamma * (reflected - centroid)
            if objective(expanded) < f_reflected:
                simplex[-1] = expanded
            else:
                simplex[-1] = reflected
        else:
            # Contraction
            contracted = centroid + rho * (simplex[-1] - centroid)
            if objective(contracted) < objective(simplex[-1]):
                simplex[-1] = contracted
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])

    # Convert back
    result = [(best_x[i*3], best_x[i*3+1], best_x[i*3+2] % 360) for i in range(n)]

    if not check_overlaps(result, buf):
        return result
    return placements  # Return original if optimization failed


# =============================================================================
# FORCE-DIRECTED COMPACTION (Physics-based)
# =============================================================================

def force_directed_compact(placements: List[Tuple[float, float, float]],
                           cfg: Config) -> List[Tuple[float, float, float]]:
    """
    Force-directed layout compaction using physics simulation.
    From: Fruchterman & Reingold (1991) - force-directed graph drawing

    - Gravity pulls all trees toward center
    - Repulsion prevents overlaps
    - Damping ensures convergence
    """
    n = len(placements)
    if n <= 1:
        return placements

    buf = cfg.collision_buffer
    current = list(placements)

    damping = 0.85
    gravity = cfg.gravity_strength

    for pass_num in range(cfg.compact_passes):
        polys = [make_tree(x, y, d) for x, y, d in current]

        # Compute centroid
        cx = sum(p[0] for p in current) / n
        cy = sum(p[1] for p in current) / n

        # Compute forces for each tree
        forces = [(0.0, 0.0) for _ in range(n)]

        for i in range(n):
            x, y, d = current[i]
            fx, fy = 0.0, 0.0

            # Gravity toward center
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6
            fx += gravity * dx / dist * dist  # Stronger pull for further trees
            fy += gravity * dy / dist * dist

            # Repulsion from nearby trees
            for j in range(n):
                if i == j:
                    continue
                ox, oy, od = current[j]
                dx = x - ox
                dy = y - oy
                dist = math.sqrt(dx*dx + dy*dy) + 1e-6

                if dist < TREE_RADIUS * 3:
                    repel = cfg.repulsion_strength / (dist * dist)
                    fx += repel * dx / dist
                    fy += repel * dy / dist

            forces[i] = (fx * damping, fy * damping)

        # Apply forces with collision checking
        step_mult = max(0.01, 1.0 - pass_num / cfg.compact_passes)

        # Sort by distance from center (furthest first for compacting)
        indices = sorted(range(n), key=lambda i: -((current[i][0]-cx)**2 + (current[i][1]-cy)**2))

        for i in indices:
            x, y, d = current[i]
            fx, fy = forces[i]

            # Scale force
            force_mag = math.sqrt(fx*fx + fy*fy)
            if force_mag > 0.1:
                fx = fx / force_mag * 0.1
                fy = fy / force_mag * 0.1

            nx = x + fx * step_mult
            ny = y + fy * step_mult

            new_poly = make_tree(nx, ny, d)
            others = polys[:i] + polys[i+1:]

            if not collides(new_poly, others, buf):
                current[i] = (nx, ny, d)
                polys[i] = new_poly

    return current


# =============================================================================
# VARIABLE NEIGHBORHOOD SEARCH (VNS)
# =============================================================================

def variable_neighborhood_search(placements: List[Tuple[float, float, float]],
                                  cfg: Config, time_limit: float) -> List[Tuple[float, float, float]]:
    """
    Variable Neighborhood Search metaheuristic.
    From: Mladenovic & Hansen (1997)

    Systematically changes neighborhoods to escape local minima.
    """
    n = len(placements)
    if n <= 1:
        return placements

    buf = cfg.collision_buffer
    start = time.time()

    current = list(placements)
    cur_score = bbox_side(current)

    best = list(current)
    best_score = cur_score

    neighborhoods = [
        0.001,   # Ultra-fine
        0.005,   # Fine
        0.02,    # Small
        0.05,    # Medium
        0.15,    # Large
    ]

    for vns_iter in range(cfg.vns_iterations):
        if time.time() - start >= time_limit:
            break

        k = 0  # Start with smallest neighborhood

        while k < len(neighborhoods) and time.time() - start < time_limit:
            # Shaking - random perturbation in neighborhood k
            scale = neighborhoods[k] * cur_score
            candidate = list(current)
            polys = [make_tree(x, y, d) for x, y, d in candidate]

            # Perturb all trees slightly
            for i in range(n):
                x, y, d = candidate[i]

                for attempt in range(10):
                    nx = x + random.gauss(0, scale)
                    ny = y + random.gauss(0, scale)
                    nd = (d + random.randint(-int(scale*100), int(scale*100))) % 360

                    new_poly = make_tree(nx, ny, nd)
                    others = polys[:i] + polys[i+1:]

                    if not collides(new_poly, others, buf):
                        candidate[i] = (nx, ny, nd)
                        polys[i] = new_poly
                        break

            # Local search
            candidate = local_search_gradient(candidate, cfg, 2.0)

            if not check_overlaps(candidate, buf):
                score = bbox_side(candidate)

                if score < cur_score:
                    current = candidate
                    cur_score = score
                    k = 0  # Reset to smallest neighborhood

                    if score < best_score:
                        best_score = score
                        best = list(current)
                else:
                    k += 1  # Move to larger neighborhood
            else:
                k += 1

    return best


# =============================================================================
# DIFFERENTIAL EVOLUTION
# =============================================================================

def differential_evolution(n: int, cfg: Config, time_limit: float,
                           seed: int) -> List[Tuple[float, float, float]]:
    """
    Differential Evolution for initial population generation.
    From: Storn & Price (1997)

    Population-based optimization that combines solutions.
    """
    if n <= 1:
        if n == 1:
            return [(0.0, 0.0, 0.0)]
        return []

    random.seed(seed)
    np.random.seed(seed)
    buf = cfg.collision_buffer
    start = time.time()

    # Generate initial population using different strategies
    population = []
    strategies = [fermat_spiral_placement, greedy_radial_placement,
                  hexagonal_placement, convex_hull_packing]

    for i in range(cfg.de_population):
        strategy = strategies[i % len(strategies)]
        placement = strategy(n, cfg, seed + i * 1000)
        if len(placement) == n and not check_overlaps(placement, buf):
            population.append(placement)

    if not population:
        return fermat_spiral_placement(n, cfg, seed)

    # Evaluate fitness
    fitness = [bbox_side(p) for p in population]
    best_idx = np.argmin(fitness)
    best = list(population[best_idx])
    best_score = fitness[best_idx]

    generation = 0
    while time.time() - start < time_limit and generation < 50:
        generation += 1

        for i in range(len(population)):
            # Select 3 random different individuals
            candidates = [j for j in range(len(population)) if j != i]
            if len(candidates) < 3:
                continue

            a, b, c = random.sample(candidates, 3)

            # Mutation and crossover
            trial = []
            polys = []

            for tree_idx in range(n):
                if random.random() < cfg.de_crossover:
                    # Mutant vector
                    xa, ya, da = population[a][tree_idx]
                    xb, yb, db = population[b][tree_idx]
                    xc, yc, dc = population[c][tree_idx]

                    nx = xa + cfg.de_mutation * (xb - xc)
                    ny = ya + cfg.de_mutation * (yb - yc)
                    nd = int(da + cfg.de_mutation * (db - dc)) % 360
                else:
                    nx, ny, nd = population[i][tree_idx]

                # Check collision
                new_poly = make_tree(nx, ny, nd)
                if collides(new_poly, polys, buf):
                    # Use original
                    nx, ny, nd = population[i][tree_idx]
                    new_poly = make_tree(nx, ny, nd)

                trial.append((nx, ny, nd))
                polys.append(new_poly)

            # Selection
            if not check_overlaps(trial, buf):
                trial_score = bbox_side(trial)
                if trial_score < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_score

                    if trial_score < best_score:
                        best_score = trial_score
                        best = list(trial)

    return best


# =============================================================================
# SOLVE SINGLE PUZZLE (ULTIMATE VERSION)
# =============================================================================

def solve_puzzle(n: int, cfg: Config, time_limit: float,
                 seed: int) -> Tuple[int, List[Tuple[float, float, float]], float]:
    """
    Solve a single puzzle using all advanced algorithms.
    """
    if n == 0:
        return n, [], 0.0
    if n == 1:
        return n, [(0.0, 0.0, 0.0)], bbox_side([(0.0, 0.0, 0.0)])

    random.seed(seed)
    np.random.seed(seed)

    # Determine restarts based on n (more for small n)
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

    # Phase 1: Multi-strategy placement with restarts
    strategies = [
        fermat_spiral_placement,
        greedy_radial_placement,
        hexagonal_placement,
        convex_hull_packing,
    ]

    for restart in range(num_restarts):
        if time.time() - start_time >= time_limit * 0.85:
            break

        restart_seed = seed + restart * 1000

        # Rotate through strategies
        strategy = strategies[restart % len(strategies)]
        placements = strategy(n, cfg, restart_seed)

        if len(placements) != n or check_overlaps(placements, cfg.collision_buffer):
            continue

        # Optimization pipeline
        remaining = min(time_per_restart * 0.9, time_limit - (time.time() - start_time))

        if remaining > 0.5:
            # 1. Simulated Annealing with Levy Flight
            placements = simulated_annealing_levy(placements, cfg, remaining * 0.4, restart_seed)

            # 2. Local search
            placements = local_search_gradient(placements, cfg, remaining * 0.15)

            # 3. Force-directed compaction
            placements = force_directed_compact(placements, cfg)

            # 4. Final local search
            placements = local_search_gradient(placements, cfg, remaining * 0.1)

        # Validate and score
        if not check_overlaps(placements, cfg.collision_buffer):
            score = bbox_side(placements)
            if score < best_score:
                best_score = score
                best_placements = center_placements(placements)

    # Phase 2: Basin Hopping on best solution (if time remains)
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2.0 and best_placements is not None:
        refined = basin_hopping(best_placements, cfg, remaining_time * 0.4, seed)
        if not check_overlaps(refined, cfg.collision_buffer):
            refined_score = bbox_side(refined)
            if refined_score < best_score:
                best_score = refined_score
                best_placements = center_placements(refined)

    # Phase 3: VNS refinement (if time remains)
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 1.0 and best_placements is not None:
        refined = variable_neighborhood_search(best_placements, cfg, remaining_time * 0.5)
        if not check_overlaps(refined, cfg.collision_buffer):
            refined_score = bbox_side(refined)
            if refined_score < best_score:
                best_score = refined_score
                best_placements = center_placements(refined)

    # Phase 4: Nelder-Mead fine-tuning (small n only)
    if n <= 15 and best_placements is not None:
        refined = nelder_mead_optimize(best_placements, cfg, cfg.simplex_iterations)
        if not check_overlaps(refined, cfg.collision_buffer):
            refined_score = bbox_side(refined)
            if refined_score < best_score:
                best_score = refined_score
                best_placements = center_placements(refined)

    # Fallback
    if best_placements is None:
        best_placements = fermat_spiral_placement(n, cfg, seed)
        best_placements = center_placements(best_placements)
        best_score = bbox_side(best_placements)

    return n, best_placements, best_score


# =============================================================================
# MAIN SOLVER CLASS
# =============================================================================

class FinalBossSolver:
    """Final Boss Solver - Extreme optimization with advanced algorithms."""

    def __init__(self, config: Optional[Config] = None, verbose: bool = True):
        self.cfg = config or extreme_config()
        self.verbose = verbose
        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.running = True

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n\nReceived stop signal. Saving progress...")
        self.running = False

    def compute_time_allocation(self, total_time: float) -> Dict[int, float]:
        """
        Allocate time inversely proportional to n.
        Small n contributes more to score, so gets more time.

        Score formula: sum(side^2 / n) -> small n has huge weight
        """
        allocations = {}
        total_weight = 0

        for n in range(1, 201):
            # Weight: small n gets much more time
            # n=1 gets 20x weight of n=200
            weight = 20.0 / n
            allocations[n] = weight
            total_weight += weight

        # Normalize to total time (reserve 5% for overhead)
        usable_time = total_time * 0.95
        for n in range(1, 201):
            allocations[n] = (allocations[n] / total_weight) * usable_time
            allocations[n] = max(allocations[n], 5.0)  # Minimum 5 seconds

        return allocations

    def solve_all(self, output_path: str = "submission.csv") -> float:
        """Solve all puzzles in parallel."""
        total_time = self.cfg.total_time_hours * 3600
        start_time = time.time()

        if self.verbose:
            print("=" * 70)
            print("SANTA 2025 - FINAL BOSS SOLVER (EXTREME OPTIMIZED)")
            print("=" * 70)
            print(f"Mode: EXTREME (default)")
            print(f"Cores: {self.cfg.num_cores}")
            print(f"Total time budget: {self.cfg.total_time_hours:.1f} hours")
            print(f"Restarts: small={self.cfg.restarts_small}, med={self.cfg.restarts_medium}, large={self.cfg.restarts_large}")
            print(f"Collision buffer: {self.cfg.collision_buffer}")
            print(f"Algorithms: Levy Flight SA, Basin Hopping, VNS, Force-Directed, Nelder-Mead")
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
                            pbar.set_postfix({'n': n, 'side': f'{score:.4f}', 'total': f'{total_score:.2f}'})
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
    """Print detailed score summary."""
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
        description="Santa 2025 Final Boss Solver (Extreme Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python final_boss_solver.py                          # Extreme mode, 4 cores (default)
  python final_boss_solver.py --output submission.csv
  python final_boss_solver.py --quick                  # Fast test (~1 hour)
  python final_boss_solver.py --standard               # Standard (~4-8 hours)
  python final_boss_solver.py --cores 8                # Use 8 cores

Target: Score < 60 (from baseline 157.08)
"""
    )
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--cores", type=int, default=4, help="Number of CPU cores (default: 4)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (~1 hour)")
    parser.add_argument("--standard", action="store_true", help="Standard mode (~4-8 hours)")
    parser.add_argument("--extreme", action="store_true", help="Extreme mode (default, ~72 hours)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Select config - EXTREME is default
    if args.quick:
        cfg = quick_config()
        print("Using QUICK mode (~1 hour)")
    elif args.standard:
        cfg = standard_config()
        print("Using STANDARD mode (~4-8 hours)")
    else:
        cfg = extreme_config()
        print("Using EXTREME mode (~72 hours) [DEFAULT]")

    cfg.num_cores = args.cores
    cfg.seed = args.seed

    print()
    if not HAS_TQDM:
        print("Note: Install tqdm for progress bars: pip install tqdm")
        print()

    # Solve
    solver = FinalBossSolver(config=cfg, verbose=True)
    solver.solve_all(output_path=args.output)

    # Summary
    print()
    print_summary(solver.solutions)


if __name__ == "__main__":
    main()
