#!/usr/bin/env python3
"""
Santa 2025 - Optimized Multi-Core Physics Solver
=================================================

Highly optimized solver running on 15 cores by default.
Combines physics-based approach with aggressive simulated annealing.

Key optimizations:
- 15-core parallel processing (independent puzzle solving)
- Tighter collision buffer (0.005) for denser packing
- More aggressive SA with diverse move types
- Enhanced wave compression (12 passes)
- Priority time allocation for low-n puzzles
- Multiple restarts with basin hopping

Target: Score < 65 (baseline ~157)

Usage:
    python optimized_solver.py --output submission.csv
    python optimized_solver.py --output submission.csv --time-hours 4
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
from shapely.strtree import STRtree

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# CONFIGURATION - OPTIMIZED DEFAULTS
# =============================================================================

@dataclass
class OptimizedConfig:
    """Optimized configuration for maximum performance."""
    # Core settings - ALWAYS 15 CORES
    num_cores: int = 15
    seed: int = 42

    # Tighter collision buffer for denser packing
    collision_buffer: float = 0.005

    # Physics parameters - tuned for better convergence
    repulsion_strength: float = 0.12
    gravity_strength: float = 0.10
    damping: float = 0.82

    # Wave compression - more aggressive
    wave_passes: int = 12
    wave_step: float = 0.0015

    # Radius compression - higher probability
    radius_compression_prob: float = 0.25
    radius_compression_strength: float = 0.10

    # Force relaxation - more iterations
    force_iterations: int = 200
    force_step: float = 0.012

    # Simulated Annealing - more aggressive
    sa_temp_initial: float = 4.0
    sa_temp_final: float = 1e-10
    sa_iterations: int = 30000

    # Multi-restart - more restarts
    num_restarts: int = 80

    # Local search
    local_iterations: int = 2000
    local_precision: float = 0.0003

    # Basin hopping
    basin_hops: int = 15
    basin_perturbation: float = 0.12


# =============================================================================
# GEOMETRY
# =============================================================================

TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

BASE_POLYGON = Polygon(TREE_COORDS)
TREE_AREA = BASE_POLYGON.area

# Pre-compute all rotations
ROTATED_POLYGONS = {deg: affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))
                    for deg in range(360)}


def make_tree(x: float, y: float, deg: float) -> Polygon:
    """Create tree polygon at position with rotation."""
    snapped = int(round(deg)) % 360
    poly = ROTATED_POLYGONS[snapped]
    return affinity.translate(poly, xoff=x, yoff=y) if x != 0 or y != 0 else poly


# =============================================================================
# COLLISION DETECTION
# =============================================================================

def has_collision(poly: Polygon, others: List[Polygon], buf: float = 0.0) -> bool:
    """Check if polygon collides with any other polygon."""
    for o in others:
        if buf > 0:
            if poly.distance(o) < buf:
                return True
        elif poly.intersects(o) and not poly.touches(o):
            return True
    return False


def has_collision_strtree(poly: Polygon, tree: STRtree, polys: List[Polygon],
                          exclude_idx: int, buf: float = 0.0) -> bool:
    """Check collision using spatial index."""
    candidates = tree.query(poly.buffer(buf + 0.01) if buf > 0 else poly)
    for idx in candidates:
        if idx == exclude_idx:
            continue
        other = polys[idx]
        if buf > 0:
            if poly.distance(other) < buf:
                return True
        elif poly.intersects(other) and not poly.touches(other):
            return True
    return False


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
# BOUNDING BOX
# =============================================================================

def bbox_side(placements: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side from placements."""
    if not placements:
        return 0.0
    polys = [make_tree(x, y, d) for x, y, d in placements]
    return bbox_side_polys(polys)


def bbox_side_polys(polys: List[Polygon]) -> float:
    """Compute bounding square side from polygon list."""
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


# =============================================================================
# INITIAL PLACEMENT STRATEGIES
# =============================================================================

def weighted_angle() -> float:
    """Generate angle weighted toward corners."""
    while True:
        a = random.uniform(0, 2 * math.pi)
        if random.random() < abs(math.sin(2 * a)):
            return a


def hexagonal_placement(n: int, cfg: OptimizedConfig) -> List[Tuple[float, float, float]]:
    """Place trees in a hexagonal grid pattern."""
    placements = []
    area_per_tree = TREE_AREA * 1.6
    total_area = n * area_per_tree
    side = math.sqrt(total_area) * 1.1

    spacing_x = 0.72
    spacing_y = 0.62

    placed = 0
    row = 0

    while placed < n:
        y = row * spacing_y - side / 2
        offset = (row % 2) * spacing_x / 2

        cols = int(side / spacing_x) + 2
        for col in range(cols):
            if placed >= n:
                break
            x = col * spacing_x + offset - side / 2
            deg = random.randint(0, 359)
            placements.append((x, y, deg))
            placed += 1
        row += 1

    return placements[:n]


def spiral_placement(n: int, cfg: OptimizedConfig) -> List[Tuple[float, float, float]]:
    """Place trees in a spiral pattern."""
    placements = []
    angle = 0
    radius = 0.25
    radius_step = 0.10
    angle_step = 2.39996  # Golden angle

    for _ in range(n):
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        deg = random.randint(0, 359)
        placements.append((x, y, deg))
        angle += angle_step
        radius += radius_step / (1 + radius * 0.4)

    return placements


def radial_greedy_placement(n: int, cfg: OptimizedConfig) -> List[Tuple[float, float, float]]:
    """Greedy radial placement - most compact."""
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, random.randint(0, 359))]

    placements = [(0.0, 0.0, random.randint(0, 359))]
    polys = [make_tree(*placements[0])]
    buf = cfg.collision_buffer

    for _ in range(1, n):
        best_pos = None
        best_dist = float('inf')

        # Try many angles and find closest valid position
        for _ in range(100 + n):
            angle = weighted_angle()
            deg = random.randint(0, 359)
            base = ROTATED_POLYGONS[deg]

            # Binary search for closest valid position
            r_max = 15.0
            r_min = 0.0

            # First find collision point
            r = r_max
            while r > 0.01:
                x, y = r * math.cos(angle), r * math.sin(angle)
                cand = affinity.translate(base, xoff=x, yoff=y)
                if has_collision(cand, polys, buf):
                    break
                r -= 0.1

            # Then find closest valid
            r_search = r + 0.15
            for step in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                while r_search < r_max:
                    x, y = r_search * math.cos(angle), r_search * math.sin(angle)
                    cand = affinity.translate(base, xoff=x, yoff=y)
                    if not has_collision(cand, polys, buf):
                        break
                    r_search += step

            if r_search < best_dist:
                x, y = r_search * math.cos(angle), r_search * math.sin(angle)
                cand = affinity.translate(base, xoff=x, yoff=y)
                if not has_collision(cand, polys, buf):
                    best_dist = r_search
                    best_pos = (x, y, deg)

        if best_pos is None:
            # Fallback
            angle = random.uniform(0, 2 * math.pi)
            best_pos = (2.0 * math.cos(angle), 2.0 * math.sin(angle), random.randint(0, 359))

        placements.append(best_pos)
        polys.append(make_tree(*best_pos))

    return placements


# =============================================================================
# PHYSICS-BASED ALGORITHMS
# =============================================================================

def force_directed_relaxation(placements: List[Tuple[float, float, float]],
                               cfg: OptimizedConfig) -> List[Tuple[float, float, float]]:
    """Apply force-directed relaxation to resolve overlaps and compact layout."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    buf = cfg.collision_buffer
    velocities = [(0.0, 0.0) for _ in range(n)]

    for iteration in range(cfg.force_iterations):
        polys = [make_tree(x, y, d) for x, y, d in current]
        forces = [(0.0, 0.0) for _ in range(n)]

        # Compute centroid
        cx = sum(p[0] for p in current) / n
        cy = sum(p[1] for p in current) / n

        # Apply repulsive forces
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = polys[i], polys[j]
                dist = pi.distance(pj)

                if dist < buf * 4:
                    ci = pi.centroid
                    cj = pj.centroid
                    dx = cj.x - ci.x
                    dy = cj.y - ci.y
                    d = math.sqrt(dx*dx + dy*dy) + 1e-6

                    if dist < buf:
                        strength = cfg.repulsion_strength * (1.5 + (buf - dist) / buf)
                    else:
                        strength = cfg.repulsion_strength * 0.25 * (1 - dist / (buf * 4))

                    fx = -strength * dx / d
                    fy = -strength * dy / d

                    forces[i] = (forces[i][0] + fx, forces[i][1] + fy)
                    forces[j] = (forces[j][0] - fx, forces[j][1] - fy)

        # Apply gravity
        for i in range(n):
            x, y, d = current[i]
            dx = cx - x
            dy = cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6

            if random.random() < cfg.radius_compression_prob:
                gx = cfg.gravity_strength * dx / dist
                gy = cfg.gravity_strength * dy / dist
                forces[i] = (forces[i][0] + gx, forces[i][1] + gy)

        # Update positions
        new_current = []
        for i in range(n):
            x, y, d = current[i]
            fx, fy = forces[i]
            vx, vy = velocities[i]

            vx = (vx + fx * cfg.force_step) * cfg.damping
            vy = (vy + fy * cfg.force_step) * cfg.damping
            velocities[i] = (vx, vy)

            nx = x + vx
            ny = y + vy

            new_poly = make_tree(nx, ny, d)
            others = polys[:i] + polys[i+1:]

            if not has_collision(new_poly, others, buf):
                new_current.append((nx, ny, d))
            else:
                for scale in [0.5, 0.25, 0.1, 0.05]:
                    test_x = x + vx * scale
                    test_y = y + vy * scale
                    test_poly = make_tree(test_x, test_y, d)
                    if not has_collision(test_poly, others, buf):
                        new_current.append((test_x, test_y, d))
                        break
                else:
                    new_current.append((x, y, d))

        current = new_current

    return current


def wave_compression(placements: List[Tuple[float, float, float]],
                     cfg: OptimizedConfig, direction: str) -> List[Tuple[float, float, float]]:
    """Apply wave compression from one direction."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    buf = cfg.collision_buffer
    step = cfg.wave_step

    directions = {'up': (0, -1), 'down': (0, 1), 'left': (1, 0), 'right': (-1, 0)}
    dx, dy = directions.get(direction, (0, 0))

    if direction == 'up':
        order = sorted(range(n), key=lambda i: current[i][1])
    elif direction == 'down':
        order = sorted(range(n), key=lambda i: -current[i][1])
    elif direction == 'left':
        order = sorted(range(n), key=lambda i: current[i][0])
    else:
        order = sorted(range(n), key=lambda i: -current[i][0])

    for _ in range(cfg.wave_passes):
        polys = [make_tree(x, y, d) for x, y, d in current]

        for idx in order:
            x, y, d = current[idx]

            for mult in [6.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]:
                nx = x + dx * step * mult
                ny = y + dy * step * mult

                new_poly = make_tree(nx, ny, d)
                others = polys[:idx] + polys[idx+1:]

                if not has_collision(new_poly, others, buf):
                    current[idx] = (nx, ny, d)
                    polys[idx] = new_poly
                    break

    return current


def four_cardinal_wave_compression(placements: List[Tuple[float, float, float]],
                                    cfg: OptimizedConfig) -> List[Tuple[float, float, float]]:
    """Apply wave compression from all 4 cardinal directions."""
    current = list(placements)
    for direction in ['up', 'down', 'left', 'right']:
        current = wave_compression(current, cfg, direction)
    return current


def gentle_radius_compression(placements: List[Tuple[float, float, float]],
                               cfg: OptimizedConfig) -> List[Tuple[float, float, float]]:
    """Gently pull all trees toward the center."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    buf = cfg.collision_buffer

    cx = sum(p[0] for p in current) / n
    cy = sum(p[1] for p in current) / n

    distances = [(i, math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)) for i, p in enumerate(current)]
    distances.sort(key=lambda x: -x[1])

    polys = [make_tree(x, y, d) for x, y, d in current]

    for idx, dist in distances:
        if dist < 0.01:
            continue

        x, y, d = current[idx]
        dx, dy = cx - x, cy - y

        for strength in [0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
            nx = x + strength * dx
            ny = y + strength * dy

            new_poly = make_tree(nx, ny, d)
            others = polys[:idx] + polys[idx+1:]

            if not has_collision(new_poly, others, buf):
                current[idx] = (nx, ny, d)
                polys[idx] = new_poly
                break

    return current


# =============================================================================
# SIMULATED ANNEALING - OPTIMIZED
# =============================================================================

def optimized_sa(placements: List[Tuple[float, float, float]],
                  cfg: OptimizedConfig, time_limit: float) -> List[Tuple[float, float, float]]:
    """Optimized Simulated Annealing with diverse move types."""
    n = len(placements)
    if n <= 1:
        return placements

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

    while time.time() - start < time_limit:
        iters += 1

        i = random.randrange(n)
        x, y, d = current[i]

        mt = random.random()

        if mt < 0.12:
            # Ultra-fine move
            nx = x + random.gauss(0, shift * 0.015)
            ny = y + random.gauss(0, shift * 0.015)
            nd = d
        elif mt < 0.25:
            # Fine move
            nx = x + random.gauss(0, shift * 0.06)
            ny = y + random.gauss(0, shift * 0.06)
            nd = d
        elif mt < 0.40:
            # Medium move
            nx = x + random.gauss(0, shift * 0.2)
            ny = y + random.gauss(0, shift * 0.2)
            nd = d
        elif mt < 0.50:
            # Large escape
            nx = x + random.uniform(-shift * 2, shift * 2)
            ny = y + random.uniform(-shift * 2, shift * 2)
            nd = d
        elif mt < 0.58:
            # Fine rotation
            nx, ny = x, y
            nd = (d + random.choice([-1, 1, -2, 2, -3, 3])) % 360
        elif mt < 0.65:
            # Medium rotation
            nx, ny = x, y
            nd = (d + random.choice([-5, 5, -10, 10, -15, 15, -20, 20])) % 360
        elif mt < 0.72:
            # Large rotation
            nx, ny = x, y
            nd = (d + random.choice([-30, 30, -45, 45, -60, 60, -90, 90])) % 360
        elif mt < 0.85:
            # Center-seeking move
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6
            step = random.uniform(0.005, 0.08) * cur_score
            nx = x + step * dx / dist
            ny = y + step * dy / dist
            nd = d
        elif mt < 0.93:
            # Swap positions
            if n >= 2:
                j = random.randrange(n)
                while j == i:
                    j = random.randrange(n)
                ox, oy, od = current[j]

                new_poly_i = make_tree(ox, oy, d)
                new_poly_j = make_tree(x, y, od)
                others = [p for k, p in enumerate(polys) if k != i and k != j]

                if not has_collision(new_poly_i, others + [new_poly_j], buf) and \
                   not has_collision(new_poly_j, others + [new_poly_i], buf):
                    old_polys = list(polys)
                    polys[i] = new_poly_i
                    polys[j] = new_poly_j
                    new_score = bbox_side_polys(polys)

                    delta = new_score - cur_score
                    if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
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
        else:
            # Boundary compression
            polys_union = unary_union(polys)
            bounds = polys_union.bounds
            directions = [
                (bounds[0] - x, 0), (bounds[2] - x, 0),
                (0, bounds[1] - y), (0, bounds[3] - y),
            ]
            ddx, ddy = random.choice(directions)
            step = random.uniform(0.01, 0.06)
            nx = x + step * ddx
            ny = y + step * ddy
            nd = d

        new_poly = make_tree(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if has_collision(new_poly, others, buf):
            progress = (time.time() - start) / time_limit
            T = cfg.sa_temp_initial * ((cfg.sa_temp_final / cfg.sa_temp_initial) ** progress)
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_score = bbox_side_polys(polys)

        delta = new_score - cur_score
        if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
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

        # Adaptive step
        if iters % 1500 == 0:
            rate = accepted / iters
            if rate > 0.35:
                shift = min(cur_score * 0.35, shift * 1.12)
            elif rate < 0.08:
                shift = max(0.0001, shift * 0.88)

    return best


# =============================================================================
# LOCAL SEARCH
# =============================================================================

def local_search(placements: List[Tuple[float, float, float]],
                 cfg: OptimizedConfig) -> List[Tuple[float, float, float]]:
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

        order = list(range(n))
        random.shuffle(order)

        for i in order:
            x, y, d = current[i]
            best_move = None
            best_improvement = 0

            moves = [
                (-prec, 0), (prec, 0), (0, -prec), (0, prec),
                (-prec, -prec), (-prec, prec), (prec, -prec), (prec, prec),
                (-prec*2, 0), (prec*2, 0), (0, -prec*2), (0, prec*2),
                (-prec*0.5, 0), (prec*0.5, 0), (0, -prec*0.5), (0, prec*0.5),
            ]

            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                new_poly = make_tree(nx, ny, d)
                others = polys[:i] + polys[i+1:]

                if not has_collision(new_poly, others, buf):
                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bbox_side_polys(polys)
                    improvement = cur_score - new_score

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (nx, ny, d, new_poly)

                    polys[i] = old_poly

            for dd in [-1, 1, -2, 2, -3, 3, -5, 5]:
                nd = (d + dd) % 360
                new_poly = make_tree(x, y, nd)
                others = polys[:i] + polys[i+1:]

                if not has_collision(new_poly, others, buf):
                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bbox_side_polys(polys)
                    improvement = cur_score - new_score

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (x, y, nd, new_poly)

                    polys[i] = old_poly

            if best_move is not None and best_improvement > 1e-7:
                current[i] = (best_move[0], best_move[1], best_move[2])
                polys[i] = best_move[3]
                cur_score -= best_improvement
                improved = True

    return current


# =============================================================================
# BASIN HOPPING
# =============================================================================

def basin_hopping(placements: List[Tuple[float, float, float]],
                  cfg: OptimizedConfig, time_limit: float) -> List[Tuple[float, float, float]]:
    """Basin hopping for escaping local minima."""
    n = len(placements)
    if n <= 1:
        return placements

    start = time.time()
    buf = cfg.collision_buffer

    best = list(placements)
    best_score = bbox_side(best)

    current = list(placements)
    cur_score = best_score

    time_per_hop = time_limit / cfg.basin_hops

    for hop in range(cfg.basin_hops):
        if time.time() - start >= time_limit * 0.95:
            break

        perturbed = list(current)
        num_perturb = max(1, int(n * cfg.basin_perturbation))
        indices = random.sample(range(n), min(num_perturb, n))

        for i in indices:
            x, y, d = perturbed[i]
            perturbed[i] = (
                x + random.gauss(0, 0.08 * cur_score),
                y + random.gauss(0, 0.08 * cur_score),
                (d + random.uniform(-25, 25)) % 360
            )

        if check_overlaps(perturbed, buf):
            continue

        remaining = min(time_per_hop * 0.6, time_limit - (time.time() - start))
        if remaining > 0.3:
            optimized = optimized_sa(perturbed, cfg, remaining * 0.5)
            optimized = local_search(optimized, cfg)
            optimized = gentle_radius_compression(optimized, cfg)

            opt_score = bbox_side(optimized)

            if not check_overlaps(optimized, buf) and opt_score < best_score:
                best = optimized
                best_score = opt_score
                current = optimized
                cur_score = opt_score
            elif opt_score < cur_score * 1.03:
                current = optimized
                cur_score = opt_score

    return best


# =============================================================================
# FULL OPTIMIZATION PIPELINE
# =============================================================================

def solve_puzzle(n: int, cfg: OptimizedConfig, time_limit: float,
                 seed: int = 42) -> Tuple[List[Tuple[float, float, float]], float]:
    """Solve a single puzzle with full optimization pipeline."""
    random.seed(seed)
    np.random.seed(seed)

    if n == 0:
        return [], 0.0
    if n == 1:
        return [(0.0, 0.0, 0.0)], bbox_side([(0.0, 0.0, 0.0)])

    best_placements = None
    best_score = float('inf')

    start_time = time.time()
    restart = 0
    time_per_restart = time_limit / max(cfg.num_restarts, 1)

    while time.time() - start_time < time_limit and restart < cfg.num_restarts:
        restart += 1
        restart_seed = seed + restart * 1000
        random.seed(restart_seed)
        np.random.seed(restart_seed)

        # Choose strategy
        strategy = restart % 5
        if strategy == 0:
            placements = radial_greedy_placement(n, cfg)
        elif strategy == 1:
            placements = hexagonal_placement(n, cfg)
        elif strategy == 2:
            placements = spiral_placement(n, cfg)
        elif strategy == 3 and best_placements is not None:
            # Perturb best
            placements = list(best_placements)
            num_p = max(1, int(n * 0.15))
            for idx in random.sample(range(n), min(num_p, n)):
                x, y, d = placements[idx]
                placements[idx] = (
                    x + random.gauss(0, 0.04 * best_score),
                    y + random.gauss(0, 0.04 * best_score),
                    (d + random.uniform(-20, 20)) % 360
                )
        else:
            placements = radial_greedy_placement(n, cfg)

        # Physics relaxation
        placements = force_directed_relaxation(placements, cfg)

        # Wave compression
        placements = four_cardinal_wave_compression(placements, cfg)

        # Radius compression
        placements = gentle_radius_compression(placements, cfg)

        # SA optimization
        remaining = time_per_restart * 0.55
        if remaining > 0.3:
            placements = optimized_sa(placements, cfg, remaining)

        # Local search
        placements = local_search(placements, cfg)

        # Final compression
        placements = four_cardinal_wave_compression(placements, cfg)
        placements = gentle_radius_compression(placements, cfg)

        # Center
        placements = center_placements(placements)

        # Validate
        if not check_overlaps(placements, 0.0):  # Strict check
            score = bbox_side(placements)
            if score < best_score:
                best_score = score
                best_placements = placements

    # Basin hopping on best
    if best_placements is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 1.0:
            best_placements = basin_hopping(best_placements, cfg, remaining * 0.5)
            best_placements = local_search(best_placements, cfg)
            best_placements = center_placements(best_placements)
            best_score = bbox_side(best_placements)

    if best_placements is None:
        best_placements = radial_greedy_placement(n, cfg)
        best_placements = center_placements(best_placements)
        best_score = bbox_side(best_placements)

    return best_placements, best_score


def solve_puzzle_worker(args: Tuple[int, dict, float, int]) -> Tuple[int, List[Tuple[float, float, float]], float]:
    """Worker function for parallel puzzle solving."""
    n, cfg_dict, time_limit, seed = args
    cfg = OptimizedConfig(**cfg_dict)
    placements, score = solve_puzzle(n, cfg, time_limit, seed)
    return n, placements, score


# =============================================================================
# MAIN SOLVER
# =============================================================================

class OptimizedSolver:
    """Optimized multi-core solver - always uses 15 cores."""

    def __init__(self, config: Optional[OptimizedConfig] = None, verbose: bool = True):
        self.cfg = config or OptimizedConfig()
        # ALWAYS use 15 cores
        self.cfg.num_cores = 15
        self.verbose = verbose
        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.running = True

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n\nReceived stop signal. Saving progress...")
        self.running = False

    def compute_time_allocation(self, total_time: float) -> Dict[int, float]:
        """Allocate more time to low-n puzzles (higher score impact)."""
        allocations = {}
        total_weight = 0

        for n in range(1, 201):
            # More aggressive priority for small n
            weight = 15.0 / n
            allocations[n] = weight
            total_weight += weight

        for n in range(1, 201):
            allocations[n] = (allocations[n] / total_weight) * total_time
            allocations[n] = max(allocations[n], 8.0)  # Minimum 8 seconds

        return allocations

    def solve_all(self, total_time_hours: float = 24.0, output_path: str = "submission.csv"):
        """Solve all 200 puzzles in parallel using 15 cores."""
        total_time = total_time_hours * 3600
        start_time = time.time()

        if self.verbose:
            print("=" * 70)
            print("OPTIMIZED MULTI-CORE PHYSICS SOLVER")
            print("=" * 70)
            print(f"Cores: {self.cfg.num_cores} (FIXED)")
            print(f"Total time: {total_time_hours:.1f} hours")
            print(f"Collision buffer: {self.cfg.collision_buffer}")
            print(f"Wave passes: {self.cfg.wave_passes}")
            print(f"Restarts per puzzle: {self.cfg.num_restarts}")
            print(f"SA iterations: {self.cfg.sa_iterations}")
            print("=" * 70)
            print()

        time_alloc = self.compute_time_allocation(total_time * 0.92)

        # Prepare config dict for serialization
        cfg_dict = {
            'num_cores': self.cfg.num_cores,
            'seed': self.cfg.seed,
            'collision_buffer': self.cfg.collision_buffer,
            'repulsion_strength': self.cfg.repulsion_strength,
            'gravity_strength': self.cfg.gravity_strength,
            'damping': self.cfg.damping,
            'wave_passes': self.cfg.wave_passes,
            'wave_step': self.cfg.wave_step,
            'radius_compression_prob': self.cfg.radius_compression_prob,
            'radius_compression_strength': self.cfg.radius_compression_strength,
            'force_iterations': self.cfg.force_iterations,
            'force_step': self.cfg.force_step,
            'sa_temp_initial': self.cfg.sa_temp_initial,
            'sa_temp_final': self.cfg.sa_temp_final,
            'sa_iterations': self.cfg.sa_iterations,
            'num_restarts': self.cfg.num_restarts,
            'local_iterations': self.cfg.local_iterations,
            'local_precision': self.cfg.local_precision,
            'basin_hops': self.cfg.basin_hops,
            'basin_perturbation': self.cfg.basin_perturbation,
        }

        work_items = []
        for n in range(1, 201):
            work_items.append((n, cfg_dict, time_alloc[n], self.cfg.seed + n))

        completed = 0

        if self.verbose:
            if HAS_TQDM:
                pbar = tqdm(total=200, desc="Solving",
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            else:
                print("Solving puzzles...")

        with ProcessPoolExecutor(max_workers=self.cfg.num_cores) as executor:
            futures = {executor.submit(solve_puzzle_worker, item): item[0] for item in work_items}

            for future in as_completed(futures):
                if not self.running:
                    break

                try:
                    n, placements, score = future.result()
                    self.solutions[n] = placements
                    completed += 1

                    if self.verbose:
                        if HAS_TQDM:
                            pbar.update(1)
                            pbar.set_postfix({'n': n, 'side': f'{score:.4f}'})
                        elif completed % 10 == 0:
                            current_score = self.compute_total_score()
                            print(f"  Progress: {completed}/200, Current score: {current_score:.4f}")
                except Exception as e:
                    print(f"Error solving n={futures[future]}: {e}")

        if self.verbose and HAS_TQDM:
            pbar.close()

        self.save_submission(output_path)

        total_time_taken = time.time() - start_time
        final_score = self.compute_total_score()

        if self.verbose:
            print()
            print("=" * 70)
            print("COMPLETE")
            print(f"  Puzzles solved: {len(self.solutions)}")
            print(f"  Final score: {final_score:.4f}")
            print(f"  Time taken: {total_time_taken/3600:.2f} hours")
            print(f"  Output: {output_path}")
            print("=" * 70)

        return final_score

    def compute_total_score(self) -> float:
        """Compute total score."""
        total = 0.0
        for n, sol in self.solutions.items():
            side = bbox_side(sol)
            total += (side ** 2) / n
        return total

    def save_submission(self, output_path: str):
        """Save solutions to CSV file."""
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimized Multi-Core Physics Solver for Santa 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run (uses 15 cores by default)
  python optimized_solver.py --output submission.csv --time-hours 4

  # Quick test (1 hour)
  python optimized_solver.py --output test.csv --time-hours 1

  # Long run (24 hours)
  python optimized_solver.py --output best.csv --time-hours 24
"""
    )
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--time-hours", type=float, default=4.0, help="Total runtime in hours (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--restarts", type=int, default=80, help="Restarts per puzzle")
    args = parser.parse_args()

    # Configure - ALWAYS 15 CORES
    cfg = OptimizedConfig()
    cfg.num_cores = 15  # Hardcoded 15 cores
    cfg.seed = args.seed
    cfg.num_restarts = args.restarts

    # Solve
    solver = OptimizedSolver(config=cfg, verbose=True)
    solver.solve_all(total_time_hours=args.time_hours, output_path=args.output)


if __name__ == "__main__":
    main()
