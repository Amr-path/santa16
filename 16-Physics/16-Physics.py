#!/usr/bin/env python3
"""
Santa 2025 - Physics-Based Tree Packing Solver
===============================================

A physics-inspired approach to the Christmas tree packing problem using:
- Force-directed relaxation (repulsive/attractive forces)
- Central gravity pull (gentle radius compression)
- Wave compression (4-cardinal directions)
- Collision resolution via iterative force application
- Multi-core parallel processing (15 cores by default)

Key concepts from physics simulation:
1. Repulsive forces push overlapping trees apart
2. Attractive gravity pulls trees toward center
3. Wave compression squeezes from all directions
4. Damping prevents oscillation

Usage:
    python 16-Physics.py --output submission.csv --cores 15
    python 16-Physics.py --output submission.csv --time-hours 24

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
from shapely.geometry import Polygon, box
from shapely import affinity
from shapely.ops import unary_union
from shapely.strtree import STRtree

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PhysicsConfig:
    """Physics simulation configuration."""
    # Core settings
    num_cores: int = 15
    seed: int = 42

    # Collision buffer
    collision_buffer: float = 0.008

    # Physics parameters
    repulsion_strength: float = 0.15      # How hard overlapping trees push apart
    gravity_strength: float = 0.08        # Pull toward center (20% prob in top solutions)
    damping: float = 0.85                 # Velocity damping to prevent oscillation

    # Wave compression
    wave_passes: int = 8                  # Number of wave compression passes
    wave_step: float = 0.002              # Step size for wave compression

    # Radius compression
    radius_compression_prob: float = 0.20  # Probability of applying radius compression
    radius_compression_strength: float = 0.08  # Strength of pull toward center

    # Force relaxation
    force_iterations: int = 150           # Iterations of force-directed relaxation
    force_step: float = 0.01              # Base step for force movements

    # Simulated Annealing (hybrid approach)
    sa_temp_initial: float = 3.0
    sa_temp_final: float = 1e-8
    sa_iterations: int = 15000

    # Multi-restart
    num_restarts: int = 50

    # Time allocation
    time_per_puzzle_base: float = 30.0    # Base seconds per puzzle


# =============================================================================
# GEOMETRY - Tree Polygon Definition
# =============================================================================

TREE_COORDS = [
    (0.0, 0.8),        # tip
    (0.125, 0.5),      # top tier outer right
    (0.0625, 0.5),     # top tier inner right
    (0.2, 0.25),       # middle tier outer right
    (0.1, 0.25),       # middle tier inner right
    (0.35, 0.0),       # bottom tier outer right
    (0.075, 0.0),      # trunk top right
    (0.075, -0.2),     # trunk bottom right
    (-0.075, -0.2),    # trunk bottom left
    (-0.075, 0.0),     # trunk top left
    (-0.35, 0.0),      # bottom tier outer left
    (-0.1, 0.25),      # middle tier inner left
    (-0.2, 0.25),      # middle tier outer left
    (-0.0625, 0.5),    # top tier inner left
    (-0.125, 0.5),     # top tier outer left
]

BASE_POLYGON = Polygon(TREE_COORDS)
TREE_AREA = BASE_POLYGON.area  # ~0.2456

# Pre-compute rotated polygons for all degrees
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


def make_tree_buffered(x: float, y: float, deg: float, buf: float) -> Polygon:
    """Create tree polygon with collision buffer."""
    poly = make_tree(x, y, deg)
    if buf > 0:
        return poly.buffer(buf, join_style=2)
    return poly


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


def find_overlapping_pairs(polys: List[Polygon], buf: float = 0.0) -> List[Tuple[int, int]]:
    """Find all pairs of overlapping polygons."""
    pairs = []
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            if buf > 0:
                if polys[i].distance(polys[j]) < buf:
                    pairs.append((i, j))
            elif polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                pairs.append((i, j))
    return pairs


# =============================================================================
# BOUNDING BOX COMPUTATION
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
# PHYSICS-BASED ALGORITHMS
# =============================================================================

def compute_centroid(placements: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """Compute centroid of all tree positions."""
    if not placements:
        return (0.0, 0.0)
    cx = sum(p[0] for p in placements) / len(placements)
    cy = sum(p[1] for p in placements) / len(placements)
    return (cx, cy)


def force_directed_relaxation(placements: List[Tuple[float, float, float]],
                               cfg: PhysicsConfig) -> List[Tuple[float, float, float]]:
    """
    Apply force-directed relaxation to resolve overlaps and compact layout.

    Physics model:
    - Repulsive forces push overlapping trees apart
    - Attractive gravity pulls all trees toward center
    - Damping prevents oscillation
    """
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    buf = cfg.collision_buffer

    # Initialize velocities (for momentum-based movement)
    velocities = [(0.0, 0.0) for _ in range(n)]

    for iteration in range(cfg.force_iterations):
        polys = [make_tree(x, y, d) for x, y, d in current]
        forces = [(0.0, 0.0) for _ in range(n)]

        # Compute centroid for gravity
        cx, cy = compute_centroid(current)

        # Apply repulsive forces between overlapping/close pairs
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = polys[i], polys[j]
                dist = pi.distance(pj)

                # Only apply repulsion if trees are close or overlapping
                if dist < buf * 3:
                    # Get centroids
                    ci = pi.centroid
                    cj = pj.centroid
                    dx = cj.x - ci.x
                    dy = cj.y - ci.y
                    d = math.sqrt(dx*dx + dy*dy) + 1e-6

                    # Repulsion strength inversely proportional to distance
                    if dist < buf:
                        # Strong repulsion for overlapping
                        strength = cfg.repulsion_strength * (1 + (buf - dist) / buf)
                    else:
                        # Weaker repulsion for close trees
                        strength = cfg.repulsion_strength * 0.3 * (1 - dist / (buf * 3))

                    # Normalize and apply force
                    fx = -strength * dx / d
                    fy = -strength * dy / d

                    forces[i] = (forces[i][0] + fx, forces[i][1] + fy)
                    forces[j] = (forces[j][0] - fx, forces[j][1] - fy)

        # Apply gravitational attraction toward center
        for i in range(n):
            x, y, d = current[i]
            dx = cx - x
            dy = cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6

            # Apply gravity with probability
            if random.random() < cfg.radius_compression_prob:
                gx = cfg.gravity_strength * dx / dist
                gy = cfg.gravity_strength * dy / dist
                forces[i] = (forces[i][0] + gx, forces[i][1] + gy)

        # Update positions with forces and damping
        new_current = []
        for i in range(n):
            x, y, d = current[i]
            fx, fy = forces[i]
            vx, vy = velocities[i]

            # Update velocity with force and damping
            vx = (vx + fx * cfg.force_step) * cfg.damping
            vy = (vy + fy * cfg.force_step) * cfg.damping
            velocities[i] = (vx, vy)

            # Update position
            nx = x + vx
            ny = y + vy

            # Check if new position is valid
            new_poly = make_tree(nx, ny, d)
            others = polys[:i] + polys[i+1:]

            if not has_collision(new_poly, others, buf):
                new_current.append((nx, ny, d))
            else:
                # Try smaller movement
                for scale in [0.5, 0.25, 0.1]:
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
                     cfg: PhysicsConfig, direction: str) -> List[Tuple[float, float, float]]:
    """
    Apply wave compression from one direction.

    Like gravity pushing from one side, this squeezes all trees
    in the specified direction (up, down, left, right).
    """
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    buf = cfg.collision_buffer
    step = cfg.wave_step

    # Direction vectors
    directions = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (1, 0),
        'right': (-1, 0)
    }
    dx, dy = directions.get(direction, (0, 0))

    # Sort trees by position in wave direction
    if direction == 'up':
        order = sorted(range(n), key=lambda i: current[i][1])
    elif direction == 'down':
        order = sorted(range(n), key=lambda i: -current[i][1])
    elif direction == 'left':
        order = sorted(range(n), key=lambda i: current[i][0])
    else:  # right
        order = sorted(range(n), key=lambda i: -current[i][0])

    # Push each tree in wave direction
    for _ in range(cfg.wave_passes):
        polys = [make_tree(x, y, d) for x, y, d in current]

        for idx in order:
            x, y, d = current[idx]

            # Try progressively smaller steps
            for mult in [4.0, 2.0, 1.0, 0.5, 0.25, 0.1]:
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
                                    cfg: PhysicsConfig) -> List[Tuple[float, float, float]]:
    """
    Apply wave compression from all 4 cardinal directions.

    This mimics the "4-cardinal wave compression" from top Kaggle solutions.
    """
    current = list(placements)

    for direction in ['up', 'down', 'left', 'right']:
        current = wave_compression(current, cfg, direction)

    return current


def gentle_radius_compression(placements: List[Tuple[float, float, float]],
                               cfg: PhysicsConfig) -> List[Tuple[float, float, float]]:
    """
    Gently pull all trees toward the center.

    From Kaggle: "gentle radius compression – pull trees toward center
    (20% prob, 0.08 strength)"
    """
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    buf = cfg.collision_buffer

    # Compute centroid
    cx, cy = compute_centroid(current)

    # Sort by distance from center (furthest first)
    distances = [(i, math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)) for i, p in enumerate(current)]
    distances.sort(key=lambda x: -x[1])

    polys = [make_tree(x, y, d) for x, y, d in current]

    for idx, dist in distances:
        if dist < 0.01:
            continue

        x, y, d = current[idx]
        dx, dy = cx - x, cy - y
        norm = math.sqrt(dx*dx + dy*dy) + 1e-6

        # Try different compression strengths
        for strength in [cfg.radius_compression_strength, 0.04, 0.02, 0.01, 0.005]:
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
# INITIAL PLACEMENT STRATEGIES
# =============================================================================

def hexagonal_grid_placement(n: int, cfg: PhysicsConfig) -> List[Tuple[float, float, float]]:
    """Place trees in a hexagonal grid pattern (dense packing)."""
    placements = []

    # Estimate grid size needed
    area_per_tree = TREE_AREA * 1.8  # With some margin
    total_area = n * area_per_tree
    side = math.sqrt(total_area) * 1.2

    spacing_x = 0.75
    spacing_y = 0.65

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


def spiral_placement(n: int, cfg: PhysicsConfig) -> List[Tuple[float, float, float]]:
    """Place trees in a spiral pattern from center outward."""
    placements = []

    angle = 0
    radius = 0.3
    radius_step = 0.12
    angle_step = 2.4  # Golden angle approximation

    for _ in range(n):
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        deg = random.randint(0, 359)
        placements.append((x, y, deg))

        angle += angle_step
        radius += radius_step / (1 + radius * 0.5)

    return placements


def random_placement(n: int, cfg: PhysicsConfig) -> List[Tuple[float, float, float]]:
    """Random placement within estimated bounds."""
    side = math.sqrt(n * TREE_AREA) * 2.5
    placements = []

    for _ in range(n):
        x = random.uniform(-side/2, side/2)
        y = random.uniform(-side/2, side/2)
        deg = random.randint(0, 359)
        placements.append((x, y, deg))

    return placements


# =============================================================================
# GREEDY PLACEMENT WITH PHYSICS
# =============================================================================

def greedy_place_with_physics(n: int, cfg: PhysicsConfig) -> List[Tuple[float, float, float]]:
    """Greedy placement followed by physics relaxation."""
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    buf = cfg.collision_buffer
    placements = []
    polys = []

    # Place first tree at origin
    placements.append((0.0, 0.0, random.randint(0, 359)))
    polys.append(make_tree(*placements[0]))

    # Greedy placement for remaining trees
    for i in range(1, n):
        best_pos = None
        best_score = float('inf')

        # Try placing near existing trees
        for _ in range(min(50 + n * 2, 200)):
            if placements:
                # Pick a random existing tree and place near it
                ref = random.choice(placements)
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(0.5, 1.5)
                x = ref[0] + dist * math.cos(angle)
                y = ref[1] + dist * math.sin(angle)
            else:
                x, y = 0.0, 0.0

            deg = random.randint(0, 359)
            poly = make_tree(x, y, deg)

            if not has_collision(poly, polys, buf):
                # Compute score (bounding box)
                test_polys = polys + [poly]
                score = bbox_side_polys(test_polys)

                if score < best_score:
                    best_score = score
                    best_pos = (x, y, deg)

        if best_pos is None:
            # Force placement if no valid position found
            angle = random.uniform(0, 2 * math.pi)
            dist = 2.0 + i * 0.1
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            best_pos = (x, y, random.randint(0, 359))

        placements.append(best_pos)
        polys.append(make_tree(*best_pos))

    return placements


# =============================================================================
# SIMULATED ANNEALING WITH PHYSICS MOVES
# =============================================================================

def physics_sa(placements: List[Tuple[float, float, float]],
               cfg: PhysicsConfig, time_limit: float) -> List[Tuple[float, float, float]]:
    """
    Simulated Annealing with physics-inspired moves.

    Move types:
    - Standard SA moves (translation, rotation)
    - Center-seeking moves (gravity)
    - Wave compression moves
    - Force-directed adjustments
    """
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
    shift = 0.1 * cur_score

    iters = 0
    accepted = 0

    while time.time() - start < time_limit:
        iters += 1

        i = random.randrange(n)
        x, y, d = current[i]

        move_type = random.random()

        if move_type < 0.15:
            # Ultra-fine move
            nx = x + random.gauss(0, shift * 0.02)
            ny = y + random.gauss(0, shift * 0.02)
            nd = d
        elif move_type < 0.30:
            # Fine move
            nx = x + random.gauss(0, shift * 0.08)
            ny = y + random.gauss(0, shift * 0.08)
            nd = d
        elif move_type < 0.45:
            # Medium move
            nx = x + random.gauss(0, shift * 0.25)
            ny = y + random.gauss(0, shift * 0.25)
            nd = d
        elif move_type < 0.55:
            # Large escape move
            nx = x + random.uniform(-shift * 1.5, shift * 1.5)
            ny = y + random.uniform(-shift * 1.5, shift * 1.5)
            nd = d
        elif move_type < 0.65:
            # Rotation moves
            nd = (d + random.choice([-1, 1, -2, 2, -5, 5, -10, 10, -15, 15, -30, 30, -45, 45])) % 360
            nx, ny = x, y
        elif move_type < 0.80:
            # Center-seeking move (gravity simulation)
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6
            step = random.uniform(0.005, 0.05) * cur_score
            nx = x + step * dx / dist
            ny = y + step * dy / dist
            nd = d
        elif move_type < 0.90:
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
                    if delta <= 0 or random.random() < math.exp(-delta / T):
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
            # Boundary-seeking move (push toward edge)
            polys_union = unary_union(polys)
            bounds = polys_union.bounds
            directions = [
                (bounds[0] - x, 0),  # left
                (bounds[2] - x, 0),  # right
                (0, bounds[1] - y),  # down
                (0, bounds[3] - y),  # up
            ]
            dx, dy = random.choice(directions)
            step = random.uniform(0.01, 0.05)
            nx = x + step * dx
            ny = y + step * dy
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
        if delta <= 0 or random.random() < math.exp(-delta / T):
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

        # Adaptive step size
        if iters % 2000 == 0:
            rate = accepted / iters
            if rate > 0.35:
                shift = min(cur_score * 0.3, shift * 1.1)
            elif rate < 0.1:
                shift = max(0.0002, shift * 0.9)

    return best


# =============================================================================
# FULL PHYSICS SOLVER PIPELINE
# =============================================================================

def solve_puzzle_physics(n: int, cfg: PhysicsConfig, time_limit: float,
                         seed: int = 42) -> Tuple[List[Tuple[float, float, float]], float]:
    """
    Solve a single puzzle using physics-based approach.

    Pipeline:
    1. Generate initial placements (multiple strategies)
    2. Apply force-directed relaxation
    3. Apply wave compression
    4. Apply gentle radius compression
    5. Run physics-enhanced SA
    6. Final compaction
    """
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

    # Calculate time per restart
    time_per_restart = time_limit / max(cfg.num_restarts, 1)

    while time.time() - start_time < time_limit and restart < cfg.num_restarts:
        restart += 1
        restart_seed = seed + restart * 1000
        random.seed(restart_seed)

        # Choose initial placement strategy
        strategy = restart % 4
        if strategy == 0:
            placements = greedy_place_with_physics(n, cfg)
        elif strategy == 1:
            placements = hexagonal_grid_placement(n, cfg)
        elif strategy == 2:
            placements = spiral_placement(n, cfg)
        else:
            placements = random_placement(n, cfg)

        # Apply physics relaxation
        placements = force_directed_relaxation(placements, cfg)

        # Apply wave compression
        placements = four_cardinal_wave_compression(placements, cfg)

        # Apply gentle radius compression
        placements = gentle_radius_compression(placements, cfg)

        # Run physics-enhanced SA
        remaining_time = time_per_restart * 0.6
        if remaining_time > 0.5:
            placements = physics_sa(placements, cfg, remaining_time)

        # Final wave compression
        placements = four_cardinal_wave_compression(placements, cfg)
        placements = gentle_radius_compression(placements, cfg)

        # Center the result
        placements = center_placements(placements)

        # Validate and score
        polys = [make_tree(x, y, d) for x, y, d in placements]
        has_overlap = False
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                    has_overlap = True
                    break
            if has_overlap:
                break

        if not has_overlap:
            score = bbox_side_polys(polys)
            if score < best_score:
                best_score = score
                best_placements = placements

    if best_placements is None:
        # Fallback to greedy if all attempts failed
        best_placements = greedy_place_with_physics(n, cfg)
        best_placements = center_placements(best_placements)
        best_score = bbox_side(best_placements)

    return best_placements, best_score


def solve_puzzle_worker(args: Tuple[int, PhysicsConfig, float, int]) -> Tuple[int, List[Tuple[float, float, float]], float]:
    """Worker function for parallel puzzle solving."""
    n, cfg, time_limit, seed = args
    placements, score = solve_puzzle_physics(n, cfg, time_limit, seed)
    return n, placements, score


# =============================================================================
# MAIN SOLVER
# =============================================================================

class PhysicsSolver:
    """Main physics-based solver with multi-core support."""

    def __init__(self, config: Optional[PhysicsConfig] = None, verbose: bool = True):
        self.cfg = config or PhysicsConfig()
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
        Allocate time per puzzle based on score contribution.

        Small n puzzles contribute more to total score (score = side²/n),
        so they get more optimization time.
        """
        allocations = {}

        # Priority weights (small n = more time)
        total_weight = 0
        for n in range(1, 201):
            # Inverse n gives more weight to small puzzles
            weight = 10.0 / n
            allocations[n] = weight
            total_weight += weight

        # Normalize to total time
        for n in range(1, 201):
            allocations[n] = (allocations[n] / total_weight) * total_time
            # Minimum time per puzzle
            allocations[n] = max(allocations[n], 5.0)

        return allocations

    def solve_all(self, total_time_hours: float = 24.0, output_path: str = "submission.csv"):
        """Solve all puzzles with physics-based approach."""
        total_time = total_time_hours * 3600
        start_time = time.time()

        if self.verbose:
            print("=" * 70)
            print("PHYSICS-BASED SOLVER")
            print("=" * 70)
            print(f"Cores: {self.cfg.num_cores}")
            print(f"Total time: {total_time_hours:.1f} hours")
            print(f"Physics params:")
            print(f"  - Repulsion strength: {self.cfg.repulsion_strength}")
            print(f"  - Gravity strength: {self.cfg.gravity_strength}")
            print(f"  - Wave passes: {self.cfg.wave_passes}")
            print(f"  - Radius compression: {self.cfg.radius_compression_prob*100:.0f}% prob, {self.cfg.radius_compression_strength} strength")
            print("=" * 70)
            print()

        # Allocate time per puzzle
        time_alloc = self.compute_time_allocation(total_time * 0.9)  # Reserve 10% for overhead

        # Prepare work items
        work_items = []
        for n in range(1, 201):
            work_items.append((n, self.cfg, time_alloc[n], self.cfg.seed + n))

        # Solve puzzles in parallel
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
                            pbar.set_postfix({'n': n, 'score': f'{score:.4f}'})
                        elif completed % 10 == 0:
                            current_score = self.compute_total_score()
                            print(f"  Progress: {completed}/200, Current score: {current_score:.4f}")
                except Exception as e:
                    print(f"Error solving n={futures[future]}: {e}")

        if self.verbose and HAS_TQDM:
            pbar.close()

        # Save results
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
        """Compute total score across all solved puzzles."""
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
        description="Physics-Based Tree Packing Solver for Santa 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run (2 hours, 15 cores)
  python 16-Physics.py --output submission.csv --time-hours 2

  # Standard run (24 hours)
  python 16-Physics.py --output submission.csv --time-hours 24 --cores 15

  # Custom physics parameters
  python 16-Physics.py --output submission.csv --gravity 0.1 --repulsion 0.2
"""
    )
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--cores", type=int, default=15, help="Number of CPU cores (default: 15)")
    parser.add_argument("--time-hours", type=float, default=24.0, help="Total runtime in hours")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gravity", type=float, default=0.08, help="Gravity strength toward center")
    parser.add_argument("--repulsion", type=float, default=0.15, help="Repulsion strength between trees")
    parser.add_argument("--wave-passes", type=int, default=8, help="Wave compression passes")
    parser.add_argument("--restarts", type=int, default=50, help="Number of restarts per puzzle")
    args = parser.parse_args()

    # Configure
    cfg = PhysicsConfig()
    cfg.num_cores = args.cores
    cfg.seed = args.seed
    cfg.gravity_strength = args.gravity
    cfg.repulsion_strength = args.repulsion
    cfg.wave_passes = args.wave_passes
    cfg.num_restarts = args.restarts

    # Solve
    solver = PhysicsSolver(config=cfg, verbose=True)
    solver.solve_all(total_time_hours=args.time_hours, output_path=args.output)


if __name__ == "__main__":
    main()
