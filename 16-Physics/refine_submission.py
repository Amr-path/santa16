#!/usr/bin/env python3
"""
Santa 2025 - Physics-Based Submission Refinement Tool
======================================================

Refines existing submission CSV files using physics-inspired techniques:
- Force-directed relaxation (repulsive/attractive forces)
- Wave compression (4-cardinal directions)
- Gentle radius compression (pull toward center)
- Physics-enhanced Simulated Annealing

Designed to work with 16-Physics.py output or any valid submission.

Key features:
- Load and improve existing submissions
- Priority-based refinement (worst contributors first)
- Physics-based compaction moves
- Multi-core parallel processing (15 cores by default)
- Graceful shutdown (Ctrl+C saves progress)
- Continuous mode for indefinite improvement

Usage:
    # Basic refinement
    python refine_submission.py --input submission.csv --output refined.csv

    # Focus on high-impact puzzles (n=1 to n=30)
    python refine_submission.py --input submission.csv --focus 1-30 --output refined.csv

    # Continuous mode with physics
    python refine_submission.py --input submission.csv --continuous --output refined.csv

    # Custom physics parameters
    python refine_submission.py --input submission.csv --gravity 0.1 --wave-passes 12

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
class PhysicsRefineConfig:
    """Configuration for physics-based refinement."""
    # Core settings
    num_cores: int = 15
    seed: int = 42

    # Collision buffer
    collision_buffer: float = 0.008

    # Physics parameters
    repulsion_strength: float = 0.12
    gravity_strength: float = 0.08
    damping: float = 0.85

    # Wave compression
    wave_passes: int = 10
    wave_step: float = 0.0015

    # Radius compression
    radius_compression_prob: float = 0.25
    radius_compression_strength: float = 0.06

    # Force relaxation
    force_iterations: int = 80
    force_step: float = 0.008

    # Simulated Annealing
    sa_temp_initial: float = 2.0
    sa_temp_final: float = 1e-9

    # Time per puzzle (seconds)
    time_per_puzzle: float = 60.0

    # Local search
    local_precision: float = 0.0002
    local_iterations: int = 2500

    # Compact passes
    compact_passes: int = 40
    compact_step: float = 0.00012


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

# Pre-compute rotated polygons
ROTATED_POLYGONS = {deg: affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))
                    for deg in range(360)}


def make_tree(x: float, y: float, deg: float) -> Polygon:
    """Create tree polygon at position with rotation."""
    snapped = int(round(deg)) % 360
    poly = ROTATED_POLYGONS[snapped]
    return affinity.translate(poly, xoff=x, yoff=y) if x != 0 or y != 0 else poly


def collides(poly: Polygon, others: List[Polygon], buf: float = 0.0) -> bool:
    """Check if polygon collides with any other polygon."""
    for o in others:
        if buf > 0:
            if poly.distance(o) < buf:
                return True
        elif poly.intersects(o) and not poly.touches(o):
            return True
    return False


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


def compute_centroid(placements: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """Compute centroid of all tree positions."""
    if not placements:
        return (0.0, 0.0)
    cx = sum(p[0] for p in placements) / len(placements)
    cy = sum(p[1] for p in placements) / len(placements)
    return (cx, cy)


# =============================================================================
# PHYSICS-BASED REFINEMENT ALGORITHMS
# =============================================================================

def force_directed_relaxation(placements: List[Tuple[float, float, float]],
                               cfg: PhysicsRefineConfig) -> List[Tuple[float, float, float]]:
    """Apply force-directed relaxation for compaction."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    buf = cfg.collision_buffer
    velocities = [(0.0, 0.0) for _ in range(n)]

    for _ in range(cfg.force_iterations):
        polys = [make_tree(x, y, d) for x, y, d in current]
        forces = [(0.0, 0.0) for _ in range(n)]
        cx, cy = compute_centroid(current)

        # Repulsive forces
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = polys[i], polys[j]
                dist = pi.distance(pj)

                if dist < buf * 2:
                    ci, cj = pi.centroid, pj.centroid
                    dx = cj.x - ci.x
                    dy = cj.y - ci.y
                    d = math.sqrt(dx*dx + dy*dy) + 1e-6

                    strength = cfg.repulsion_strength * (1 + max(0, buf - dist) / buf)
                    fx = -strength * dx / d
                    fy = -strength * dy / d

                    forces[i] = (forces[i][0] + fx, forces[i][1] + fy)
                    forces[j] = (forces[j][0] - fx, forces[j][1] - fy)

        # Gravity toward center
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

            nx, ny = x + vx, y + vy
            new_poly = make_tree(nx, ny, d)
            others = polys[:i] + polys[i+1:]

            if not collides(new_poly, others, buf):
                new_current.append((nx, ny, d))
            else:
                for scale in [0.5, 0.25, 0.1]:
                    tx, ty = x + vx * scale, y + vy * scale
                    tp = make_tree(tx, ty, d)
                    if not collides(tp, others, buf):
                        new_current.append((tx, ty, d))
                        break
                else:
                    new_current.append((x, y, d))

        current = new_current

    return current


def wave_compression(placements: List[Tuple[float, float, float]],
                     cfg: PhysicsRefineConfig, direction: str) -> List[Tuple[float, float, float]]:
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
            for mult in [3.0, 2.0, 1.0, 0.5, 0.25, 0.1]:
                nx = x + dx * step * mult
                ny = y + dy * step * mult
                new_poly = make_tree(nx, ny, d)
                others = polys[:idx] + polys[idx+1:]

                if not collides(new_poly, others, buf):
                    current[idx] = (nx, ny, d)
                    polys[idx] = new_poly
                    break

    return current


def four_cardinal_wave(placements: List[Tuple[float, float, float]],
                       cfg: PhysicsRefineConfig) -> List[Tuple[float, float, float]]:
    """Apply wave compression from all 4 directions."""
    current = list(placements)
    for direction in ['up', 'down', 'left', 'right']:
        current = wave_compression(current, cfg, direction)
    return current


def gentle_radius_compression(placements: List[Tuple[float, float, float]],
                               cfg: PhysicsRefineConfig) -> List[Tuple[float, float, float]]:
    """Gently pull all trees toward center."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    buf = cfg.collision_buffer
    cx, cy = compute_centroid(current)

    distances = [(i, math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)) for i, p in enumerate(current)]
    distances.sort(key=lambda x: -x[1])

    polys = [make_tree(x, y, d) for x, y, d in current]

    for idx, dist in distances:
        if dist < 0.01:
            continue

        x, y, d = current[idx]
        dx, dy = cx - x, cy - y
        norm = math.sqrt(dx*dx + dy*dy) + 1e-6

        for strength in [cfg.radius_compression_strength, 0.04, 0.02, 0.01, 0.005]:
            nx = x + strength * dx
            ny = y + strength * dy
            new_poly = make_tree(nx, ny, d)
            others = polys[:idx] + polys[idx+1:]

            if not collides(new_poly, others, buf):
                current[idx] = (nx, ny, d)
                polys[idx] = new_poly
                break

    return current


def physics_sa_refine(placements: List[Tuple[float, float, float]],
                      cfg: PhysicsRefineConfig, time_limit: float,
                      seed: int = 42) -> List[Tuple[float, float, float]]:
    """Physics-enhanced Simulated Annealing for refinement."""
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
    shift = 0.08 * cur_score
    iters = 0
    accepted = 0

    while time.time() - start < time_limit:
        iters += 1

        i = random.randrange(n)
        x, y, d = current[i]
        move_type = random.random()

        if move_type < 0.12:
            # Ultra-fine
            nx = x + random.gauss(0, shift * 0.02)
            ny = y + random.gauss(0, shift * 0.02)
            nd = d
        elif move_type < 0.28:
            # Fine
            nx = x + random.gauss(0, shift * 0.06)
            ny = y + random.gauss(0, shift * 0.06)
            nd = d
        elif move_type < 0.45:
            # Small
            nx = x + random.gauss(0, shift * 0.15)
            ny = y + random.gauss(0, shift * 0.15)
            nd = d
        elif move_type < 0.55:
            # Medium
            nx = x + random.gauss(0, shift * 0.4)
            ny = y + random.gauss(0, shift * 0.4)
            nd = d
        elif move_type < 0.62:
            # Large
            nx = x + random.uniform(-shift * 1.2, shift * 1.2)
            ny = y + random.uniform(-shift * 1.2, shift * 1.2)
            nd = d
        elif move_type < 0.72:
            # Fine rotation
            nx, ny = x, y
            nd = (d + random.choice([-1, 1, -2, 2, -3, 3])) % 360
        elif move_type < 0.82:
            # Larger rotation
            nx, ny = x, y
            nd = (d + random.choice([-5, 5, -10, 10, -15, 15, -30, 30, -45, 45])) % 360
        elif move_type < 0.92:
            # Center seeking (gravity)
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6
            step = random.uniform(0.003, 0.04) * cur_score
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

        if iters % 2000 == 0:
            rate = accepted / iters
            if rate > 0.35:
                shift = min(cur_score * 0.2, shift * 1.08)
            elif rate < 0.1:
                shift = max(0.0002, shift * 0.92)

    return best


def local_search(placements: List[Tuple[float, float, float]],
                 cfg: PhysicsRefineConfig) -> List[Tuple[float, float, float]]:
    """Ultra-fine local search."""
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

            moves = []
            for scale in [0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0]:
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

            if best_move is not None and best_improvement > 1e-9:
                current[i] = (best_move[0], best_move[1], best_move[2])
                polys[i] = best_move[3]
                cur_score -= best_improvement
                improved = True

    return current


def compact_toward_center(placements: List[Tuple[float, float, float]],
                          cfg: PhysicsRefineConfig) -> List[Tuple[float, float, float]]:
    """Compact trees toward center using fine steps."""
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
            if dist < 0.01:
                continue

            x, y, d = current[idx]
            dx, dy = cx - x, cy - y
            norm = math.sqrt(dx*dx + dy*dy) + 1e-6

            for mult in [1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.007]:
                move = step * mult * dist * 20
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
# FULL REFINEMENT PIPELINE
# =============================================================================

def refine_puzzle_physics(placements: List[Tuple[float, float, float]],
                          cfg: PhysicsRefineConfig, time_limit: float,
                          seed: int = 42) -> Tuple[List[Tuple[float, float, float]], float]:
    """Full physics-based refinement pipeline."""
    best = list(placements)
    best_score = bbox_side(best)

    for s in range(3):
        actual_seed = seed + s * 1000
        random.seed(actual_seed)

        remaining = time_limit * (1 - s * 0.25)
        if remaining < 1:
            break

        # Physics-based refinement
        result = force_directed_relaxation(best, cfg)
        result = four_cardinal_wave(result, cfg)
        result = gentle_radius_compression(result, cfg)

        # SA refinement with physics moves
        result = physics_sa_refine(result, cfg, remaining * 0.45, actual_seed)

        # Local optimization
        result = local_search(result, cfg)
        result = compact_toward_center(result, cfg)
        result = local_search(result, cfg)

        # Final wave compression
        result = four_cardinal_wave(result, cfg)

        score = bbox_side(result)
        if not check_overlaps(result, cfg.collision_buffer) and score < best_score:
            best_score = score
            best = result

    return center_placements(best), best_score


def refine_worker(args: Tuple[int, List[Tuple[float, float, float]], PhysicsRefineConfig, float, int]) -> Tuple[int, List[Tuple[float, float, float]], float, bool]:
    """Worker function for parallel refinement."""
    n, placements, cfg, time_limit, seed = args
    old_score = bbox_side(placements)
    result, new_score = refine_puzzle_physics(placements, cfg, time_limit, seed)
    improved = new_score < old_score
    return n, result, new_score, improved


# =============================================================================
# I/O
# =============================================================================

def load_submission(path: str) -> Dict[int, List[Tuple[float, float, float]]]:
    """Load solutions from submission CSV."""
    solutions = {}

    with open(path, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue

            id_part = parts[0]
            n = int(id_part.split("_")[0])

            x = float(parts[1].lstrip('s'))
            y = float(parts[2].lstrip('s'))
            deg = float(parts[3].lstrip('s'))

            if n not in solutions:
                solutions[n] = []
            solutions[n].append((x, y, deg))

    return solutions


def save_submission(solutions: Dict[int, List[Tuple[float, float, float]]],
                    output: str = "submission.csv"):
    """Save solutions to submission CSV."""
    with open(output, "w") as f:
        f.write("id,x,y,deg\n")
        for n in sorted(solutions.keys()):
            pos = solutions[n]
            if not pos:
                continue
            polys = [make_tree(x, y, d) for x, y, d in pos]
            b = unary_union(polys).bounds
            for idx, (x, y, d) in enumerate(pos):
                f.write(f"{n:03d}_{idx},s{x - b[0]:.6f},s{y - b[1]:.6f},s{d:.6f}\n")


def compute_score(solutions: Dict[int, List[Tuple[float, float, float]]]) -> float:
    """Compute total score."""
    total = 0.0
    for n, sol in solutions.items():
        side = bbox_side(sol)
        total += (side ** 2) / n
    return total


def get_contributions(solutions: Dict[int, List[Tuple[float, float, float]]]) -> Dict[int, float]:
    """Get score contribution per puzzle."""
    contribs = {}
    for n, sol in solutions.items():
        side = bbox_side(sol)
        contribs[n] = (side ** 2) / n
    return contribs


# =============================================================================
# PROGRESS BAR
# =============================================================================

class ProgressBar:
    """Simple progress bar."""

    def __init__(self, total: int, desc: str = "", width: int = 40):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0

    def update(self, n: int = 1, extra: str = ""):
        self.current += n

        now = time.time()
        if now - self.last_update < 0.1 and self.current < self.total:
            return
        self.last_update = now

        progress = self.current / self.total
        filled = int(self.width * progress)
        bar = "=" * filled + ">" + " " * (self.width - filled - 1)

        elapsed = now - self.start_time
        if progress > 0:
            eta = elapsed / progress - elapsed
            eta_str = f"ETA: {int(eta//60):02d}:{int(eta%60):02d}"
        else:
            eta_str = "ETA: --:--"

        percent = progress * 100
        line = f"\r{self.desc}: [{bar}] {percent:5.1f}% ({self.current}/{self.total}) {eta_str}"
        if extra:
            line += f" | {extra}"

        sys.stdout.write(line + " " * 10)
        sys.stdout.flush()

        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def close(self):
        if self.current < self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


# =============================================================================
# MAIN REFINER
# =============================================================================

class PhysicsRefiner:
    """Physics-based solution refiner with multi-core support."""

    def __init__(self, solutions: Dict[int, List[Tuple[float, float, float]]],
                 config: Optional[PhysicsRefineConfig] = None, verbose: bool = True):
        self.solutions = {n: list(sol) for n, sol in solutions.items()}
        self.cfg = config or PhysicsRefineConfig()
        self.verbose = verbose
        self.running = True

        self.best_score = compute_score(self.solutions)
        self.improvements = []

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n\nReceived stop signal. Finishing current puzzle and saving...")
        self.running = False

    def refine_priority(self, focus_range: Optional[Tuple[int, int]] = None,
                        iterations: int = 1, output_path: Optional[str] = None):
        """Refine puzzles by priority with physics-based approach."""
        start_time = time.time()
        start_score = compute_score(self.solutions)

        if self.verbose:
            print("=" * 70)
            print("PHYSICS-BASED REFINEMENT")
            print("=" * 70)
            print(f"Starting score: {start_score:.4f}")
            print(f"Cores: {self.cfg.num_cores}")
            print(f"Time per puzzle: {self.cfg.time_per_puzzle:.0f}s")
            print(f"Physics: gravity={self.cfg.gravity_strength}, repulsion={self.cfg.repulsion_strength}")
            print()

        for iteration in range(iterations):
            if not self.running:
                break

            iter_start = time.time()
            iter_improvements = 0

            contribs = get_contributions(self.solutions)

            if focus_range:
                puzzles = [(n, c) for n, c in contribs.items()
                           if focus_range[0] <= n <= focus_range[1]]
            else:
                puzzles = list(contribs.items())

            puzzles.sort(key=lambda x: -x[1])

            if self.verbose:
                print(f"Iteration {iteration + 1}/{iterations}")
                print(f"Puzzles to refine: {len(puzzles)}")
                print()

            # Prepare work items
            work_items = []
            for n, contrib in puzzles:
                priority = 1.0 + (contrib / max(c for _, c in puzzles)) * 2.0
                time_limit = self.cfg.time_per_puzzle * priority
                work_items.append((n, self.solutions[n], self.cfg, time_limit, self.cfg.seed + n))

            # Progress tracking
            if HAS_TQDM:
                pbar = tqdm(total=len(puzzles), desc=f"Iter {iteration+1}",
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
            else:
                pbar = ProgressBar(len(puzzles), desc=f"Iter {iteration+1}")

            # Parallel refinement
            with ProcessPoolExecutor(max_workers=self.cfg.num_cores) as executor:
                futures = {executor.submit(refine_worker, item): item[0] for item in work_items}

                for future in as_completed(futures):
                    if not self.running:
                        break

                    try:
                        n, result, new_score, improved = future.result()
                        self.solutions[n] = result

                        if improved:
                            iter_improvements += 1

                        current = compute_score(self.solutions)

                        if HAS_TQDM:
                            status = "IMP" if improved else "---"
                            pbar.set_postfix({'score': f'{current:.2f}', 'n': n, 'status': status})
                            pbar.update(1)
                        else:
                            status = "IMPROVED" if improved else ""
                            pbar.update(1, extra=f"n={n}, score={current:.2f} {status}")
                    except Exception as e:
                        print(f"Error refining n={futures[future]}: {e}")

            if HAS_TQDM:
                pbar.close()

            if output_path:
                save_submission(self.solutions, output_path)

            iter_time = time.time() - iter_start
            current_score = compute_score(self.solutions)

            if self.verbose:
                print()
                print(f"Iteration {iteration + 1} complete:")
                print(f"  Improvements: {iter_improvements}")
                print(f"  Score: {current_score:.4f}")
                print(f"  Time: {iter_time/60:.1f} min")
                print()

        total_time = time.time() - start_time
        final_score = compute_score(self.solutions)

        if self.verbose:
            print("=" * 70)
            print("REFINEMENT COMPLETE")
            print(f"  Start score: {start_score:.4f}")
            print(f"  Final score: {final_score:.4f}")
            print(f"  Improvement: {start_score - final_score:.4f} ({(start_score - final_score) / start_score * 100:.2f}%)")
            print(f"  Total time: {total_time / 3600:.2f} hours")
            print("=" * 70)

    def refine_continuous(self, output_path: str):
        """Run continuous refinement until stopped."""
        print("Running continuous physics-based refinement. Press Ctrl+C to stop...")
        print()

        iteration = 0
        while self.running:
            iteration += 1
            self.refine_priority(iterations=1, output_path=output_path)

            if self.running:
                score = compute_score(self.solutions)
                print(f"\n--- After iteration {iteration}: score = {score:.4f} ---\n")


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

    worst = sorted(contribs.items(), key=lambda x: -x[1])[:15]
    print("\nTop 15 worst contributions (focus here):")
    for n, c in worst:
        print(f"  n={n:3d}: side={sides[n]:.4f}, contrib={c:.4f}")

    print("\nScore by n ranges:")
    ranges = [(1, 10), (11, 50), (51, 100), (101, 150), (151, 200)]
    for start, end in ranges:
        range_score = sum(contribs.get(n, 0) for n in range(start, end + 1))
        print(f"  n={start:3d}-{end:3d}: {range_score:.4f}")

    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Physics-Based Refinement for Santa 2025 Submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement
  python refine_submission.py --input submission.csv --output refined.csv

  # Focus on puzzles 1-30 (highest impact)
  python refine_submission.py --input submission.csv --focus 1-30 --output refined.csv

  # Run continuously until Ctrl+C
  python refine_submission.py --input submission.csv --continuous --output refined.csv

  # Multiple iterations with custom physics
  python refine_submission.py --input submission.csv --iterations 5 --gravity 0.1
"""
    )
    parser.add_argument("--input", required=True, help="Input submission CSV")
    parser.add_argument("--output", default="refined.csv", help="Output CSV path")
    parser.add_argument("--focus", type=str, help="Focus range, e.g., '1-30'")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--continuous", action="store_true", help="Run until Ctrl+C")
    parser.add_argument("--time", type=float, default=60.0, help="Time per puzzle (seconds)")
    parser.add_argument("--cores", type=int, default=15, help="Number of CPU cores")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gravity", type=float, default=0.08, help="Gravity strength")
    parser.add_argument("--repulsion", type=float, default=0.12, help="Repulsion strength")
    parser.add_argument("--wave-passes", type=int, default=10, help="Wave compression passes")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    solutions = load_submission(args.input)

    if not solutions:
        print("Error: Could not load solutions")
        return

    initial_score = compute_score(solutions)
    print(f"Loaded {len(solutions)} puzzles, score: {initial_score:.4f}")
    print()

    if not HAS_TQDM:
        print("Note: Install tqdm for better progress bars: pip install tqdm")
        print()

    focus_range = None
    if args.focus:
        parts = args.focus.split("-")
        focus_range = (int(parts[0]), int(parts[1]))
        print(f"Focusing on n={focus_range[0]} to n={focus_range[1]}")

    cfg = PhysicsRefineConfig()
    cfg.time_per_puzzle = args.time
    cfg.num_cores = args.cores
    cfg.seed = args.seed
    cfg.gravity_strength = args.gravity
    cfg.repulsion_strength = args.repulsion
    cfg.wave_passes = args.wave_passes

    refiner = PhysicsRefiner(solutions, config=cfg, verbose=True)

    if args.continuous:
        refiner.refine_continuous(args.output)
    else:
        refiner.refine_priority(
            focus_range=focus_range,
            iterations=args.iterations,
            output_path=args.output
        )

    save_submission(refiner.solutions, args.output)
    final_score = compute_score(refiner.solutions)

    print()
    print_summary(refiner.solutions)

    print(f"\nSaved: {args.output}")
    print(f"Final score: {final_score:.4f}")


if __name__ == "__main__":
    main()
