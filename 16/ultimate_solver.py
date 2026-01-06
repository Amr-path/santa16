#!/usr/bin/env python3
"""
Santa 2025 - ULTIMATE VPS Solver
=================================

Designed for multi-day VPS runs to achieve score < 65.

Key features:
- Priority optimization for low-n puzzles (highest score impact)
- Progressive refinement - can resume from existing solutions
- Parallel processing using all CPU cores
- Extremely aggressive optimization parameters
- Checkpoint saving every N puzzles
- Multi-seed exploration for diversity

Scoring insight: score = sum(side^2 / n) for n=1..200
- n=1 contributes side^2 (huge impact)
- n=2 contributes side^2 / 2
- n=200 contributes side^2 / 200 (small impact)

Strategy:
- Spend MUCH more time on low-n puzzles
- Use many restarts with different seeds
- Run continuous improvement loops

Usage:
    python ultimate_solver.py [options]

    # Standard long run (estimated 24-48 hours)
    python ultimate_solver.py --output submission.csv

    # Resume from existing solution
    python ultimate_solver.py --resume existing.csv --output improved.csv

    # Quick test mode (2-3 hours)
    python ultimate_solver.py --quick --output test.csv

    # Ultra mode (3-7 days)
    python ultimate_solver.py --ultra --output ultra.csv
"""

import os
import sys
import math
import time
import random
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UltimateConfig:
    """Configuration for ultimate optimization."""

    # Time budgets (in seconds)
    # Priority scaling: more time for low-n puzzles
    base_time_per_puzzle: float = 120.0  # 2 minutes base
    priority_multiplier: float = 10.0     # n=1 gets 10x more time than n=200

    # Total time limits
    max_total_hours: float = 48.0  # Max 48 hours

    # Restarts - VERY aggressive
    base_restarts: int = 500       # Base restarts for n=200
    priority_restarts: int = 5000  # Extra restarts for n=1

    # Placement attempts
    num_placement_attempts: int = 500
    start_radius: float = 20.0
    step_in: float = 0.05
    step_out: float = 0.001

    # Collision buffer
    collision_buffer: float = 0.015

    # Simulated Annealing
    sa_iterations: int = 200000
    sa_temp_initial: float = 3.0
    sa_temp_final: float = 0.000001

    # Local search
    local_precision: float = 0.0005
    local_iterations: int = 5000

    # Compacting
    compact_passes: int = 50
    compact_step: float = 0.0002

    # Basin hopping
    basin_hops: int = 20
    basin_perturbation: float = 0.15

    # Parallel processing
    num_workers: int = 0  # 0 = auto-detect (all cores - 1)

    # Checkpointing
    checkpoint_interval: int = 10  # Save every N puzzles
    checkpoint_path: str = "checkpoint.csv"

    # Random seeds for diversity
    num_seeds: int = 10
    base_seed: int = 42

    def get_time_for_n(self, n: int) -> float:
        """Get time budget for puzzle n (more time for smaller n)."""
        # Priority factor: 1.0 for n=200, up to priority_multiplier for n=1
        priority = 1.0 + (self.priority_multiplier - 1.0) * (200 - n) / 199
        return self.base_time_per_puzzle * priority

    def get_restarts_for_n(self, n: int) -> int:
        """Get number of restarts for puzzle n (more for smaller n)."""
        # More restarts for smaller n
        extra = int((self.priority_restarts - self.base_restarts) * (200 - n) / 199)
        return self.base_restarts + extra


# Preset configurations
def quick_config() -> UltimateConfig:
    """Quick test mode (2-3 hours total)."""
    return UltimateConfig(
        base_time_per_puzzle=10.0,
        priority_multiplier=5.0,
        max_total_hours=3.0,
        base_restarts=50,
        priority_restarts=500,
        num_placement_attempts=100,
        sa_iterations=20000,
        local_iterations=500,
        compact_passes=10,
        basin_hops=5,
        num_seeds=3,
        checkpoint_interval=20
    )


def standard_config() -> UltimateConfig:
    """Standard long run (24-48 hours)."""
    return UltimateConfig()  # Defaults


def ultra_config() -> UltimateConfig:
    """Ultra aggressive mode (3-7 days)."""
    return UltimateConfig(
        base_time_per_puzzle=300.0,  # 5 minutes base
        priority_multiplier=20.0,    # n=1 gets 20x more time
        max_total_hours=168.0,       # Up to 7 days
        base_restarts=1000,
        priority_restarts=10000,
        num_placement_attempts=1000,
        step_in=0.02,
        step_out=0.0005,
        collision_buffer=0.01,
        sa_iterations=500000,
        sa_temp_final=0.0000001,
        local_precision=0.0002,
        local_iterations=10000,
        compact_passes=100,
        compact_step=0.0001,
        basin_hops=50,
        basin_perturbation=0.2,
        num_seeds=20,
        checkpoint_interval=5
    )


# =============================================================================
# GEOMETRY
# =============================================================================

TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

BASE_POLYGON = Polygon(TREE_COORDS)
ROTATED_POLYGONS = {deg: affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))
                    for deg in range(0, 360, 1)}  # Every 1 degree for precision


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


def collides_fast(poly: Polygon, tree_idx: STRtree, all_polys: List[Polygon], buf: float = 0.0) -> bool:
    """Fast collision check using spatial index."""
    query = poly.buffer(buf) if buf > 0 else poly
    for idx in tree_idx.query(query):
        if buf > 0:
            if poly.distance(all_polys[idx]) < buf:
                return True
        elif poly.intersects(all_polys[idx]) and not poly.touches(all_polys[idx]):
            return True
    return False


def bbox_side(placements: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side from placements."""
    if not placements:
        return 0.0
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def bbox_side_polys(polys: List[Polygon]) -> float:
    """Compute bounding square side from polygon list."""
    if not polys:
        return 0.0
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def center(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
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
# PLACEMENT STRATEGIES
# =============================================================================

def weighted_angle() -> float:
    """Generate angle weighted toward corners."""
    while True:
        a = random.uniform(0, 2 * math.pi)
        if random.random() < abs(math.sin(2 * a)):
            return a


def place_radial(base: Polygon, existing: List[Polygon], cfg: UltimateConfig) -> Tuple[float, float]:
    """Place tree using radial greedy approach."""
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


def place_grid(base: Polygon, existing: List[Polygon], cfg: UltimateConfig) -> Tuple[float, float]:
    """Place using fine grid search."""
    if not existing:
        return 0.0, 0.0

    idx = STRtree(existing)
    best = (None, None)
    min_dist = float('inf')

    # Compute current bounds
    b = unary_union(existing).bounds
    extent = max(b[2] - b[0], b[3] - b[1])

    # Search grid
    step = 0.1
    for x in np.arange(-extent, extent, step):
        for y in np.arange(-extent, extent, step):
            cand = affinity.translate(base, xoff=x, yoff=y)
            if not collides_fast(cand, idx, existing, cfg.collision_buffer):
                dist = x*x + y*y
                if dist < min_dist:
                    min_dist = dist
                    best = (x, y)

    return best[0] if best[0] is not None else 0.0, best[1] if best[1] is not None else 0.0


# =============================================================================
# SIMULATED ANNEALING - ULTIMATE
# =============================================================================

def sa_ultimate(placements: List[Tuple[float, float, float]], cfg: UltimateConfig,
                time_limit: float, seed: int = 42) -> List[Tuple[float, float, float]]:
    """Ultimate Simulated Annealing with maximum diversity."""
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
    iters = 0
    accepted = 0

    # Adaptive parameters
    shift = 0.15 * cur_score
    rot = 45.0

    while time.time() - start < time_limit:
        iters += 1

        i = random.randrange(n)
        x, y, d = current[i]

        # Diverse move types
        mt = random.random()

        if mt < 0.20:
            # Very small precise move
            nx = x + random.gauss(0, shift * 0.1)
            ny = y + random.gauss(0, shift * 0.1)
            nd = d
        elif mt < 0.40:
            # Small move
            nx = x + random.gauss(0, shift * 0.3)
            ny = y + random.gauss(0, shift * 0.3)
            nd = d
        elif mt < 0.55:
            # Medium move
            nx = x + random.gauss(0, shift)
            ny = y + random.gauss(0, shift)
            nd = d
        elif mt < 0.65:
            # Large escape
            nx = x + random.uniform(-shift * 3, shift * 3)
            ny = y + random.uniform(-shift * 3, shift * 3)
            nd = d
        elif mt < 0.75:
            # Rotation only (small)
            nx, ny = x, y
            nd = (d + random.choice([-1, 1, -2, 2, -3, 3, -5, 5])) % 360
        elif mt < 0.85:
            # Rotation only (larger)
            nx, ny = x, y
            nd = (d + random.choice([-10, 10, -15, 15, -30, 30, -45, 45, -90, 90])) % 360
        elif mt < 0.92:
            # Center seeking
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 0.001
            step = random.uniform(0.01, 0.1) * cur_score
            nx = x + step * dx / dist
            ny = y + step * dy / dist
            nd = d
        else:
            # Swap with neighbor
            if n >= 2:
                j = random.randrange(n)
                while j == i:
                    j = random.randrange(n)
                ox, oy, od = current[j]

                new_poly_i = make_tree(ox, oy, d)
                new_poly_j = make_tree(x, y, od)

                others_i = [p for k, p in enumerate(polys) if k != i and k != j]

                if not collides(new_poly_i, others_i + [new_poly_j], buf) and \
                   not collides(new_poly_j, others_i + [new_poly_i], buf):
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

                # Update temperature
                progress = (time.time() - start) / time_limit
                T = cfg.sa_temp_initial * ((cfg.sa_temp_final / cfg.sa_temp_initial) ** progress)
                continue

            nx, ny, nd = x, y, d

        # Apply move
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

        # Temperature update
        progress = (time.time() - start) / time_limit
        T = cfg.sa_temp_initial * ((cfg.sa_temp_final / cfg.sa_temp_initial) ** progress)

        # Adaptive step sizes
        if iters % 2000 == 0:
            rate = accepted / iters
            if rate > 0.35:
                shift = min(cur_score * 0.5, shift * 1.1)
                rot = min(90, rot * 1.1)
            elif rate < 0.1:
                shift = max(0.001, shift * 0.9)
                rot = max(1, rot * 0.9)

    return best


# =============================================================================
# LOCAL SEARCH - ULTRA FINE
# =============================================================================

def local_search_ultra(placements: List[Tuple[float, float, float]],
                       cfg: UltimateConfig) -> List[Tuple[float, float, float]]:
    """Ultra fine-grained local search."""
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

            # Try moves in many directions
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

                if not collides(new_poly, others, buf):
                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bbox_side_polys(polys)
                    improvement = cur_score - new_score

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (nx, ny, d, new_poly)

                    polys[i] = old_poly

            # Try rotations
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

            if best_move is not None and best_improvement > 0.000001:
                current[i] = (best_move[0], best_move[1], best_move[2])
                polys[i] = best_move[3]
                cur_score -= best_improvement
                improved = True

    return current


# =============================================================================
# COMPACTING
# =============================================================================

def compact_aggressive(placements: List[Tuple[float, float, float]],
                       cfg: UltimateConfig) -> List[Tuple[float, float, float]]:
    """Aggressive compacting toward center."""
    n = len(placements)
    if n <= 1:
        return placements

    buf = cfg.collision_buffer
    step = cfg.compact_step

    current = list(placements)

    for _ in range(cfg.compact_passes):
        polys = [make_tree(x, y, d) for x, y, d in current]

        # Compute centroid
        cx = sum(p[0] for p in current) / n
        cy = sum(p[1] for p in current) / n

        # Sort by distance (farthest first)
        dists = [(i, math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)) for i, p in enumerate(current)]
        dists.sort(key=lambda x: -x[1])

        for idx, dist in dists:
            if dist < 0.02:
                continue

            x, y, d = current[idx]
            dx, dy = cx - x, cy - y
            norm = math.sqrt(dx*dx + dy*dy) + 0.001

            # Try multiple step sizes
            for mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
                move = step * mult * dist * 10
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
# BASIN HOPPING
# =============================================================================

def basin_hop(placements: List[Tuple[float, float, float]], cfg: UltimateConfig,
              time_limit: float) -> List[Tuple[float, float, float]]:
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

        # Perturb current solution
        perturbed = list(current)
        num_perturb = max(1, int(n * cfg.basin_perturbation))
        indices = random.sample(range(n), min(num_perturb, n))

        for i in indices:
            x, y, d = perturbed[i]
            perturbed[i] = (
                x + random.gauss(0, 0.1 * cur_score),
                y + random.gauss(0, 0.1 * cur_score),
                (d + random.uniform(-30, 30)) % 360
            )

        if check_overlaps(perturbed, buf):
            continue

        # Local minimization
        remaining = min(time_per_hop * 0.5, time_limit - (time.time() - start))
        if remaining > 0.5:
            optimized = sa_ultimate(perturbed, cfg, remaining, seed=hop)
            optimized = local_search_ultra(optimized, cfg)
            optimized = compact_aggressive(optimized, cfg)

            opt_score = bbox_side(optimized)

            if not check_overlaps(optimized, buf) and opt_score < best_score:
                best = optimized
                best_score = opt_score
                current = optimized
                cur_score = opt_score
            elif opt_score < cur_score * 1.05:
                current = optimized
                cur_score = opt_score

    return best


# =============================================================================
# WORKER FUNCTION FOR PARALLEL PROCESSING
# =============================================================================

def optimize_single_restart(args) -> Tuple[List[Tuple[float, float, float]], float]:
    """Worker function for parallel restart optimization."""
    placements, cfg_dict, time_budget, seed = args

    # Reconstruct config
    cfg = UltimateConfig(**cfg_dict)

    random.seed(seed)
    np.random.seed(seed)

    # Run optimization pipeline
    result = sa_ultimate(placements, cfg, time_budget * 0.6, seed)
    result = local_search_ultra(result, cfg)
    result = compact_aggressive(result, cfg)
    result = local_search_ultra(result, cfg)

    score = bbox_side(result)
    valid = not check_overlaps(result, cfg.collision_buffer)

    return (result, score) if valid else (None, float('inf'))


# =============================================================================
# MAIN SOLVER
# =============================================================================

class UltimateSolver:
    """Ultimate VPS solver with priority optimization."""

    def __init__(self, config: Optional[UltimateConfig] = None, verbose: bool = True):
        self.cfg = config or UltimateConfig()
        self.verbose = verbose

        # Auto-detect workers
        if self.cfg.num_workers <= 0:
            self.cfg.num_workers = max(1, mp.cpu_count() - 1)

        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.scores: Dict[int, float] = {}

        self.start_time = None
        self.total_score_history = []

    def solve_single(self, n: int, prev: List[Tuple[float, float, float]],
                     existing_best: Optional[List[Tuple[float, float, float]]] = None) -> Tuple[List[Tuple[float, float, float]], float]:
        """Solve for n trees with ultimate optimization."""
        if n <= 0:
            return [], 0.0

        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            return sol, bbox_side(sol)

        n_start = time.time()
        time_budget = self.cfg.get_time_for_n(n)
        num_restarts = self.cfg.get_restarts_for_n(n)

        # Check total time limit
        if self.start_time:
            elapsed = time.time() - self.start_time
            max_seconds = self.cfg.max_total_hours * 3600
            remaining = max_seconds - elapsed
            if remaining < 60:
                # Running out of time - use minimal optimization
                time_budget = min(time_budget, 10.0)
                num_restarts = min(num_restarts, 10)

        prev_polys = [make_tree(x, y, d) for x, y, d in prev]

        best_sol = None
        best_score = float('inf')

        # If we have an existing solution, start from it
        if existing_best is not None and len(existing_best) == n:
            best_sol = list(existing_best)
            best_score = bbox_side(best_sol)

        angles = list(range(0, 360, 5))  # 72 angles

        # Prepare restart tasks
        restart_tasks = []
        time_per_restart = max(0.5, time_budget / max(1, num_restarts))

        for restart in range(num_restarts):
            seed = self.cfg.base_seed + restart * 1000 + n
            random.seed(seed)

            # Rotation for new tree
            angle = angles[restart % len(angles)]
            base = ROTATED_POLYGONS[angle]

            # Placement
            if restart % 3 == 0:
                bx, by = place_radial(base, prev_polys, self.cfg)
            elif restart % 3 == 1:
                bx, by = place_grid(base, prev_polys, self.cfg)
            else:
                # Combined
                candidates = []
                for fn in [place_radial, place_grid]:
                    px, py = fn(base, prev_polys, self.cfg)
                    candidates.append((px*px + py*py, px, py))
                candidates.sort()
                bx, by = candidates[0][1], candidates[0][2]

            sol = prev + [(bx, by, float(angle))]

            # Perturb from best on later restarts
            if restart > 0 and best_sol is not None and restart % 2 != 0:
                sol = list(best_sol)
                num_p = max(1, int(n * 0.15))
                for idx in random.sample(range(n), min(num_p, n)):
                    x, y, d = sol[idx]
                    sol[idx] = (
                        x + random.gauss(0, 0.03 * best_score),
                        y + random.gauss(0, 0.03 * best_score),
                        (d + random.uniform(-15, 15)) % 360
                    )

            if not check_overlaps(sol, self.cfg.collision_buffer):
                restart_tasks.append((sol, seed))

        # Run optimization - parallel if we have workers
        if self.cfg.num_workers > 1 and len(restart_tasks) >= 4:
            # Prepare config dict for serialization
            cfg_dict = {
                'collision_buffer': self.cfg.collision_buffer,
                'sa_iterations': self.cfg.sa_iterations,
                'sa_temp_initial': self.cfg.sa_temp_initial,
                'sa_temp_final': self.cfg.sa_temp_final,
                'local_precision': self.cfg.local_precision,
                'local_iterations': min(self.cfg.local_iterations, 500),  # Limit for parallel
                'compact_passes': min(self.cfg.compact_passes, 10),
                'compact_step': self.cfg.compact_step,
            }

            # Split tasks into batches
            batch_size = min(self.cfg.num_workers * 2, len(restart_tasks))
            batch = restart_tasks[:batch_size]

            args_list = [(sol, cfg_dict, time_per_restart * 0.5, seed)
                         for sol, seed in batch]

            try:
                with ProcessPoolExecutor(max_workers=self.cfg.num_workers) as executor:
                    futures = [executor.submit(optimize_single_restart, args)
                               for args in args_list]

                    for future in as_completed(futures, timeout=time_budget * 0.7):
                        try:
                            result, score = future.result()
                            if result is not None and score < best_score:
                                best_score = score
                                best_sol = result
                        except Exception:
                            pass
            except Exception:
                pass

        # Sequential fallback or additional restarts
        remaining_time = time_budget - (time.time() - n_start)
        if remaining_time > 1.0:
            for sol, seed in restart_tasks[:min(10, len(restart_tasks))]:
                if time.time() - n_start >= time_budget * 0.9:
                    break

                random.seed(seed)
                remaining = time_budget - (time.time() - n_start)
                per_restart = remaining / 5

                if per_restart > 0.3:
                    optimized = sa_ultimate(sol, self.cfg, per_restart * 0.5, seed)
                    optimized = local_search_ultra(optimized, self.cfg)
                    optimized = compact_aggressive(optimized, self.cfg)

                    score = bbox_side(optimized)
                    if not check_overlaps(optimized, self.cfg.collision_buffer) and score < best_score:
                        best_score = score
                        best_sol = optimized

        # Final basin hopping
        if best_sol is not None:
            remaining = time_budget - (time.time() - n_start)
            if remaining > 2.0:
                best_sol = basin_hop(best_sol, self.cfg, remaining * 0.4)
                best_sol = compact_aggressive(best_sol, self.cfg)
                best_sol = local_search_ultra(best_sol, self.cfg)
                best_score = bbox_side(best_sol)

        if best_sol is None:
            # Fallback
            angle = random.uniform(0, 360)
            base = affinity.rotate(BASE_POLYGON, angle, origin=(0, 0))
            bx, by = place_radial(base, prev_polys, self.cfg)
            best_sol = prev + [(bx, by, angle)]
            best_score = bbox_side(best_sol)

        return center(best_sol), best_score

    def solve_all(self, max_n: int = 200,
                  resume_from: Optional[Dict[int, List[Tuple[float, float, float]]]] = None) -> Dict[int, List[Tuple[float, float, float]]]:
        """Solve for all n from 1 to max_n."""
        self.start_time = time.time()

        if self.verbose:
            print("=" * 70)
            print("SANTA 2025 - ULTIMATE VPS SOLVER")
            print("=" * 70)
            print(f"Workers: {self.cfg.num_workers}")
            print(f"Base time per puzzle: {self.cfg.base_time_per_puzzle:.0f}s")
            print(f"Priority multiplier: {self.cfg.priority_multiplier:.1f}x")
            print(f"Base restarts: {self.cfg.base_restarts}")
            print(f"Priority restarts: {self.cfg.priority_restarts}")
            print(f"Max hours: {self.cfg.max_total_hours:.1f}")
            print(f"Collision buffer: {self.cfg.collision_buffer}")
            print()

            # Estimate total time
            total_est = sum(self.cfg.get_time_for_n(n) for n in range(1, max_n + 1))
            print(f"Estimated total time: {total_est / 3600:.1f} hours")
            print()

        for n in range(1, max_n + 1):
            n_start = time.time()

            prev = self.solutions[n - 1] if n > 1 else []
            existing = resume_from.get(n) if resume_from else None

            sol, score = self.solve_single(n, prev, existing)

            self.solutions[n] = sol
            self.scores[n] = score

            elapsed_n = time.time() - n_start
            total_elapsed = time.time() - self.start_time

            if self.verbose and (n <= 10 or n % 10 == 0 or n == max_n):
                total = self.total_score()
                eta_hours = (total_elapsed / n) * (max_n - n) / 3600
                print(f"n={n:3d}: side={score:.4f}, time={elapsed_n:.1f}s, "
                      f"total={total:.2f}, ETA={eta_hours:.1f}h")

            # Checkpoint
            if n % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint()

        if self.verbose:
            total_time = time.time() - self.start_time
            total = self.total_score()
            print()
            print("=" * 70)
            print(f"COMPLETE - Time: {total_time/3600:.2f} hours")
            print(f"Final Score: {total:.4f}")
            print("=" * 70)

        return self.solutions

    def total_score(self) -> float:
        """Compute competition score."""
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, bbox_side(sol))
            total += (side ** 2) / n
        return total

    def save_checkpoint(self):
        """Save checkpoint to file."""
        try:
            create_submission(self.solutions, self.cfg.checkpoint_path)
        except Exception:
            pass


# =============================================================================
# I/O
# =============================================================================

def create_submission(solutions: Dict[int, List[Tuple[float, float, float]]],
                      output: str = "submission.csv") -> str:
    """Create submission CSV."""
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
    return output


def load_submission(path: str) -> Dict[int, List[Tuple[float, float, float]]]:
    """Load solutions from submission CSV."""
    solutions = {}

    with open(path, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue

            id_part = parts[0]
            n = int(id_part.split("_")[0])

            # Parse values (remove 's' prefix if present)
            x = float(parts[1].lstrip('s'))
            y = float(parts[2].lstrip('s'))
            deg = float(parts[3].lstrip('s'))

            if n not in solutions:
                solutions[n] = []
            solutions[n].append((x, y, deg))

    return solutions


def validate_all(solutions: Dict[int, List[Tuple[float, float, float]]]) -> bool:
    """Validate all solutions."""
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
    """Print detailed score summary."""
    print("=" * 60)
    sides = {n: bbox_side(sol) for n, sol in solutions.items()}
    contribs = {n: (s ** 2) / n for n, s in sides.items()}
    total = sum(contribs.values())

    print(f"Total score: {total:.4f}")
    print(f"Baseline: 157.08")
    print(f"Improvement: {(157.08 - total) / 157.08 * 100:.1f}%")

    # Top contributors (these need most improvement)
    worst = sorted(contribs.items(), key=lambda x: -x[1])[:15]
    print("\nTop 15 worst contributions (focus here):")
    for n, c in worst:
        print(f"  n={n:3d}: side={sides[n]:.4f}, contrib={c:.4f}")

    # Score by ranges
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
        description="Santa 2025 Ultimate VPS Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard long run (24-48 hours)
  python ultimate_solver.py --output submission.csv

  # Resume from existing solution
  python ultimate_solver.py --resume existing.csv --output improved.csv

  # Quick test mode (2-3 hours)
  python ultimate_solver.py --quick --output test.csv

  # Ultra mode (3-7 days)
  python ultimate_solver.py --ultra --output ultra.csv
"""
    )
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--resume", type=str, help="Resume from existing submission CSV")
    parser.add_argument("--max-n", type=int, default=200, help="Maximum n to solve")
    parser.add_argument("--quick", action="store_true", help="Quick mode (2-3 hours)")
    parser.add_argument("--ultra", action="store_true", help="Ultra mode (3-7 days)")
    parser.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0=auto)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    # Select configuration
    if args.quick:
        cfg = quick_config()
        print("Using QUICK mode (2-3 hours)")
    elif args.ultra:
        cfg = ultra_config()
        print("Using ULTRA mode (3-7 days)")
    else:
        cfg = standard_config()
        print("Using STANDARD mode (24-48 hours)")

    cfg.base_seed = args.seed
    if args.workers > 0:
        cfg.num_workers = args.workers

    # Load existing solution if resuming
    resume_from = None
    if args.resume:
        print(f"\nLoading existing solution from: {args.resume}")
        resume_from = load_submission(args.resume)
        if resume_from:
            existing_score = sum((bbox_side(sol) ** 2) / n for n, sol in resume_from.items())
            print(f"Existing score: {existing_score:.4f}")
            print("Will try to improve each puzzle...")

    # Create solver and run
    solver = UltimateSolver(config=cfg, verbose=True)
    solutions = solver.solve_all(max_n=args.max_n, resume_from=resume_from)

    # Validate
    print("\nValidating...")
    if validate_all(solutions):
        print("All solutions valid!")

    # Summary
    print()
    print_summary(solutions)

    # Save
    print(f"\nSaving: {args.output}")
    create_submission(solutions, args.output)
    print(f"Final Score: {solver.total_score():.4f}")


if __name__ == "__main__":
    main()
