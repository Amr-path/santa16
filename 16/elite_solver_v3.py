#!/usr/bin/env python3
"""
Santa 2025 - ELITE Solver V3 (Ultimate Optimization)
=====================================================

Target: Score < 65

This is the most advanced version combining:
1. CMA-ES with restart strategies
2. Bottom-Left-Fill (BLF) packing algorithm
3. Optimal angle search
4. Multi-scale local optimization
5. Aggressive compaction with boundary refinement
6. Priority-weighted time allocation (10x more time for n=1 vs n=200)

Key insight: Score = sum(side^2 / n), so small n values matter MORE:
- n=1 contributes side^2 (huge!)
- n=200 contributes side^2/200 (200x smaller)

Therefore we spend 10-15x more time optimizing small n values.

Usage:
    python elite_solver_v3.py [--output submission.csv]
    python elite_solver_v3.py --quick    # 2 hours
    python elite_solver_v3.py --ultra    # 48 hours

Requirements:
    pip install numpy shapely pyclipper tqdm cma scipy
"""

import os
import sys
import math
import time
import random
import argparse
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely import affinity
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.prepared import prep

# Optional imports
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("Note: pip install cma for CMA-ES optimization")

try:
    from scipy.optimize import minimize, differential_evolution
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

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
TREE_AREA = BASE_POLYGON.area  # ~0.2825

# Precompute rotations at 1-degree granularity
ROTATED_CACHE = {}
for deg in range(360):
    ROTATED_CACHE[deg] = affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))

# Find optimal angle (minimizes bounding box of single tree)
def _find_optimal_angle():
    best_angle = 0
    best_bbox_area = float('inf')
    for deg in range(0, 180):  # Symmetry
        b = ROTATED_CACHE[deg].bounds
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area < best_bbox_area:
            best_bbox_area = area
            best_angle = deg
    return best_angle

OPTIMAL_ANGLE = _find_optimal_angle()  # ~45 degrees


def make_tree(x: float, y: float, deg: float) -> Polygon:
    """Create tree polygon at given position with rotation."""
    key = int(round(deg)) % 360
    poly = ROTATED_CACHE[key]
    return affinity.translate(poly, xoff=x, yoff=y)


def make_tree_precise(x: float, y: float, deg: float) -> Polygon:
    """Create tree with precise angle (not cached)."""
    poly = affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))
    return affinity.translate(poly, xoff=x, yoff=y)


def tree_bbox_at_angle(deg: int) -> Tuple[float, float]:
    """Get bounding box dimensions at given angle."""
    b = ROTATED_CACHE[deg % 360].bounds
    return (b[2] - b[0], b[3] - b[1])


def bbox_side(placements: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side length."""
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


def check_overlaps(placements: List[Tuple[float, float, float]], buf: float = 0.001) -> bool:
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


def overlap_penalty(placements: List[Tuple[float, float, float]], buf: float = 0.001) -> float:
    """Compute total overlap penalty."""
    polys = [make_tree(x, y, d) for x, y, d in placements]
    penalty = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            dist = polys[i].distance(polys[j])
            if dist < buf:
                penalty += (buf - dist) * 50
            if polys[i].intersects(polys[j]):
                try:
                    inter = polys[i].intersection(polys[j])
                    penalty += inter.area * 500
                except:
                    penalty += 0.5
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
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Solver configuration."""
    max_hours: float = 24.0
    time_per_puzzle_base: float = 60.0
    priority_multiplier: float = 15.0  # Much more time for small n

    # CMA-ES
    cma_population: int = 50
    cma_max_iter: int = 600
    cma_sigma: float = 0.25

    # Restarts
    num_restarts: int = 40

    # Local search
    local_iterations: int = 3000
    local_precision: float = 0.0001

    # Compaction
    compact_passes: int = 80
    compact_step: float = 0.00015

    # Collision
    collision_buffer: float = 0.001

    # Small n special handling
    small_n_threshold: int = 20
    small_n_time_multiplier: float = 3.0

    checkpoint_interval: int = 5
    checkpoint_path: str = "elite_v3_checkpoint.csv"
    seed: int = 42

    def get_time_for_n(self, n: int) -> float:
        """Get time budget for puzzle n."""
        # Priority: 15x more time for n=1 than n=200
        priority = 1.0 + (self.priority_multiplier - 1.0) * (200 - n) / 199
        base_time = self.time_per_puzzle_base * priority

        # Extra time for very small n
        if n <= self.small_n_threshold:
            base_time *= self.small_n_time_multiplier

        return base_time


def quick_config() -> Config:
    return Config(
        max_hours=2.0,
        time_per_puzzle_base=12.0,
        priority_multiplier=8.0,
        cma_population=25,
        cma_max_iter=150,
        num_restarts=15,
        local_iterations=800,
        compact_passes=30,
        small_n_time_multiplier=2.0,
    )


def standard_config() -> Config:
    return Config()


def ultra_config() -> Config:
    return Config(
        max_hours=48.0,
        time_per_puzzle_base=200.0,
        priority_multiplier=20.0,
        cma_population=80,
        cma_max_iter=1500,
        num_restarts=80,
        local_iterations=8000,
        compact_passes=150,
        small_n_time_multiplier=5.0,
    )


# =============================================================================
# PACKING PATTERNS
# =============================================================================

def pattern_grid(n: int, angle: float = None) -> List[Tuple[float, float, float]]:
    """Offset grid pattern."""
    if angle is None:
        angle = OPTIMAL_ANGLE

    spacing = 0.78  # Tight spacing
    placements = []
    cols = int(math.ceil(math.sqrt(n * 1.1)))

    for i in range(n):
        row = i // cols
        col = i % cols
        x = col * spacing + (row % 2) * (spacing / 2)
        y = row * spacing * 0.85
        placements.append((x, y, float(angle)))

    return center(placements)


def pattern_hex(n: int, angle: float = None) -> List[Tuple[float, float, float]]:
    """Hexagonal grid pattern."""
    if angle is None:
        angle = OPTIMAL_ANGLE

    spacing = 0.76
    dx = spacing
    dy = spacing * math.sqrt(3) / 2
    placements = []
    cols = int(math.ceil(math.sqrt(n * 1.2)))

    for i in range(n):
        row = i // cols
        col = i % cols
        x = col * dx + (row % 2) * (dx / 2)
        y = row * dy
        placements.append((x, y, float(angle)))

    return center(placements[:n])


def pattern_spiral(n: int, angle: float = None) -> List[Tuple[float, float, float]]:
    """Golden angle spiral pattern."""
    if angle is None:
        angle = OPTIMAL_ANGLE

    golden_angle = 2.39996323
    placements = []
    radius = 0.0
    radius_inc = 0.22

    for i in range(n):
        theta = i * golden_angle
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        placements.append((x, y, float(angle)))
        radius += radius_inc / (1 + i * 0.008)

    return center(placements)


def pattern_concentric(n: int, angle: float = None) -> List[Tuple[float, float, float]]:
    """Concentric rings pattern."""
    if angle is None:
        angle = OPTIMAL_ANGLE

    placements = []
    if n == 0:
        return []

    placements.append((0.0, 0.0, float(angle)))
    remaining = n - 1
    ring = 1
    ring_spacing = 0.80

    while remaining > 0:
        radius = ring * ring_spacing
        trees_in_ring = min(remaining, max(6, int(2 * math.pi * radius / 0.68)))

        for i in range(trees_in_ring):
            theta = 2 * math.pi * i / trees_in_ring
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            placements.append((x, y, float(angle)))

        remaining -= trees_in_ring
        ring += 1

    return center(placements[:n])


def pattern_interlocked(n: int) -> List[Tuple[float, float, float]]:
    """Interlocked pairs pattern - trees facing each other."""
    placements = []
    spacing_x = 0.85
    spacing_y = 0.85

    pairs_per_row = int(math.ceil(math.sqrt(n / 2)))
    idx = 0

    for row in range(pairs_per_row + 2):
        for col in range(pairs_per_row + 2):
            if idx >= n:
                break

            base_x = col * spacing_x * 2
            base_y = row * spacing_y * 2

            # First tree pointing up-right
            placements.append((base_x, base_y, 45.0))
            idx += 1

            if idx >= n:
                break

            # Second tree pointing down-left, offset
            placements.append((base_x + spacing_x, base_y + spacing_y * 0.5, 225.0))
            idx += 1

        if idx >= n:
            break

    return center(placements[:n])


def pattern_diamond(n: int, angle: float = None) -> List[Tuple[float, float, float]]:
    """Diamond lattice pattern."""
    if angle is None:
        angle = OPTIMAL_ANGLE

    spacing = 0.65
    placements = []
    size = int(math.ceil(math.sqrt(n * 2)))

    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            if len(placements) >= n:
                break
            x = (i + j) * spacing
            y = (i - j) * spacing
            placements.append((x, y, float(angle)))
        if len(placements) >= n:
            break

    placements.sort(key=lambda p: p[0]**2 + p[1]**2)
    return center(placements[:n])


# =============================================================================
# BOTTOM-LEFT-FILL (BLF) PLACEMENT
# =============================================================================

class BLFPlacer:
    """Bottom-Left-Fill placement algorithm."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def place(self, n: int, angles: List[float] = None) -> List[Tuple[float, float, float]]:
        """Place n trees using BLF algorithm."""
        if angles is None:
            angles = [OPTIMAL_ANGLE] * n

        placements = []
        polys = []

        for i in range(n):
            angle = angles[i]
            pos = self.find_blf_position(polys, angle)
            placements.append((pos[0], pos[1], angle))
            polys.append(make_tree(pos[0], pos[1], angle))

        return center(placements)

    def find_blf_position(self, existing: List[Polygon], angle: float) -> Tuple[float, float]:
        """Find bottom-left-fill position for new tree."""
        if not existing:
            return (0.0, 0.0)

        # Get bounds of existing trees
        bounds = unary_union(existing).bounds
        min_x, min_y = bounds[0] - 1.5, bounds[1] - 1.5
        max_x, max_y = bounds[2] + 1.5, bounds[3] + 1.5

        best_pos = None
        best_key = (float('inf'), float('inf'))

        # Scan from bottom-left
        y_step = 0.05
        x_step = 0.05

        for y in np.arange(min_y, max_y, y_step):
            for x in np.arange(min_x, max_x, x_step):
                poly = make_tree(x, y, angle)

                # Check collision
                ok = True
                for p in existing:
                    if poly.distance(p) < self.cfg.collision_buffer:
                        ok = False
                        break

                if ok:
                    # Compute key: (y, x) for bottom-left priority
                    key = (y, x)
                    if key < best_key:
                        best_key = key
                        best_pos = (x, y)

        if best_pos is None:
            # Fallback: place at edge
            best_pos = (max_x, min_y)

        return best_pos


# =============================================================================
# GREEDY NFP PLACEMENT
# =============================================================================

class NFPPlacer:
    """No-Fit Polygon based placement."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.nfp_cache = {}

    def compute_nfp(self, fixed_angle: float, moving_angle: float) -> Optional[Polygon]:
        """Compute No-Fit Polygon using Minkowski sum."""
        if not HAS_PYCLIPPER:
            return None

        key = (int(fixed_angle) % 360, int(moving_angle) % 360)
        if key in self.nfp_cache:
            return self.nfp_cache[key]

        try:
            fixed = ROTATED_CACHE[key[0]]
            moving = ROTATED_CACHE[key[1]]

            # Reflect moving polygon
            moving_neg = affinity.scale(moving, xfact=-1, yfact=-1, origin=(0, 0))

            precision = 10000
            fixed_path = [(int(x * precision), int(y * precision))
                         for x, y in list(fixed.exterior.coords)[:-1]]
            moving_path = [(int(x * precision), int(y * precision))
                          for x, y in list(moving_neg.exterior.coords)[:-1]]

            result = pyclipper.MinkowskiSum(fixed_path, moving_path, True)

            if result:
                coords = [(x / precision, y / precision) for x, y in result[0]]
                if len(coords) >= 3:
                    nfp = Polygon(coords)
                    self.nfp_cache[key] = nfp
                    return nfp
        except:
            pass

        return None

    def place(self, n: int, angles: List[float] = None) -> List[Tuple[float, float, float]]:
        """Place n trees using NFP-guided greedy placement."""
        if angles is None:
            angles = [OPTIMAL_ANGLE] * n

        placements = []
        polys = []

        for i in range(n):
            angle = angles[i]

            if i == 0:
                pos = (0.0, 0.0)
            else:
                pos = self.find_best_position(placements, polys, angle)

            placements.append((pos[0], pos[1], angle))
            polys.append(make_tree(pos[0], pos[1], angle))

        return center(placements)

    def find_best_position(self, placements: List[Tuple[float, float, float]],
                          polys: List[Polygon], angle: float) -> Tuple[float, float]:
        """Find position minimizing bounding box using NFP."""
        candidates = []

        # Sample NFP boundaries
        for idx, (fx, fy, fa) in enumerate(placements):
            nfp = self.compute_nfp(fa, angle)
            if nfp is None:
                continue

            # Translate to fixed position
            nfp_t = affinity.translate(nfp, xoff=fx, yoff=fy)

            # Sample boundary
            try:
                boundary = nfp_t.exterior
                for t in np.linspace(0, 1, 60, endpoint=False):
                    pt = boundary.interpolate(t, normalized=True)
                    # Add small offset to avoid touching
                    offset = self.cfg.collision_buffer * 2
                    candidates.append((pt.x + offset, pt.y + offset))
            except:
                pass

        if not candidates:
            # Fallback to radial search
            return self.radial_search(polys, angle)

        # Find best candidate
        best_pos = None
        best_score = float('inf')

        for x, y in candidates:
            new_poly = make_tree(x, y, angle)

            # Check collision
            ok = True
            for p in polys:
                if new_poly.distance(p) < self.cfg.collision_buffer:
                    ok = False
                    break

            if not ok:
                continue

            # Compute score (bounding box)
            all_polys = polys + [new_poly]
            side = bbox_side_polys(all_polys)

            if side < best_score:
                best_score = side
                best_pos = (x, y)

        if best_pos is None:
            return self.radial_search(polys, angle)

        return best_pos

    def radial_search(self, polys: List[Polygon], angle: float) -> Tuple[float, float]:
        """Fallback radial search for valid position."""
        if not polys:
            return (0.0, 0.0)

        bounds = unary_union(polys).bounds
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        radius = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2 + 0.8

        best_pos = None
        best_score = float('inf')

        for _ in range(200):
            theta = random.uniform(0, 2 * math.pi)
            for r in np.arange(0.1, radius * 2, 0.1):
                x = cx + r * math.cos(theta)
                y = cy + r * math.sin(theta)

                poly = make_tree(x, y, angle)
                ok = True
                for p in polys:
                    if poly.distance(p) < self.cfg.collision_buffer:
                        ok = False
                        break

                if ok:
                    all_polys = polys + [poly]
                    side = bbox_side_polys(all_polys)
                    if side < best_score:
                        best_score = side
                        best_pos = (x, y)
                    break

        if best_pos is None:
            best_pos = (cx + radius * 1.5, cy)

        return best_pos


# =============================================================================
# CMA-ES OPTIMIZER
# =============================================================================

class CMAOptimizer:
    """CMA-ES based optimizer."""

    def __init__(self, n: int, cfg: Config):
        self.n = n
        self.cfg = cfg
        self.best_solution = None
        self.best_score = float('inf')

    def encode(self, placements: List[Tuple[float, float, float]]) -> np.ndarray:
        """Encode placements to vector."""
        vec = []
        for x, y, a in placements:
            vec.extend([x, y, a / 360.0])
        return np.array(vec)

    def decode(self, vec: np.ndarray) -> List[Tuple[float, float, float]]:
        """Decode vector to placements."""
        result = []
        for i in range(0, len(vec), 3):
            x = vec[i]
            y = vec[i + 1]
            a = (vec[i + 2] % 1.0) * 360.0
            result.append((x, y, a))
        return result

    def objective(self, vec: np.ndarray) -> float:
        """Objective: minimize bbox + collision penalty."""
        placements = self.decode(vec)
        penalty = overlap_penalty(placements, self.cfg.collision_buffer)

        if penalty > 0:
            return bbox_side(placements) + penalty * 500

        side = bbox_side(placements)

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

        max_coord = max(1.0, bbox_side(initial) * 0.6)

        opts = {
            'maxiter': self.cfg.cma_max_iter,
            'popsize': self.cfg.cma_population,
            'bounds': [[-max_coord * 3] * (self.n * 2) + [0] * self.n,
                      [max_coord * 3] * (self.n * 2) + [1] * self.n],
            'tolfun': 1e-9,
            'tolx': 1e-9,
            'verb_disp': 0,
            'verb_log': 0,
            'verbose': -9,
        }

        try:
            es = cma.CMAEvolutionStrategy(x0, self.cfg.cma_sigma, opts)

            while not es.stop() and time.time() - start_time < time_limit:
                solutions = es.ask()
                fitness = [self.objective(s) for s in solutions]
                es.tell(solutions, fitness)

            result = self.decode(es.result.xbest)
            if not check_overlaps(result, self.cfg.collision_buffer):
                return result

        except Exception:
            pass

        return self.best_solution if self.best_solution else initial


# =============================================================================
# LOCAL SEARCH
# =============================================================================

def local_search(placements: List[Tuple[float, float, float]], cfg: Config,
                 time_limit: float) -> List[Tuple[float, float, float]]:
    """Multi-scale local search."""
    n = len(placements)
    if n <= 1:
        return placements

    start_time = time.time()
    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]
    cur_score = bbox_side_polys(polys)

    improved = True
    iteration = 0

    while improved and iteration < cfg.local_iterations and time.time() - start_time < time_limit:
        improved = False
        iteration += 1

        order = list(range(n))
        random.shuffle(order)

        for i in order:
            x, y, d = current[i]
            best_move = None
            best_improvement = 0

            # Multi-scale moves
            for scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
                p = cfg.local_precision * scale
                for dx, dy in [(-p, 0), (p, 0), (0, -p), (0, p),
                              (-p, -p), (-p, p), (p, -p), (p, p)]:
                    nx, ny = x + dx, y + dy
                    new_poly = make_tree(nx, ny, d)

                    ok = True
                    for j, other in enumerate(polys):
                        if j != i and new_poly.distance(other) < cfg.collision_buffer:
                            ok = False
                            break

                    if not ok:
                        continue

                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bbox_side_polys(polys)
                    improvement = cur_score - new_score
                    polys[i] = old_poly

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (nx, ny, d)

            # Rotation moves
            for dd in [-1, 1, -2, 2, -3, 3, -5, 5, -10, 10, -15, 15]:
                nd = (d + dd) % 360
                new_poly = make_tree(x, y, nd)

                ok = True
                for j, other in enumerate(polys):
                    if j != i and new_poly.distance(other) < cfg.collision_buffer:
                        ok = False
                        break

                if not ok:
                    continue

                old_poly = polys[i]
                polys[i] = new_poly
                new_score = bbox_side_polys(polys)
                improvement = cur_score - new_score
                polys[i] = old_poly

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = (x, y, nd)

            if best_move and best_improvement > 1e-10:
                current[i] = best_move
                polys[i] = make_tree(*best_move)
                cur_score -= best_improvement
                improved = True

    return current


# =============================================================================
# COMPACTION
# =============================================================================

def compact(placements: List[Tuple[float, float, float]], cfg: Config,
            time_limit: float) -> List[Tuple[float, float, float]]:
    """Aggressive compaction toward center."""
    n = len(placements)
    if n <= 1:
        return placements

    start_time = time.time()
    current = list(placements)

    for pass_num in range(cfg.compact_passes):
        if time.time() - start_time >= time_limit:
            break

        polys = [make_tree(x, y, d) for x, y, d in current]

        cx = sum(p[0] for p in current) / n
        cy = sum(p[1] for p in current) / n

        # Sort by distance (farthest first)
        indexed = [(i, math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)) for i, p in enumerate(current)]
        indexed.sort(key=lambda x: -x[1])

        for idx, dist in indexed:
            if dist < 0.005:
                continue

            x, y, d = current[idx]
            dx, dy = cx - x, cy - y
            norm = math.sqrt(dx*dx + dy*dy) + 1e-10

            for mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]:
                move = cfg.compact_step * mult * dist * 15
                nx = x + move * dx / norm
                ny = y + move * dy / norm

                new_poly = make_tree(nx, ny, d)
                ok = True
                for j, other in enumerate(polys):
                    if j != idx and new_poly.distance(other) < cfg.collision_buffer:
                        ok = False
                        break

                if ok:
                    current[idx] = (nx, ny, d)
                    polys[idx] = new_poly
                    break

    return current


# =============================================================================
# SIMULATED ANNEALING
# =============================================================================

def simulated_annealing(placements: List[Tuple[float, float, float]], cfg: Config,
                        time_limit: float, seed: int = 42) -> List[Tuple[float, float, float]]:
    """Simulated annealing with adaptive moves."""
    n = len(placements)
    if n <= 1:
        return placements

    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]
    cur_score = bbox_side_polys(polys)

    best = list(current)
    best_score = cur_score

    T = 3.0
    shift = 0.12 * cur_score

    while time.time() - start_time < time_limit:
        i = random.randrange(n)
        x, y, d = current[i]

        mt = random.random()

        if mt < 0.25:
            # Very small move
            nx = x + random.gauss(0, shift * 0.1)
            ny = y + random.gauss(0, shift * 0.1)
            nd = d
        elif mt < 0.45:
            # Small move
            nx = x + random.gauss(0, shift * 0.3)
            ny = y + random.gauss(0, shift * 0.3)
            nd = d
        elif mt < 0.60:
            # Medium move
            nx = x + random.gauss(0, shift * 0.6)
            ny = y + random.gauss(0, shift * 0.6)
            nd = d
        elif mt < 0.75:
            # Rotation
            nx, ny = x, y
            nd = (d + random.choice([-1, 1, -3, 3, -5, 5, -10, 10, -15, 15, -30, 30])) % 360
        elif mt < 0.88:
            # Center-seeking
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-10
            step = random.uniform(0.01, 0.1) * cur_score
            nx = x + step * dx / dist
            ny = y + step * dy / dist
            nd = d
        else:
            # Large escape
            nx = x + random.uniform(-shift * 2.5, shift * 2.5)
            ny = y + random.uniform(-shift * 2.5, shift * 2.5)
            nd = d

        new_poly = make_tree(nx, ny, nd)

        ok = True
        for j, other in enumerate(polys):
            if j != i and new_poly.distance(other) < cfg.collision_buffer:
                ok = False
                break

        if not ok:
            progress = (time.time() - start_time) / time_limit
            T = 3.0 * (1e-8 / 3.0) ** progress
            continue

        old_poly = polys[i]
        polys[i] = new_poly
        new_score = bbox_side_polys(polys)

        delta = new_score - cur_score
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current[i] = (nx, ny, nd)
            cur_score = new_score

            if cur_score < best_score:
                best = list(current)
                best_score = cur_score
        else:
            polys[i] = old_poly

        progress = (time.time() - start_time) / time_limit
        T = 3.0 * (1e-8 / 3.0) ** progress
        shift = max(0.001, 0.1 * cur_score)

    return best


# =============================================================================
# SPECIAL SMALL N SOLVERS
# =============================================================================

def solve_n1() -> List[Tuple[float, float, float]]:
    """Optimal solution for n=1."""
    return [(0.0, 0.0, float(OPTIMAL_ANGLE))]


def solve_n2(cfg: Config) -> List[Tuple[float, float, float]]:
    """Grid search for n=2."""
    best = None
    best_score = float('inf')

    for a1 in range(0, 180, 3):
        for a2 in range(0, 180, 3):
            for spacing in np.arange(0.55, 1.1, 0.01):
                p = [(0.0, 0.0, float(a1)), (spacing, 0.0, float(a2))]

                if check_overlaps(p, cfg.collision_buffer):
                    continue

                side = bbox_side(p)
                if side < best_score:
                    best_score = side
                    best = p

    return center(best) if best else center([(0.0, 0.0, 45.0), (0.75, 0.0, 45.0)])


def solve_n3(cfg: Config) -> List[Tuple[float, float, float]]:
    """Grid search for n=3 (triangle formations)."""
    best = None
    best_score = float('inf')

    for angle in range(0, 180, 5):
        for sx in np.arange(0.55, 1.0, 0.02):
            for sy in np.arange(0.4, 1.0, 0.02):
                # Triangle formation
                p = [
                    (0.0, 0.0, float(angle)),
                    (sx, 0.0, float(angle)),
                    (sx / 2, sy, float(angle)),
                ]

                if check_overlaps(p, cfg.collision_buffer):
                    continue

                side = bbox_side(p)
                if side < best_score:
                    best_score = side
                    best = p

    return center(best) if best else center(pattern_grid(3))


def solve_n4(cfg: Config) -> List[Tuple[float, float, float]]:
    """Grid search for n=4 (2x2 formations)."""
    best = None
    best_score = float('inf')

    for angle in range(0, 180, 5):
        for sx in np.arange(0.55, 1.0, 0.02):
            for sy in np.arange(0.55, 1.0, 0.02):
                for offset in np.arange(0, sx, 0.05):
                    p = [
                        (0.0, 0.0, float(angle)),
                        (sx, 0.0, float(angle)),
                        (offset, sy, float(angle)),
                        (offset + sx, sy, float(angle)),
                    ]

                    if check_overlaps(p, cfg.collision_buffer):
                        continue

                    side = bbox_side(p)
                    if side < best_score:
                        best_score = side
                        best = p

    return center(best) if best else center(pattern_grid(4))


def solve_small_n(n: int, cfg: Config, time_limit: float) -> List[Tuple[float, float, float]]:
    """Specialized solver for small n."""
    if n == 1:
        return solve_n1()
    elif n == 2:
        return solve_n2(cfg)
    elif n == 3:
        return solve_n3(cfg)
    elif n == 4:
        return solve_n4(cfg)

    # For n > 4, use exhaustive initialization + optimization
    start_time = time.time()
    best = None
    best_score = float('inf')

    # Try many patterns
    patterns = [
        pattern_grid(n),
        pattern_hex(n),
        pattern_spiral(n),
        pattern_concentric(n),
        pattern_interlocked(n),
        pattern_diamond(n),
    ]

    # Also try different angles
    for base_angle in [0, 15, 30, 45, 60, 75, 90]:
        patterns.append(pattern_grid(n, base_angle))
        patterns.append(pattern_hex(n, base_angle))

    for init in patterns:
        if time.time() - start_time >= time_limit * 0.5:
            break

        if check_overlaps(init, cfg.collision_buffer):
            continue

        remaining = time_limit - (time.time() - start_time)

        # Optimize
        if HAS_CMA and remaining > 1.0:
            opt = CMAOptimizer(n, cfg)
            result = opt.optimize(init, remaining * 0.2)
        else:
            result = simulated_annealing(init, cfg, remaining * 0.2)

        result = local_search(result, cfg, remaining * 0.1)
        result = compact(result, cfg, remaining * 0.05)
        result = local_search(result, cfg, remaining * 0.05)

        if not check_overlaps(result, cfg.collision_buffer):
            side = bbox_side(result)
            if side < best_score:
                best_score = side
                best = result

    return center(best) if best else center(pattern_grid(n))


# =============================================================================
# MAIN SOLVER
# =============================================================================

class EliteSolverV3:
    """Main solver combining all techniques."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.scores: Dict[int, float] = {}

        self.nfp_placer = NFPPlacer(cfg) if HAS_PYCLIPPER else None
        self.blf_placer = BLFPlacer(cfg)

    def solve_puzzle(self, n: int, time_budget: float) -> List[Tuple[float, float, float]]:
        """Solve a single puzzle with all techniques."""
        if n == 0:
            return []

        start_time = time.time()

        # Small n: specialized solvers
        if n <= self.cfg.small_n_threshold:
            return solve_small_n(n, self.cfg, time_budget)

        best = None
        best_score = float('inf')

        # Generate initializations
        inits = [
            ('grid', pattern_grid(n)),
            ('hex', pattern_hex(n)),
            ('spiral', pattern_spiral(n)),
            ('concentric', pattern_concentric(n)),
            ('interlocked', pattern_interlocked(n)),
            ('diamond', pattern_diamond(n)),
        ]

        # Add different angle variants
        for angle in [0, 30, 60, 90]:
            inits.append((f'grid_{angle}', pattern_grid(n, angle)))

        # Add NFP-based initialization
        if self.nfp_placer and time.time() - start_time < time_budget * 0.2:
            try:
                nfp_init = self.nfp_placer.place(n)
                inits.append(('nfp', nfp_init))
            except:
                pass

        # Multi-restart optimization
        time_per_restart = time_budget / max(1, self.cfg.num_restarts)

        for restart in range(self.cfg.num_restarts):
            if time.time() - start_time >= time_budget * 0.92:
                break

            # Select initialization
            if restart < len(inits):
                name, init = inits[restart]
            elif best is not None:
                # Perturb best
                init = self.perturb(best, 0.08 * best_score)
            else:
                init = inits[restart % len(inits)][1]

            if check_overlaps(init, self.cfg.collision_buffer):
                continue

            remaining = min(time_per_restart, time_budget - (time.time() - start_time))

            # CMA-ES or SA optimization
            if HAS_CMA and remaining > 1.0:
                opt = CMAOptimizer(n, self.cfg)
                result = opt.optimize(init, remaining * 0.4)
            else:
                result = simulated_annealing(init, self.cfg, remaining * 0.4, seed=restart)

            # Refinement
            remaining = time_budget - (time.time() - start_time)
            if remaining > 0.5:
                result = local_search(result, self.cfg, remaining * 0.12)
                result = compact(result, self.cfg, remaining * 0.08)
                result = local_search(result, self.cfg, remaining * 0.05)

            if not check_overlaps(result, self.cfg.collision_buffer):
                side = bbox_side(result)
                if side < best_score:
                    best_score = side
                    best = result

        # Final polish
        if best is not None:
            remaining = time_budget - (time.time() - start_time)
            if remaining > 0.5:
                best = compact(best, self.cfg, remaining * 0.4)
                best = local_search(best, self.cfg, remaining * 0.3)

        return center(best) if best else center(pattern_grid(n))

    def perturb(self, placements: List[Tuple[float, float, float]],
                magnitude: float) -> List[Tuple[float, float, float]]:
        """Perturb solution."""
        result = []
        for x, y, d in placements:
            nx = x + random.gauss(0, magnitude)
            ny = y + random.gauss(0, magnitude)
            nd = (d + random.uniform(-20, 20)) % 360
            result.append((nx, ny, nd))
        return result

    def solve_all(self, output_path: str = "submission.csv"):
        """Solve all puzzles."""
        start_time = time.time()
        max_time = self.cfg.max_hours * 3600

        total_priority = sum(1.0 + (self.cfg.priority_multiplier - 1.0) * (200 - n) / 199
                            for n in range(1, 201))

        pbar = tqdm(range(1, 201), desc="Solving", ncols=100) if HAS_TQDM else range(1, 201)

        for n in pbar:
            elapsed = time.time() - start_time
            if elapsed >= max_time:
                print(f"\nTime limit at n={n}")
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

            side = bbox_side(placements)
            self.scores[n] = side

            total_score = sum((self.scores[k] ** 2) / k for k in self.scores)

            if HAS_TQDM:
                pbar.set_postfix({
                    'score': f'{total_score:.2f}',
                    'side': f'{side:.4f}',
                    'time': f'{time_budget:.1f}s'
                })

            if n % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint()

        # Fill missing
        for n in range(1, 201):
            if n not in self.solutions:
                self.solutions[n] = pattern_grid(n)
                self.scores[n] = bbox_side(self.solutions[n])

        self.save_submission(output_path)
        return self.solutions

    def total_score(self) -> float:
        return sum((self.scores.get(n, bbox_side(self.solutions.get(n, []))) ** 2) / n
                  for n in range(1, 201) if n in self.solutions)

    def save_checkpoint(self):
        self.save_submission(self.cfg.checkpoint_path)

    def save_submission(self, path: str):
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
    parser = argparse.ArgumentParser(description='Santa 2025 Elite Solver V3')
    parser.add_argument('--output', '-o', default='submission_elite_v3.csv')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--standard', action='store_true')
    parser.add_argument('--ultra', action='store_true')
    parser.add_argument('--hours', type=float)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.quick:
        cfg = quick_config()
    elif args.ultra:
        cfg = ultra_config()
    else:
        cfg = standard_config()

    if args.hours:
        cfg.max_hours = args.hours

    cfg.seed = args.seed

    print("=" * 70)
    print("SANTA 2025 - ELITE SOLVER V3 (Ultimate)")
    print("=" * 70)
    print(f"Max hours: {cfg.max_hours}")
    print(f"Time priority multiplier: {cfg.priority_multiplier}x")
    print(f"CMA population: {cfg.cma_population}")
    print(f"Restarts: {cfg.num_restarts}")
    print(f"CMA-ES: {'enabled' if HAS_CMA else 'disabled'}")
    print(f"NFP: {'enabled' if HAS_PYCLIPPER else 'disabled'}")
    print(f"Optimal angle: {OPTIMAL_ANGLE}")
    print(f"Seed: {cfg.seed}")
    print("=" * 70)

    solver = EliteSolverV3(cfg)
    solutions = solver.solve_all(args.output)

    total_score = solver.total_score()
    print()
    print("=" * 70)
    print(f"FINAL SCORE: {total_score:.4f}")
    print(f"Saved to: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
