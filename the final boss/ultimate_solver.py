#!/usr/bin/env python3
"""
Santa 2025 - ULTIMATE Solver (Self-Contained)
==============================================

The single best optimization solver combining all techniques:
- No-Fit Polygon (NFP) placement using pyclipper
- Radial greedy placement with fine-stepping
- Hexagonal grid placement (optimal for packing)
- Boundary placement along convex hull
- Priority-based time allocation (more time for low-n puzzles)
- Multi-restart strategy with diverse seeds
- Simulated Annealing with 6 move types
- Ultra-fine local search (0.0003 precision)
- Aggressive compacting toward centroid
- Basin hopping for escaping local minima
- STRtree spatial indexing for fast collision detection
- Live progress bar for real-time monitoring

Usage:
    python ultimate_solver.py [--output submission.csv] [--seed 42]
    python ultimate_solver.py --quick    # Fast test run (~1-2 hours)
    python ultimate_solver.py --standard # Standard run (~6-12 hours)
    python ultimate_solver.py --ultra    # Maximum quality (~24-48 hours)

Requirements:
    pip install numpy shapely pyclipper tqdm
"""

import os
import sys
import math
import time
import random
import argparse
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon, Point
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# Try to import pyclipper for NFP computation
try:
    import pyclipper
    HAS_PYCLIPPER = True
except ImportError:
    HAS_PYCLIPPER = False

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UltimateConfig:
    """Configuration for ultimate optimization."""

    # Time budgets
    base_time_per_puzzle: float = 30.0    # Base time per puzzle (seconds)
    priority_multiplier: float = 8.0       # n=1 gets 8x more time than n=200
    max_total_hours: float = 24.0          # Maximum total runtime

    # Multi-restart
    base_restarts: int = 100               # Restarts for n=200
    priority_restarts: int = 500           # Extra restarts for n=1
    num_seeds: int = 5                     # Seeds per restart batch

    # Placement
    num_placement_attempts: int = 300
    start_radius: float = 20.0
    step_in: float = 0.05
    step_out: float = 0.001

    # Collision
    collision_buffer: float = 0.012

    # Simulated Annealing
    sa_temp_initial: float = 3.0
    sa_temp_final: float = 0.0000001

    # Local search
    local_precision: float = 0.0003
    local_iterations: int = 2000

    # Compacting
    compact_passes: int = 40
    compact_step: float = 0.0002

    # Basin hopping
    basin_hops: int = 15
    basin_perturbation: float = 0.12

    # NFP
    use_nfp: bool = True
    nfp_precision: int = 1000

    # Checkpointing
    checkpoint_interval: int = 10
    checkpoint_path: str = "checkpoint.csv"

    seed: int = 42

    def get_time_for_n(self, n: int) -> float:
        """Get time budget for puzzle n (more for smaller n)."""
        priority = 1.0 + (self.priority_multiplier - 1.0) * (200 - n) / 199
        return self.base_time_per_puzzle * priority

    def get_restarts_for_n(self, n: int) -> int:
        """Get restarts for puzzle n (more for smaller n)."""
        extra = int((self.priority_restarts - self.base_restarts) * (200 - n) / 199)
        return self.base_restarts + extra


def quick_config() -> UltimateConfig:
    """Quick test mode (~1-2 hours)."""
    return UltimateConfig(
        base_time_per_puzzle=8.0,
        priority_multiplier=5.0,
        max_total_hours=2.0,
        base_restarts=20,
        priority_restarts=100,
        num_placement_attempts=100,
        local_iterations=500,
        compact_passes=15,
        basin_hops=5,
        checkpoint_interval=20
    )


def standard_config() -> UltimateConfig:
    """Standard mode (~6-12 hours)."""
    return UltimateConfig()


def ultra_config() -> UltimateConfig:
    """Ultra quality mode (~24-48 hours)."""
    return UltimateConfig(
        base_time_per_puzzle=120.0,
        priority_multiplier=15.0,
        max_total_hours=48.0,
        base_restarts=300,
        priority_restarts=2000,
        num_placement_attempts=500,
        step_in=0.03,
        step_out=0.0005,
        collision_buffer=0.01,
        local_precision=0.0002,
        local_iterations=5000,
        compact_passes=80,
        compact_step=0.0001,
        basin_hops=30,
        basin_perturbation=0.15,
        checkpoint_interval=5
    )


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

# Pre-compute rotated polygons (every 1 degree for precision)
ROTATED_POLYGONS = {deg: affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))
                    for deg in range(0, 360, 1)}


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
# NO-FIT POLYGON (NFP) COMPUTATION
# =============================================================================

class NFPComputer:
    """No-Fit Polygon computation using pyclipper (Minkowski sum)."""

    def __init__(self, precision: int = 1000):
        self.precision = precision
        self.cache = {}

    def _to_clipper(self, coords: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        return [(int(x * self.precision), int(y * self.precision)) for x, y in coords]

    def _from_clipper(self, coords: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        return [(x / self.precision, y / self.precision) for x, y in coords]

    def _negate_polygon(self, coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [(-x, -y) for x, y in coords]

    def compute_nfp(self, fixed_poly: Polygon, moving_poly: Polygon) -> Optional[Polygon]:
        """Compute NFP using Minkowski sum."""
        if not HAS_PYCLIPPER:
            return None

        try:
            fixed_coords = list(fixed_poly.exterior.coords)[:-1]
            moving_coords = list(moving_poly.exterior.coords)[:-1]

            neg_moving = self._negate_polygon(moving_coords)
            fixed_int = self._to_clipper(fixed_coords)
            moving_int = self._to_clipper(neg_moving)

            result = pyclipper.MinkowskiSum(fixed_int, moving_int, True)

            if result and len(result) > 0:
                nfp_coords = self._from_clipper(result[0])
                return Polygon(nfp_coords)
        except Exception:
            pass

        return None

    def compute_combined_nfp(self, fixed_polys: List[Polygon], moving_poly: Polygon):
        """Compute combined NFP for multiple fixed polygons."""
        nfps = []
        for fp in fixed_polys:
            nfp = self.compute_nfp(fp, moving_poly)
            if nfp and nfp.is_valid:
                nfps.append(nfp)

        if nfps:
            return unary_union(nfps)
        return None

    def find_valid_position(self, nfp, center: Tuple[float, float] = (0, 0),
                            num_samples: int = 150) -> Optional[Tuple[float, float]]:
        """Find valid position on NFP boundary closest to center."""
        if nfp is None:
            return None

        cx, cy = center
        best_pos = None
        best_dist = float('inf')

        boundary = nfp.exterior if hasattr(nfp, 'exterior') else nfp.boundary
        length = boundary.length

        for i in range(num_samples):
            t = i / num_samples
            point = boundary.interpolate(t * length)
            px, py = point.x, point.y

            dx = px - cx
            dy = py - cy
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0:
                offset = 0.015
                test_x = px + offset * dx / dist
                test_y = py + offset * dy / dist

                test_point = Point(test_x, test_y)
                if not nfp.contains(test_point):
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (test_x, test_y)

        return best_pos


nfp_computer = NFPComputer(precision=1000)


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


def place_hexagonal(base: Polygon, existing: List[Polygon], cfg: UltimateConfig) -> Tuple[float, float]:
    """Place using hexagonal grid pattern - optimal for circle packing."""
    if not existing:
        return 0.0, 0.0

    idx = STRtree(existing)
    best = (None, None)
    min_dist = float('inf')

    dx = 0.75
    dy = dx * math.sqrt(3) / 2

    for ring in range(25):
        for i in range(-ring, ring + 1):
            for j in range(-ring, ring + 1):
                if abs(i) != ring and abs(j) != ring:
                    continue

                x = i * dx + (j % 2) * dx / 2
                y = j * dy

                cand = affinity.translate(base, xoff=x, yoff=y)
                if not collides_fast(cand, idx, existing, cfg.collision_buffer):
                    dist = x*x + y*y
                    if dist < min_dist:
                        min_dist = dist
                        best = (x, y)

    return best[0] if best[0] is not None else 0.0, best[1] if best[1] is not None else 0.0


def place_boundary(base: Polygon, existing: List[Polygon], cfg: UltimateConfig) -> Tuple[float, float]:
    """Place on boundary of existing polygon union."""
    if not existing:
        return 0.0, 0.0

    idx = STRtree(existing)
    hull = unary_union(existing).convex_hull
    boundary = hull.boundary

    best = (None, None)
    min_dist = float('inf')

    length = boundary.length
    for i in range(150):
        t = i / 150.0
        pt = boundary.interpolate(t * length)

        for offset in np.arange(-0.4, 0.4, 0.08):
            for angle in range(0, 360, 20):
                rad = math.radians(angle)
                x = pt.x + offset * math.cos(rad)
                y = pt.y + offset * math.sin(rad)

                cand = affinity.translate(base, xoff=x, yoff=y)
                if not collides_fast(cand, idx, existing, cfg.collision_buffer):
                    dist = x*x + y*y
                    if dist < min_dist:
                        min_dist = dist
                        best = (x, y)

    return best[0] if best[0] is not None else 0.0, best[1] if best[1] is not None else 0.0


def place_nfp(base: Polygon, existing: List[Polygon], cfg: UltimateConfig) -> Optional[Tuple[float, float]]:
    """Place using No-Fit Polygon."""
    if not existing or not cfg.use_nfp or not HAS_PYCLIPPER:
        return None

    combined_nfp = nfp_computer.compute_combined_nfp(existing, base)
    if combined_nfp is not None:
        pos = nfp_computer.find_valid_position(combined_nfp, center=(0, 0), num_samples=200)
        if pos:
            test_poly = affinity.translate(base, xoff=pos[0], yoff=pos[1])
            if not collides(test_poly, existing, cfg.collision_buffer):
                return pos
    return None


# =============================================================================
# SIMULATED ANNEALING - ULTIMATE
# =============================================================================

def sa_ultimate(placements: List[Tuple[float, float, float]], cfg: UltimateConfig,
                time_limit: float, seed: int = 42) -> List[Tuple[float, float, float]]:
    """Ultimate Simulated Annealing with 6 move types."""
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

    shift = 0.12 * cur_score

    while time.time() - start < time_limit:
        iters += 1

        i = random.randrange(n)
        x, y, d = current[i]

        mt = random.random()

        if mt < 0.15:
            # Very precise move
            nx = x + random.gauss(0, shift * 0.08)
            ny = y + random.gauss(0, shift * 0.08)
            nd = d
        elif mt < 0.35:
            # Small move
            nx = x + random.gauss(0, shift * 0.25)
            ny = y + random.gauss(0, shift * 0.25)
            nd = d
        elif mt < 0.50:
            # Medium move
            nx = x + random.gauss(0, shift * 0.6)
            ny = y + random.gauss(0, shift * 0.6)
            nd = d
        elif mt < 0.60:
            # Large escape
            nx = x + random.uniform(-shift * 2.5, shift * 2.5)
            ny = y + random.uniform(-shift * 2.5, shift * 2.5)
            nd = d
        elif mt < 0.72:
            # Fine rotation
            nx, ny = x, y
            nd = (d + random.choice([-1, 1, -2, 2, -3, 3, -5, 5])) % 360
        elif mt < 0.82:
            # Larger rotation
            nx, ny = x, y
            nd = (d + random.choice([-10, 10, -15, 15, -30, 30, -45, 45, -90, 90])) % 360
        elif mt < 0.92:
            # Center seeking
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 0.001
            step = random.uniform(0.01, 0.08) * cur_score
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

        # Adaptive step sizes
        if iters % 2000 == 0:
            rate = accepted / iters
            if rate > 0.35:
                shift = min(cur_score * 0.4, shift * 1.08)
            elif rate < 0.12:
                shift = max(0.001, shift * 0.92)

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

            # Multi-scale moves
            moves = []
            for scale in [0.25, 0.5, 1.0, 1.5, 2.0]:
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

            # Fine rotations
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

            if best_move is not None and best_improvement > 1e-8:
                current[i] = (best_move[0], best_move[1], best_move[2])
                polys[i] = best_move[3]
                cur_score -= best_improvement
                improved = True

    return current


# =============================================================================
# COMPACTING - AGGRESSIVE
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

        cx = sum(p[0] for p in current) / n
        cy = sum(p[1] for p in current) / n

        dists = [(i, math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)) for i, p in enumerate(current)]
        dists.sort(key=lambda x: -x[1])

        for idx, dist in dists:
            if dist < 0.015:
                continue

            x, y, d = current[idx]
            dx, dy = cx - x, cy - y
            norm = math.sqrt(dx*dx + dy*dy) + 0.001

            for mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]:
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
            elif opt_score < cur_score * 1.03:
                current = optimized
                cur_score = opt_score

    return best


# =============================================================================
# PROGRESS BAR
# =============================================================================

class ProgressBar:
    """Simple progress bar for terminal output."""

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
# MAIN SOLVER
# =============================================================================

class UltimateSolver:
    """Ultimate solver with all optimization techniques."""

    def __init__(self, config: Optional[UltimateConfig] = None, verbose: bool = True):
        self.cfg = config or UltimateConfig()
        self.verbose = verbose

        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.scores: Dict[int, float] = {}

        self.start_time = None

    def solve_single(self, n: int, prev: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float, float]], float]:
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
                time_budget = min(time_budget, 5.0)
                num_restarts = min(num_restarts, 10)

        prev_polys = [make_tree(x, y, d) for x, y, d in prev]

        best_sol = None
        best_score = float('inf')

        angles = list(range(0, 360, 5))  # 72 angles

        for restart in range(num_restarts):
            if time.time() - n_start >= time_budget * 0.95:
                break

            seed = self.cfg.seed + restart * 1000 + n
            random.seed(seed)

            angle = angles[restart % len(angles)]
            base = ROTATED_POLYGONS[angle]

            # Placement strategy rotation
            strat = restart % 5
            if strat == 0:
                pos = place_nfp(base, prev_polys, self.cfg)
                if pos is None:
                    bx, by = place_radial(base, prev_polys, self.cfg)
                else:
                    bx, by = pos
            elif strat == 1:
                bx, by = place_radial(base, prev_polys, self.cfg)
            elif strat == 2:
                bx, by = place_hexagonal(base, prev_polys, self.cfg)
            elif strat == 3:
                bx, by = place_boundary(base, prev_polys, self.cfg)
            else:
                # Best of all strategies
                candidates = []
                for fn in [place_radial, place_hexagonal, place_boundary]:
                    px, py = fn(base, prev_polys, self.cfg)
                    candidates.append((px*px + py*py, px, py))
                nfp_pos = place_nfp(base, prev_polys, self.cfg)
                if nfp_pos:
                    candidates.append((nfp_pos[0]**2 + nfp_pos[1]**2, nfp_pos[0], nfp_pos[1]))
                candidates.sort()
                bx, by = candidates[0][1], candidates[0][2]

            sol = prev + [(bx, by, float(angle))]

            # Perturb from best on later restarts
            if restart > 0 and best_sol is not None and restart % 2 != 0:
                sol = list(best_sol)
                num_p = max(1, int(n * 0.12))
                for idx in random.sample(range(n), min(num_p, n)):
                    x, y, d = sol[idx]
                    sol[idx] = (
                        x + random.gauss(0, 0.025 * best_score),
                        y + random.gauss(0, 0.025 * best_score),
                        (d + random.uniform(-12, 12)) % 360
                    )

            if check_overlaps(sol, self.cfg.collision_buffer):
                continue

            # Optimization pipeline
            remaining = time_budget - (time.time() - n_start)
            per_restart = remaining / max(1, num_restarts - restart)

            if per_restart > 0.3:
                sol = sa_ultimate(sol, self.cfg, per_restart * 0.5, seed)
                sol = local_search_ultra(sol, self.cfg)
                sol = compact_aggressive(sol, self.cfg)
                sol = local_search_ultra(sol, self.cfg)

            score = bbox_side(sol)
            if not check_overlaps(sol, self.cfg.collision_buffer) and score < best_score:
                best_score = score
                best_sol = sol

        # Final basin hopping
        if best_sol is not None and time.time() - n_start < time_budget * 0.9:
            remaining = time_budget - (time.time() - n_start)
            if remaining > 2.0:
                best_sol = basin_hop(best_sol, self.cfg, remaining * 0.4)
                best_sol = compact_aggressive(best_sol, self.cfg)
                best_sol = local_search_ultra(best_sol, self.cfg)
                best_score = bbox_side(best_sol)

        if best_sol is None:
            angle = random.uniform(0, 360)
            base = affinity.rotate(BASE_POLYGON, angle, origin=(0, 0))
            bx, by = place_radial(base, prev_polys, self.cfg)
            best_sol = prev + [(bx, by, angle)]
            best_score = bbox_side(best_sol)

        return center(best_sol), best_score

    def solve_all(self, max_n: int = 200) -> Dict[int, List[Tuple[float, float, float]]]:
        """Solve for all n from 1 to max_n."""
        self.start_time = time.time()

        if self.verbose:
            print("=" * 70)
            print("SANTA 2025 - ULTIMATE SOLVER")
            print("=" * 70)
            print(f"Base time per puzzle: {self.cfg.base_time_per_puzzle:.0f}s")
            print(f"Priority multiplier: {self.cfg.priority_multiplier:.1f}x")
            print(f"Base restarts: {self.cfg.base_restarts}")
            print(f"Priority restarts: {self.cfg.priority_restarts}")
            print(f"NFP enabled: {self.cfg.use_nfp and HAS_PYCLIPPER}")
            print(f"Max hours: {self.cfg.max_total_hours:.1f}")
            print()

            est_time = sum(self.cfg.get_time_for_n(n) for n in range(1, max_n + 1))
            print(f"Estimated time: {est_time / 3600:.1f} hours")
            print()

        # Use tqdm if available, otherwise our custom progress bar
        if HAS_TQDM:
            pbar = tqdm(range(1, max_n + 1), desc="Solving", unit="puzzle",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
        else:
            pbar = ProgressBar(max_n, desc="Solving")

        for n in range(1, max_n + 1):
            n_start = time.time()

            prev = self.solutions[n - 1] if n > 1 else []
            sol, score = self.solve_single(n, prev)

            self.solutions[n] = sol
            self.scores[n] = score

            elapsed_n = time.time() - n_start
            total_score = self.total_score()

            if HAS_TQDM:
                pbar.set_postfix({'score': f'{total_score:.2f}', 'side': f'{score:.4f}'})
                pbar.update(1)
            else:
                pbar.update(1, extra=f"n={n}, side={score:.4f}, total={total_score:.2f}")

            # Checkpoint
            if n % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint()

        if HAS_TQDM:
            pbar.close()

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
        """Save checkpoint."""
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
        description="Santa 2025 Ultimate Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ultimate_solver.py --output submission.csv
  python ultimate_solver.py --quick --output test.csv
  python ultimate_solver.py --standard --output submission.csv
  python ultimate_solver.py --ultra --output best.csv
"""
    )
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--max-n", type=int, default=200, help="Maximum n to solve")
    parser.add_argument("--quick", action="store_true", help="Quick mode (~1-2 hours)")
    parser.add_argument("--standard", action="store_true", help="Standard mode (~6-12 hours)")
    parser.add_argument("--ultra", action="store_true", help="Ultra mode (~24-48 hours)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Select configuration
    if args.quick:
        cfg = quick_config()
        print("Using QUICK mode (~1-2 hours)")
    elif args.ultra:
        cfg = ultra_config()
        print("Using ULTRA mode (~24-48 hours)")
    else:
        cfg = standard_config()
        print("Using STANDARD mode (~6-12 hours)")

    cfg.seed = args.seed

    print()
    if not HAS_PYCLIPPER:
        print("Warning: pyclipper not installed. Install with: pip install pyclipper")
    if not HAS_TQDM:
        print("Note: Install tqdm for better progress bars: pip install tqdm")
    print()

    # Solve
    solver = UltimateSolver(config=cfg, verbose=True)
    solutions = solver.solve_all(max_n=args.max_n)

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
