#!/usr/bin/env python3
"""
Santa 2025 - EXTREME Optimization Solver
=========================================

Extreme optimizations for sub-55 score:
- No-Fit Polygon (NFP) using pyclipper for precise positioning
- 20+ multi-restart strategy
- 50,000+ SA iterations with 5 move types
- 72 rotation angles (every 5 degrees)
- 60 seconds per puzzle (~200 min total)
- Fine-grained 0.005 precision local search

Usage:
    python solver.py [--output submission.csv] [--seed 42]
"""

import os
import sys
import math
import time
import random
import argparse
import multiprocessing as mp
from decimal import Decimal, getcontext
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapely.prepared import prep

# Try to import pyclipper for NFP computation
try:
    import pyclipper
    HAS_PYCLIPPER = True
except ImportError:
    HAS_PYCLIPPER = False
    print("Warning: pyclipper not found. NFP will use fallback method.")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExtremeConfig:
    """Configuration for extreme optimization."""
    # Time budget
    time_per_puzzle: float = 60.0  # 60 seconds per puzzle
    total_time_limit: float = 12000.0  # ~200 minutes total

    # Multi-restart
    num_restarts: int = 25  # 20+ restart configurations
    num_seeds: int = 5  # Different random seeds per restart

    # Greedy placement
    num_placement_attempts: int = 200  # Many more attempts
    start_radius: float = 25.0
    step_in: float = 0.1  # Finer step
    step_out: float = 0.005  # Very fine backup step (0.005 precision)

    # Rotation angles
    num_rotation_angles: int = 72  # Every 5 degrees

    # Simulated Annealing
    sa_iterations_base: int = 50000  # 50,000+ iterations
    sa_temp_initial: float = 2.0
    sa_temp_final: float = 0.00001
    sa_move_weights: Tuple[float, ...] = (0.30, 0.15, 0.25, 0.15, 0.15)  # 5 move types

    # Local search
    local_search_precision: float = 0.005
    local_search_iterations: int = 500

    # NFP
    use_nfp: bool = True
    nfp_precision: int = 1000  # Scaling for pyclipper

    seed: int = 42


# =============================================================================
# GEOMETRY - Tree Polygon Definition
# =============================================================================

# High precision settings
getcontext().prec = 25

# Tree dimensions (from official code)
TRUNK_W = 0.15
TRUNK_H = 0.2
BASE_W = 0.7
MID_W = 0.4
TOP_W = 0.25
TIP_Y = 0.8
TIER_1_Y = 0.5
TIER_2_Y = 0.25
BASE_Y = 0.0
TRUNK_BOTTOM_Y = -TRUNK_H

# Tree polygon coordinates (15 vertices)
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
TREE_WIDTH = 0.7
TREE_HEIGHT = 1.0

# Pre-compute rotated polygons for all 72 angles
ROTATED_POLYGONS = {}
for deg in range(0, 360, 5):
    ROTATED_POLYGONS[deg] = affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))


def make_tree_polygon(x: float, y: float, angle_deg: float) -> Polygon:
    """Create tree polygon at position with rotation."""
    # Snap to nearest 5-degree angle for cache
    snapped_angle = int(round(angle_deg / 5) * 5) % 360
    poly = ROTATED_POLYGONS.get(snapped_angle, BASE_POLYGON)
    if snapped_angle != int(angle_deg):
        poly = affinity.rotate(BASE_POLYGON, angle_deg, origin=(0, 0))
    if x != 0 or y != 0:
        poly = affinity.translate(poly, xoff=x, yoff=y)
    return poly


def make_tree_polygon_fast(x: float, y: float, angle_deg: float) -> Polygon:
    """Fast tree polygon creation using cached rotations."""
    snapped = int(round(angle_deg / 5) * 5) % 360
    poly = ROTATED_POLYGONS[snapped]
    if x != 0 or y != 0:
        return affinity.translate(poly, xoff=x, yoff=y)
    return poly


# =============================================================================
# NO-FIT POLYGON (NFP) IMPLEMENTATION
# =============================================================================

class NFPComputer:
    """
    No-Fit Polygon computation using pyclipper (Minkowski sum).

    NFP allows finding all valid positions where a polygon can be placed
    without overlapping another polygon.
    """

    def __init__(self, precision: int = 1000):
        self.precision = precision
        self.cache = {}

    def _to_clipper(self, coords: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Convert float coords to integer for pyclipper."""
        return [(int(x * self.precision), int(y * self.precision)) for x, y in coords]

    def _from_clipper(self, coords: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Convert integer coords back to float."""
        return [(x / self.precision, y / self.precision) for x, y in coords]

    def _negate_polygon(self, coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Negate polygon coordinates for Minkowski difference."""
        return [(-x, -y) for x, y in coords]

    def compute_nfp(self, fixed_poly: Polygon, moving_poly: Polygon) -> Optional[Polygon]:
        """
        Compute No-Fit Polygon using Minkowski sum.

        The NFP is the Minkowski sum of the fixed polygon and the negated moving polygon.
        A point is inside the NFP if placing the moving polygon there causes overlap.
        """
        if not HAS_PYCLIPPER:
            return None

        try:
            fixed_coords = list(fixed_poly.exterior.coords)[:-1]
            moving_coords = list(moving_poly.exterior.coords)[:-1]

            # Negate moving polygon for Minkowski difference
            neg_moving = self._negate_polygon(moving_coords)

            # Convert to clipper format
            fixed_int = self._to_clipper(fixed_coords)
            moving_int = self._to_clipper(neg_moving)

            # Compute Minkowski sum
            result = pyclipper.MinkowskiSum(fixed_int, moving_int, True)

            if result and len(result) > 0:
                nfp_coords = self._from_clipper(result[0])
                return Polygon(nfp_coords)
        except Exception:
            pass

        return None

    def compute_combined_nfp(
        self,
        fixed_polys: List[Polygon],
        moving_poly: Polygon
    ) -> Optional[MultiPolygon]:
        """Compute combined NFP for multiple fixed polygons."""
        nfps = []
        for fp in fixed_polys:
            nfp = self.compute_nfp(fp, moving_poly)
            if nfp and nfp.is_valid:
                nfps.append(nfp)

        if nfps:
            combined = unary_union(nfps)
            return combined
        return None

    def find_valid_position_on_nfp_boundary(
        self,
        nfp: Polygon,
        center: Tuple[float, float] = (0, 0),
        num_samples: int = 100
    ) -> Optional[Tuple[float, float]]:
        """
        Find a valid position on the NFP boundary closest to center.

        Valid positions are OUTSIDE the NFP (no overlap).
        """
        if nfp is None:
            return None

        cx, cy = center
        best_pos = None
        best_dist = float('inf')

        # Sample boundary points
        boundary = nfp.exterior if hasattr(nfp, 'exterior') else nfp.boundary
        length = boundary.length

        for i in range(num_samples):
            t = i / num_samples
            point = boundary.interpolate(t * length)
            px, py = point.x, point.y

            # Small offset to ensure we're outside
            dx = px - cx
            dy = py - cy
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0:
                # Move slightly outward
                offset = 0.01
                test_x = px + offset * dx / dist
                test_y = py + offset * dy / dist

                test_point = Point(test_x, test_y)
                if not nfp.contains(test_point):
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (test_x, test_y)

        return best_pos


# Global NFP computer
nfp_computer = NFPComputer(precision=1000)


# =============================================================================
# COLLISION DETECTION
# =============================================================================

def has_collision(tree_poly: Polygon, other_polys: List[Polygon]) -> bool:
    """Check if tree_poly overlaps any other polygon."""
    for poly in other_polys:
        if tree_poly.intersects(poly) and not tree_poly.touches(poly):
            return True
    return False


def has_collision_strtree(
    tree_poly: Polygon,
    tree_index: STRtree,
    all_polys: List[Polygon]
) -> bool:
    """Fast collision check using spatial index."""
    candidates = tree_index.query(tree_poly)
    for idx in candidates:
        if tree_poly.intersects(all_polys[idx]) and not tree_poly.touches(all_polys[idx]):
            return True
    return False


def has_collision_prepared(tree_poly: Polygon, prepared_polys: List) -> bool:
    """Fast collision check using prepared geometries."""
    for pp in prepared_polys:
        if pp.intersects(tree_poly) and not pp.touches(tree_poly):
            return True
    return False


# =============================================================================
# BOUNDING BOX AND SCORING
# =============================================================================

def compute_bounds(polygons: List[Polygon]) -> Tuple[float, float, float, float]:
    """Compute bounding box of all polygons."""
    if not polygons:
        return (0, 0, 0, 0)
    union = unary_union(polygons)
    return union.bounds


def compute_bounding_square_side(placements: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side from placements."""
    if not placements:
        return 0.0
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return max(width, height)


def bounding_square_side_from_polys(polygons: List[Polygon]) -> float:
    """Compute bounding square side from polygon list."""
    if not polygons:
        return 0.0
    bounds = unary_union(polygons).bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return max(width, height)


def compute_score(solutions: Dict[int, List[Tuple[float, float, float]]]) -> float:
    """Compute total competition score: sum of (side^2 / n)."""
    total = 0.0
    for n, placements in solutions.items():
        side = compute_bounding_square_side(placements)
        total += (side ** 2) / n
    return total


# =============================================================================
# PLACEMENT UTILITIES
# =============================================================================

def center_placements(
    placements: List[Tuple[float, float, float]]
) -> List[Tuple[float, float, float]]:
    """Center placements around origin."""
    if not placements:
        return placements
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]


def normalize_to_origin(
    placements: List[Tuple[float, float, float]]
) -> List[Tuple[float, float, float]]:
    """Shift placements so min x,y are at 0."""
    if not placements:
        return placements
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    return [(x - bounds[0], y - bounds[1], d) for x, y, d in placements]


def check_all_overlaps(
    placements: List[Tuple[float, float, float]]
) -> List[Tuple[int, int]]:
    """Find all overlapping pairs."""
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    overlaps = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                overlaps.append((i, j))
    return overlaps


# =============================================================================
# WEIGHTED ANGLE GENERATION
# =============================================================================

def generate_weighted_angle() -> float:
    """
    Generate random angle weighted by abs(sin(2*angle)).
    Favors diagonal directions (corners).
    """
    while True:
        angle = random.uniform(0, 2 * math.pi)
        if random.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle


def get_rotation_angles(num_angles: int = 72) -> List[float]:
    """Get list of rotation angles (every 360/num_angles degrees)."""
    return [i * 360.0 / num_angles for i in range(num_angles)]


# =============================================================================
# GREEDY PLACEMENT WITH NFP
# =============================================================================

def place_tree_greedy_nfp(
    base_polygon: Polygon,
    existing_polys: List[Polygon],
    config: ExtremeConfig
) -> Tuple[float, float]:
    """
    Place tree using NFP for precise valid position finding.
    """
    if not existing_polys:
        return 0.0, 0.0

    if config.use_nfp and HAS_PYCLIPPER:
        # Compute combined NFP
        combined_nfp = nfp_computer.compute_combined_nfp(existing_polys, base_polygon)

        if combined_nfp is not None:
            # Find valid position closest to center
            pos = nfp_computer.find_valid_position_on_nfp_boundary(
                combined_nfp, center=(0, 0), num_samples=200
            )
            if pos:
                # Verify it's valid
                test_poly = affinity.translate(base_polygon, xoff=pos[0], yoff=pos[1])
                if not has_collision(test_poly, existing_polys):
                    return pos

    # Fallback to radial approach
    return place_tree_greedy_radial(base_polygon, existing_polys, config)


def place_tree_greedy_radial(
    base_polygon: Polygon,
    existing_polys: List[Polygon],
    config: ExtremeConfig
) -> Tuple[float, float]:
    """
    Place tree using radial greedy approach with fine stepping.
    """
    if not existing_polys:
        return 0.0, 0.0

    # Build spatial index
    tree_index = STRtree(existing_polys)

    best_x, best_y = None, None
    min_radius = float('inf')

    for _ in range(config.num_placement_attempts):
        # Random angle weighted toward corners
        angle = generate_weighted_angle()
        vx = math.cos(angle)
        vy = math.sin(angle)

        # Start far and move toward center
        radius = config.start_radius
        collision_found = False

        while radius >= 0:
            px = radius * vx
            py = radius * vy

            candidate_poly = affinity.translate(base_polygon, xoff=px, yoff=py)

            if has_collision_strtree(candidate_poly, tree_index, existing_polys):
                collision_found = True
                break

            radius -= config.step_in

        # Fine backup until no collision
        if collision_found:
            while radius < config.start_radius * 2:
                radius += config.step_out  # 0.005 precision
                px = radius * vx
                py = radius * vy

                candidate_poly = affinity.translate(base_polygon, xoff=px, yoff=py)

                if not has_collision_strtree(candidate_poly, tree_index, existing_polys):
                    break
        else:
            radius = 0
            px, py = 0.0, 0.0

        # Keep best (closest to center)
        if radius < min_radius:
            min_radius = radius
            best_x, best_y = px, py

    return best_x if best_x is not None else 0.0, best_y if best_y is not None else 0.0


# =============================================================================
# ADVANCED SIMULATED ANNEALING WITH 5 MOVE TYPES
# =============================================================================

def simulated_annealing_advanced(
    placements: List[Tuple[float, float, float]],
    config: ExtremeConfig,
    time_limit: float,
    current_best_score: float = float('inf')
) -> Tuple[List[Tuple[float, float, float]], float]:
    """
    Advanced Simulated Annealing with 5 move types:
    1. Small moves (random walk) - 30%
    2. Large moves (escape local minima) - 15%
    3. Rotation changes - 25%
    4. Position swaps between trees - 15%
    5. Center-seeking moves - 15%
    """
    n = len(placements)
    if n <= 1:
        return placements, compute_bounding_square_side(placements)

    start_time = time.time()

    current = list(placements)
    polys = [make_tree_polygon(x, y, d) for x, y, d in current]

    current_score = bounding_square_side_from_polys(polys)
    best_score = current_score
    best = list(current)

    # Temperature schedule
    T = config.sa_temp_initial
    iterations = 0
    accepted = 0

    # Move type weights
    w_small, w_large, w_rotate, w_swap, w_center = config.sa_move_weights
    cum_weights = [w_small, w_small + w_large, w_small + w_large + w_rotate,
                   w_small + w_large + w_rotate + w_swap, 1.0]

    # Adaptive parameters
    max_shift_small = 0.05 * current_score
    max_shift_large = 0.3 * current_score
    max_rotate = 30.0

    while time.time() - start_time < time_limit:
        iterations += 1

        # Pick move type
        r = random.random()

        if r < cum_weights[0]:
            # Move Type 1: Small random walk
            i = random.randrange(n)
            x, y, deg = current[i]
            new_x = x + random.gauss(0, max_shift_small)
            new_y = y + random.gauss(0, max_shift_small)
            new_deg = deg
            move_type = "small"

        elif r < cum_weights[1]:
            # Move Type 2: Large escape move
            i = random.randrange(n)
            x, y, deg = current[i]
            new_x = x + random.uniform(-max_shift_large, max_shift_large)
            new_y = y + random.uniform(-max_shift_large, max_shift_large)
            new_deg = deg
            move_type = "large"

        elif r < cum_weights[2]:
            # Move Type 3: Rotation change
            i = random.randrange(n)
            x, y, deg = current[i]
            new_x, new_y = x, y
            # Use 5-degree increments for efficiency
            angle_change = random.choice([-5, 5, -10, 10, -15, 15, -30, 30, -45, 45])
            new_deg = (deg + angle_change) % 360
            move_type = "rotate"

        elif r < cum_weights[3]:
            # Move Type 4: Position swap
            if n < 2:
                continue
            i, j = random.sample(range(n), 2)
            # Swap positions but keep rotations
            x1, y1, d1 = current[i]
            x2, y2, d2 = current[j]

            # Try swapping
            new_current = list(current)
            new_current[i] = (x2, y2, d1)
            new_current[j] = (x1, y1, d2)

            new_polys = list(polys)
            new_polys[i] = make_tree_polygon(x2, y2, d1)
            new_polys[j] = make_tree_polygon(x1, y1, d2)

            # Check collisions
            valid = True
            for k in range(n):
                if k != i and k != j:
                    if has_collision(new_polys[i], [new_polys[k]]) or \
                       has_collision(new_polys[j], [new_polys[k]]):
                        valid = False
                        break
            if has_collision(new_polys[i], [new_polys[j]]):
                valid = False

            if valid:
                new_score = bounding_square_side_from_polys(new_polys)
                delta = new_score - current_score

                if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    current = new_current
                    polys = new_polys
                    current_score = new_score
                    accepted += 1

                    if current_score < best_score:
                        best_score = current_score
                        best = list(current)

            # Update temperature
            progress = (time.time() - start_time) / time_limit
            T = config.sa_temp_initial * math.pow(config.sa_temp_final / config.sa_temp_initial, progress)
            continue

        else:
            # Move Type 5: Center-seeking move
            i = random.randrange(n)
            x, y, deg = current[i]

            # Compute centroid of all trees
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n

            # Move toward center
            dx = cx - x
            dy = cy - y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0.01:
                step = random.uniform(0.01, 0.1) * current_score
                new_x = x + step * dx / dist
                new_y = y + step * dy / dist
            else:
                new_x, new_y = x, y
            new_deg = deg
            move_type = "center"

        if move_type != "swap":
            # Create new polygon and check collision
            new_poly = make_tree_polygon(new_x, new_y, new_deg)
            others = polys[:i] + polys[i+1:]

            if has_collision(new_poly, others):
                # Update temperature
                progress = (time.time() - start_time) / time_limit
                T = config.sa_temp_initial * math.pow(config.sa_temp_final / config.sa_temp_initial, progress)
                continue

            # Compute new score
            old_poly = polys[i]
            polys[i] = new_poly
            new_score = bounding_square_side_from_polys(polys)

            # Acceptance
            delta = new_score - current_score
            if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current[i] = (new_x, new_y, new_deg)
                current_score = new_score
                accepted += 1

                if current_score < best_score:
                    best_score = current_score
                    best = list(current)
            else:
                polys[i] = old_poly

        # Update temperature
        progress = (time.time() - start_time) / time_limit
        T = config.sa_temp_initial * math.pow(config.sa_temp_final / config.sa_temp_initial, progress)

        # Adaptive step sizes
        if iterations % 1000 == 0:
            accept_rate = accepted / iterations if iterations > 0 else 0
            if accept_rate > 0.3:
                max_shift_small *= 1.05
                max_shift_large *= 1.05
            elif accept_rate < 0.1:
                max_shift_small *= 0.95
                max_shift_large *= 0.95

    return best, best_score


# =============================================================================
# LOCAL SEARCH WITH FINE-GRAINED OPTIMIZATION
# =============================================================================

def local_search_fine(
    placements: List[Tuple[float, float, float]],
    precision: float = 0.005,
    max_iterations: int = 500
) -> List[Tuple[float, float, float]]:
    """
    Fine-grained local search with 0.005 precision.
    """
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)
    polys = [make_tree_polygon(x, y, d) for x, y, d in current]
    current_score = bounding_square_side_from_polys(polys)

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(n):
            x, y, deg = current[i]

            # Try small moves in 8 directions
            for dx, dy in [(-precision, 0), (precision, 0),
                           (0, -precision), (0, precision),
                           (-precision, -precision), (-precision, precision),
                           (precision, -precision), (precision, precision)]:
                new_x, new_y = x + dx, y + dy
                new_poly = make_tree_polygon(new_x, new_y, deg)

                others = polys[:i] + polys[i+1:]
                if not has_collision(new_poly, others):
                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bounding_square_side_from_polys(polys)

                    if new_score < current_score:
                        current[i] = (new_x, new_y, deg)
                        current_score = new_score
                        improved = True
                    else:
                        polys[i] = old_poly

            # Try small rotation changes
            for d_deg in [-5, 5]:
                new_deg = (deg + d_deg) % 360
                new_poly = make_tree_polygon(x, y, new_deg)

                others = polys[:i] + polys[i+1:]
                if not has_collision(new_poly, others):
                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bounding_square_side_from_polys(polys)

                    if new_score < current_score:
                        current[i] = (x, y, new_deg)
                        current_score = new_score
                        improved = True
                    else:
                        polys[i] = old_poly

    return current


# =============================================================================
# MAIN SOLVER WITH MULTI-RESTART
# =============================================================================

class ExtremeSolver:
    """
    Extreme optimization solver with multi-restart strategy.
    """

    def __init__(self, config: Optional[ExtremeConfig] = None, seed: int = 42):
        self.config = config or ExtremeConfig()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.scores: Dict[int, float] = {}
        self.rotation_angles = get_rotation_angles(self.config.num_rotation_angles)

    def solve_single_with_restarts(
        self,
        n: int,
        prev_solution: List[Tuple[float, float, float]],
        time_budget: float,
        verbose: bool = False
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Solve for n trees with multi-restart strategy.
        """
        if n <= 0:
            return [], 0.0

        if n == 1:
            return [(0.0, 0.0, 0.0)], compute_bounding_square_side([(0.0, 0.0, 0.0)])

        start_time = time.time()
        time_per_restart = time_budget / max(1, self.config.num_restarts)

        prev_polys = [make_tree_polygon(x, y, d) for x, y, d in prev_solution]
        tree_index = STRtree(prev_polys) if prev_polys else None

        best_solution = None
        best_score = float('inf')

        # Try different rotation angles for initial placement
        angles_to_try = self.rotation_angles[:min(len(self.rotation_angles), 12)]

        for restart in range(self.config.num_restarts):
            if time.time() - start_time >= time_budget * 0.95:
                break

            # Vary the random seed for each restart
            random.seed(self.seed + restart * 1000 + n)

            # Try different rotation for new tree
            angle_idx = restart % len(angles_to_try)
            new_angle = angles_to_try[angle_idx] + random.uniform(-2.5, 2.5)

            new_base = affinity.rotate(BASE_POLYGON, new_angle, origin=(0, 0))

            # Place using NFP or radial
            if restart < self.config.num_restarts // 2:
                best_x, best_y = place_tree_greedy_nfp(new_base, prev_polys, self.config)
            else:
                best_x, best_y = place_tree_greedy_radial(new_base, prev_polys, self.config)

            # Build initial solution
            new_placement = (best_x, best_y, new_angle)
            solution = prev_solution + [new_placement]

            # Perturb for restarts > 0
            if restart > 0 and best_solution is not None:
                # Start from best and perturb
                solution = list(best_solution)
                num_to_perturb = max(1, int(n * 0.2))
                indices = random.sample(range(n), num_to_perturb)

                for idx in indices:
                    x, y, d = solution[idx]
                    solution[idx] = (
                        x + random.gauss(0, 0.05 * best_score),
                        y + random.gauss(0, 0.05 * best_score),
                        (d + random.uniform(-15, 15)) % 360
                    )

            # Run SA
            remaining_time = min(
                time_per_restart * 0.8,
                time_budget - (time.time() - start_time)
            )

            if remaining_time > 0.5:
                optimized, score = simulated_annealing_advanced(
                    solution, self.config, remaining_time, best_score
                )

                # Local search refinement
                if remaining_time > 1.0:
                    optimized = local_search_fine(
                        optimized,
                        self.config.local_search_precision,
                        self.config.local_search_iterations // 2
                    )
                    score = compute_bounding_square_side(optimized)

                if score < best_score:
                    # Validate
                    overlaps = check_all_overlaps(optimized)
                    if not overlaps:
                        best_score = score
                        best_solution = optimized

        # Final local search
        if best_solution is not None and time.time() - start_time < time_budget * 0.98:
            best_solution = local_search_fine(
                best_solution,
                self.config.local_search_precision,
                self.config.local_search_iterations
            )
            best_score = compute_bounding_square_side(best_solution)

        if best_solution is None:
            # Fallback
            new_angle = random.uniform(0, 360)
            new_base = affinity.rotate(BASE_POLYGON, new_angle, origin=(0, 0))
            best_x, best_y = place_tree_greedy_radial(new_base, prev_polys, self.config)
            best_solution = prev_solution + [(best_x, best_y, new_angle)]
            best_score = compute_bounding_square_side(best_solution)

        return center_placements(best_solution), best_score

    def solve_all(self, max_n: int = 200, verbose: bool = True) -> Dict[int, List[Tuple[float, float, float]]]:
        """Solve for all n from 1 to max_n."""

        total_start = time.time()

        if verbose:
            print("=" * 70)
            print("SANTA 2025 - EXTREME OPTIMIZATION SOLVER")
            print("=" * 70)
            print(f"Restarts per puzzle: {self.config.num_restarts}")
            print(f"Time per puzzle: {self.config.time_per_puzzle:.0f}s")
            print(f"Rotation angles: {self.config.num_rotation_angles}")
            print(f"NFP enabled: {self.config.use_nfp and HAS_PYCLIPPER}")
            print(f"Placement attempts: {self.config.num_placement_attempts}")
            print()

        for n in range(1, max_n + 1):
            n_start = time.time()

            # Get previous solution
            if n == 1:
                prev_solution = []
            else:
                prev_solution = self.solutions[n - 1]

            # Adaptive time budget - more time for larger n (harder)
            time_factor = 1.0 + (n / max_n) * 0.5
            time_budget = self.config.time_per_puzzle * time_factor

            # Remaining time check
            elapsed = time.time() - total_start
            remaining = self.config.total_time_limit - elapsed
            time_budget = min(time_budget, remaining / (max_n - n + 1))

            solution, score = self.solve_single_with_restarts(
                n, prev_solution, time_budget, verbose
            )

            self.solutions[n] = solution
            self.scores[n] = score

            elapsed_n = time.time() - n_start

            if verbose and (n <= 10 or n % 10 == 0 or n == max_n):
                total_score = self.compute_total_score()
                print(f"n={n:3d}: side={score:.4f}, time={elapsed_n:.1f}s, total={total_score:.2f}")

        if verbose:
            total_time = time.time() - total_start
            total_score = self.compute_total_score()
            print()
            print("=" * 70)
            print(f"COMPLETE - Time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"Final Score: {total_score:.4f}")
            print("=" * 70)

        return self.solutions

    def compute_total_score(self) -> float:
        """Compute competition score: sum of (side^2 / n)."""
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, compute_bounding_square_side(sol))
            total += (side ** 2) / n
        return total


# =============================================================================
# VALIDATION
# =============================================================================

def validate_solution(
    solution: List[Tuple[float, float, float]],
    n: int
) -> Tuple[bool, str]:
    """Validate a single solution."""
    if len(solution) != n:
        return False, f"Expected {n} trees, got {len(solution)}"

    if n == 0:
        return True, "OK"

    overlaps = check_all_overlaps(solution)
    if overlaps:
        return False, f"{len(overlaps)} overlapping pair(s)"

    return True, "OK"


def validate_all_solutions(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200,
    verbose: bool = True
) -> bool:
    """Validate all solutions."""
    all_valid = True
    invalid = []

    for n in range(1, max_n + 1):
        if n not in solutions:
            all_valid = False
            invalid.append((n, "Missing"))
            continue

        valid, msg = validate_solution(solutions[n], n)
        if not valid:
            all_valid = False
            invalid.append((n, msg))

    if verbose:
        if all_valid:
            print(f"All {max_n} solutions are valid")
        else:
            print(f"{len(invalid)} invalid solution(s)")
            for n, msg in invalid[:10]:
                print(f"  n={n}: {msg}")

    return all_valid


# =============================================================================
# I/O
# =============================================================================

def create_submission(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    output_path: str = "submission.csv",
    decimals: int = 6
) -> str:
    """Create submission CSV matching official format."""
    with open(output_path, "w") as f:
        f.write("id,x,y,deg\n")

        for n in range(1, 201):
            if n not in solutions:
                raise ValueError(f"Missing solution for n={n}")

            positions = solutions[n]
            if len(positions) != n:
                raise ValueError(f"Wrong count for n={n}")

            # Get bounds to shift coordinates
            polys = [make_tree_polygon(x, y, d) for x, y, d in positions]
            bounds = unary_union(polys).bounds
            min_x, min_y = bounds[0], bounds[1]

            for idx, (x, y, deg) in enumerate(positions):
                X = x - min_x
                Y = y - min_y
                f.write(f"{n:03d}_{idx},s{X:.{decimals}f},s{Y:.{decimals}f},s{deg:.{decimals}f}\n")

    return output_path


def print_score_summary(
    solutions: Dict[int, List[Tuple[float, float, float]]],
    max_n: int = 200
):
    """Print detailed score summary."""
    print("=" * 60)
    print("SCORE SUMMARY")
    print("=" * 60)

    sides = {}
    contributions = {}

    for n in range(1, max_n + 1):
        if n not in solutions:
            continue
        side = compute_bounding_square_side(solutions[n])
        sides[n] = side
        contributions[n] = (side ** 2) / n

    total = sum(contributions.values())
    baseline = 157.08

    print(f"Total score: {total:.4f}")
    print(f"Baseline: {baseline}")
    print(f"Improvement: {(baseline - total) / baseline * 100:.1f}%")
    print()

    # Worst contributors
    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 worst score contributions:")
    for n, contrib in sorted_contrib[:10]:
        print(f"  n={n:3d}: side={sides[n]:.4f}, contrib={contrib:.4f}")

    print()
    print(f"Side length range: {min(sides.values()):.4f} - {max(sides.values()):.4f}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Santa 2025 Extreme Optimization Solver")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-n", type=int, default=200, help="Maximum n to solve")
    parser.add_argument("--time-per-puzzle", type=float, default=60.0, help="Seconds per puzzle")
    parser.add_argument("--restarts", type=int, default=25, help="Number of restarts per puzzle")
    parser.add_argument("--quick", action="store_true", help="Quick mode (less time)")
    args = parser.parse_args()

    # Configure
    config = ExtremeConfig()
    config.seed = args.seed

    if args.quick:
        config.time_per_puzzle = 10.0
        config.num_restarts = 5
        config.num_placement_attempts = 50
        config.total_time_limit = 3000.0
    else:
        config.time_per_puzzle = args.time_per_puzzle
        config.num_restarts = args.restarts

    print("Configuration:")
    print(f"  Time per puzzle: {config.time_per_puzzle}s")
    print(f"  Restarts: {config.num_restarts}")
    print(f"  Rotation angles: {config.num_rotation_angles}")
    print(f"  Placement attempts: {config.num_placement_attempts}")
    print(f"  NFP: {config.use_nfp and HAS_PYCLIPPER}")
    print()

    # Solve
    solver = ExtremeSolver(config=config, seed=args.seed)
    solutions = solver.solve_all(max_n=args.max_n, verbose=True)

    # Validate
    print("\nValidating...")
    validate_all_solutions(solutions, max_n=args.max_n, verbose=True)

    # Score summary
    print()
    print_score_summary(solutions, max_n=args.max_n)

    # Create submission
    print(f"\nCreating submission: {args.output}")
    create_submission(solutions, args.output)
    print(f"Saved: {args.output}")

    total_score = compute_score(solutions)
    print(f"\nFinal Score: {total_score:.4f}")

    return solutions


if __name__ == "__main__":
    main()
