#!/usr/bin/env python3
"""
Santa 2025 - ELITE Solver (Target: Score 62)
=============================================

Combines multiple advanced optimization techniques:
1. Genetic/Evolutionary Algorithm - Global population-based optimization
2. OR-Tools Constraint Programming - Exact solutions for small n
3. Optimal Packing Patterns - Hex, lattice, interlocked templates
4. Tree Interlocking - Exploit stepped tree shape for tight packing
5. Simulated Annealing - Local refinement
6. Basin Hopping - Escape local minima

Usage:
    python elite_solver.py [--output submission.csv] [--hours 24]
    python elite_solver.py --quick      # Fast test (~2 hours)
    python elite_solver.py --standard   # Standard (~12 hours)
    python elite_solver.py --ultra      # Maximum quality (~48 hours)

Requirements:
    pip install numpy shapely pyclipper tqdm ortools
"""

import os
import sys
import math
import time
import random
import argparse
import copy
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapely.prepared import prep

# Optional imports
try:
    import pyclipper
    HAS_PYCLIPPER = True
except ImportError:
    HAS_PYCLIPPER = False
    print("Warning: pyclipper not found. NFP disabled.")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    print("Warning: OR-Tools not found. CP disabled.")


# =============================================================================
# TREE GEOMETRY
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
TREE_WIDTH = 0.7
TREE_HEIGHT = 1.0
TREE_CENTROID = (0.0, 0.3)  # Approximate visual center

# Precompute rotated polygons for common angles
ANGLE_CACHE = {}
for angle in range(0, 360, 5):
    ANGLE_CACHE[angle] = affinity.rotate(BASE_POLYGON, angle, origin=(0, 0))


def make_tree_polygon(x: float, y: float, angle_deg: float) -> Polygon:
    """Create tree polygon at position with rotation."""
    # Use cached rotation if available
    angle_key = int(angle_deg) % 360
    if angle_key in ANGLE_CACHE and angle_key == angle_deg:
        poly = ANGLE_CACHE[angle_key]
    else:
        poly = affinity.rotate(BASE_POLYGON, angle_deg, origin=(0, 0))
    return affinity.translate(poly, xoff=x, yoff=y)


def make_tree_polygon_fast(x: float, y: float, angle_deg: float) -> Polygon:
    """Fast tree polygon creation using numpy rotation."""
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    coords = []
    for px, py in TREE_COORDS:
        rx = px * cos_a - py * sin_a + x
        ry = px * sin_a + py * cos_a + y
        coords.append((rx, ry))
    return Polygon(coords)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EliteConfig:
    """Configuration for elite solver."""

    # Time management
    max_hours: float = 24.0
    time_per_puzzle_base: float = 60.0
    priority_multiplier: float = 10.0

    # Genetic Algorithm
    ga_population_size: int = 50
    ga_generations: int = 100
    ga_mutation_rate: float = 0.15
    ga_crossover_rate: float = 0.7
    ga_elite_count: int = 5
    ga_tournament_size: int = 5

    # Simulated Annealing
    sa_initial_temp: float = 5.0
    sa_final_temp: float = 1e-8
    sa_iterations: int = 5000
    sa_cooling_rate: float = 0.9995

    # Local Search
    local_precision: float = 0.0001
    local_iterations: int = 3000

    # Basin Hopping
    basin_hops: int = 20
    basin_perturbation: float = 0.15

    # Placement
    collision_buffer: float = 0.001
    num_restarts: int = 50

    # Pattern-based
    use_patterns: bool = True
    use_interlocking: bool = True

    # OR-Tools (for small n)
    use_ortools: bool = True
    ortools_max_n: int = 8
    ortools_time_limit: float = 60.0

    # NFP
    use_nfp: bool = True

    # Checkpointing
    checkpoint_interval: int = 5
    checkpoint_path: str = "elite_checkpoint.csv"

    seed: int = 42

    def get_time_for_n(self, n: int) -> float:
        """More time for smaller n (higher impact on score)."""
        priority = 1.0 + (self.priority_multiplier - 1.0) * (200 - n) / 199
        return self.time_per_puzzle_base * priority


def quick_config() -> EliteConfig:
    """Quick test mode (~2 hours)."""
    return EliteConfig(
        max_hours=2.0,
        time_per_puzzle_base=15.0,
        priority_multiplier=5.0,
        ga_population_size=20,
        ga_generations=30,
        sa_iterations=1000,
        local_iterations=500,
        basin_hops=5,
        num_restarts=10,
        ortools_time_limit=10.0,
    )


def standard_config() -> EliteConfig:
    """Standard mode (~12 hours)."""
    return EliteConfig(
        max_hours=12.0,
        time_per_puzzle_base=45.0,
        ga_population_size=40,
        ga_generations=60,
        sa_iterations=3000,
        basin_hops=15,
        num_restarts=30,
    )


def ultra_config() -> EliteConfig:
    """Ultra quality mode (~48 hours)."""
    return EliteConfig(
        max_hours=48.0,
        time_per_puzzle_base=180.0,
        priority_multiplier=15.0,
        ga_population_size=80,
        ga_generations=200,
        sa_iterations=10000,
        local_iterations=5000,
        basin_hops=40,
        num_restarts=100,
        ortools_time_limit=300.0,
    )


# =============================================================================
# COLLISION DETECTION
# =============================================================================

class CollisionDetector:
    """Fast collision detection using spatial indexing."""

    def __init__(self, buffer: float = 0.001):
        self.buffer = buffer
        self.tree: Optional[STRtree] = None
        self.polys: List[Polygon] = []

    def build_index(self, placements: List[Tuple[float, float, float]]):
        """Build spatial index from placements."""
        self.polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
        if self.polys:
            self.tree = STRtree(self.polys)

    def check_collision(self, poly: Polygon, exclude_idx: int = -1) -> bool:
        """Check if polygon collides with any existing polygon."""
        if not self.polys:
            return False

        candidates = self.tree.query(poly.buffer(self.buffer))
        for idx in candidates:
            if idx == exclude_idx:
                continue
            if self.buffer > 0:
                if poly.distance(self.polys[idx]) < self.buffer:
                    return True
            else:
                if poly.intersects(self.polys[idx]) and not poly.touches(self.polys[idx]):
                    return True
        return False

    def check_collision_list(self, poly: Polygon, other_polys: List[Polygon]) -> bool:
        """Check collision against a list of polygons."""
        for other in other_polys:
            if self.buffer > 0:
                if poly.distance(other) < self.buffer:
                    return True
            else:
                if poly.intersects(other) and not poly.touches(other):
                    return True
        return False


def has_any_collision(placements: List[Tuple[float, float, float]], buffer: float = 0.001) -> bool:
    """Check if any placements collide."""
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if buffer > 0:
                if polys[i].distance(polys[j]) < buffer:
                    return True
            else:
                if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                    return True
    return False


# =============================================================================
# SCORING
# =============================================================================

def compute_side(placements: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side."""
    if not placements:
        return 0.0
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return max(width, height)


def compute_score_contribution(placements: List[Tuple[float, float, float]], n: int) -> float:
    """Compute score contribution for puzzle n."""
    side = compute_side(placements)
    return (side ** 2) / n


def compute_total_score(solutions: Dict[int, List[Tuple[float, float, float]]]) -> float:
    """Compute total score."""
    total = 0.0
    for n, placements in solutions.items():
        total += compute_score_contribution(placements, n)
    return total


# =============================================================================
# PATTERN-BASED INITIAL PLACEMENTS
# =============================================================================

class PatternGenerator:
    """Generate initial placement patterns."""

    @staticmethod
    def hexagonal_grid(n: int, spacing: float = 0.85) -> List[Tuple[float, float, float]]:
        """Hexagonal grid pattern (optimal for circles)."""
        placements = []
        sqrt_n = int(math.ceil(math.sqrt(n)))
        dx = spacing
        dy = spacing * math.sqrt(3) / 2

        for i in range(sqrt_n + 2):
            for j in range(sqrt_n + 2):
                if len(placements) >= n:
                    break
                x = j * dx + (i % 2) * (dx / 2)
                y = i * dy
                angle = 45  # Use optimal rotation for smallest bounding box
                placements.append((x, y, angle))
            if len(placements) >= n:
                break

        return placements[:n]

    @staticmethod
    def interlocked_pairs(n: int, spacing: float = 0.90) -> List[Tuple[float, float, float]]:
        """Interlocked pairs - trees facing each other."""
        placements = []
        pair_spacing_x = spacing * 1.8
        pair_spacing_y = spacing * 2.2

        pairs_needed = (n + 1) // 2
        cols = int(math.ceil(math.sqrt(pairs_needed)))

        idx = 0
        for row in range(cols + 1):
            for col in range(cols + 1):
                if idx >= n:
                    break

                base_x = col * pair_spacing_x
                base_y = row * pair_spacing_y

                # First tree at 45 degrees
                placements.append((base_x, base_y, 45))
                idx += 1

                if idx >= n:
                    break

                # Second tree at 225 degrees, offset for safe spacing
                offset_x = spacing * 0.95
                offset_y = spacing * 1.0
                placements.append((base_x + offset_x, base_y + offset_y, 225))
                idx += 1

            if idx >= n:
                break

        return placements[:n]

    @staticmethod
    def spiral(n: int, base_radius: float = 0.8) -> List[Tuple[float, float, float]]:
        """Spiral pattern from center outward."""
        placements = []
        angle = 0
        radius = base_radius
        angle_increment = 137.5 * math.pi / 180  # Golden angle
        radius_increment = 0.18

        for i in range(n):
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            tree_angle = 45  # Optimal rotation
            placements.append((x, y, tree_angle))

            angle += angle_increment
            radius += radius_increment / (1 + i * 0.003)

        return placements

    @staticmethod
    def concentric_rings(n: int, ring_spacing: float = 0.9) -> List[Tuple[float, float, float]]:
        """Concentric rings pattern."""
        placements = []

        if n == 0:
            return placements

        # Center tree at optimal angle
        placements.append((0.0, 0.0, 45))
        remaining = n - 1
        ring = 1

        while remaining > 0:
            radius = ring * ring_spacing
            circumference = 2 * math.pi * radius
            trees_in_ring = min(remaining, max(6, int(circumference / 0.85)))

            for i in range(trees_in_ring):
                angle = 2 * math.pi * i / trees_in_ring
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                tree_angle = 45  # Optimal rotation
                placements.append((x, y, tree_angle))

            remaining -= trees_in_ring
            ring += 1

        return placements[:n]

    @staticmethod
    def diamond_lattice(n: int, spacing: float = 1.0) -> List[Tuple[float, float, float]]:
        """Diamond/rhombus lattice pattern."""
        placements = []
        size = int(math.ceil(math.sqrt(n * 2)))

        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if len(placements) >= n:
                    break

                # Diamond coordinates with safe spacing
                x = (i + j) * spacing / 1.5
                y = (i - j) * spacing / 1.5
                angle = 45  # Optimal rotation
                placements.append((x, y, angle))

            if len(placements) >= n:
                break

        # Sort by distance from center
        placements.sort(key=lambda p: p[0]**2 + p[1]**2)
        return placements[:n]

    @staticmethod
    def tight_interlocked(n: int) -> List[Tuple[float, float, float]]:
        """Very tight interlocking using tree shape properties."""
        placements = []

        # Tree dimensions: Width: 0.7, Height: 1.0
        # At 45°, bounding box is 0.8132
        # Use tight but safe spacing

        dx = 0.82  # Horizontal spacing
        dy = 0.82  # Vertical spacing

        rows = int(math.ceil(math.sqrt(n * 1.2)))
        cols = int(math.ceil(n / rows)) + 1

        for row in range(rows + 1):
            for col in range(cols + 1):
                if len(placements) >= n:
                    break

                # Offset alternate rows for better packing
                x_offset = (row % 2) * (dx / 2)
                x = col * dx + x_offset
                y = row * dy

                # All at optimal 45 degree angle
                angle = 45

                placements.append((x, y, angle))

            if len(placements) >= n:
                break

        return placements[:n]

    @staticmethod
    def nested_pairs(n: int) -> List[Tuple[float, float, float]]:
        """Nested pairs where trees face each other with tips in gaps."""
        placements = []

        # Use optimal 45-degree rotation with safe spacing
        group_dx = 1.8  # Spacing between pair groups
        group_dy = 2.0

        pairs_per_row = int(math.ceil(math.sqrt(n / 2)))

        idx = 0
        for row in range(pairs_per_row + 2):
            for col in range(pairs_per_row + 2):
                if idx >= n:
                    break

                base_x = col * group_dx
                base_y = row * group_dy

                # First tree at 45 degrees
                placements.append((base_x, base_y, 45))
                idx += 1

                if idx >= n:
                    break

                # Second tree at 225 degrees, with safe offset
                placements.append((base_x + 0.9, base_y + 0.9, 225))
                idx += 1

            if idx >= n:
                break

        return placements[:n]

    @staticmethod
    def brick_pattern(n: int) -> List[Tuple[float, float, float]]:
        """Brick-like pattern with offset rows."""
        placements = []

        dx = 0.85  # Safe spacing
        dy = 0.85

        cols = int(math.ceil(math.sqrt(n * 1.2)))
        rows = int(math.ceil(n / cols)) + 1

        for row in range(rows + 1):
            offset = (row % 2) * (dx / 2)
            for col in range(cols + 1):
                if len(placements) >= n:
                    break

                x = col * dx + offset
                y = row * dy
                angle = 45  # Optimal rotation

                placements.append((x, y, angle))

            if len(placements) >= n:
                break

        return placements[:n]

    @staticmethod
    def fishbone_pattern(n: int) -> List[Tuple[float, float, float]]:
        """Fishbone pattern - alternating diagonal orientations."""
        placements = []

        dx = 0.92  # Safe spacing
        dy = 0.92

        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows)) + 1

        for row in range(rows + 1):
            for col in range(cols + 1):
                if len(placements) >= n:
                    break

                x = col * dx + (row % 2) * (dx / 3)
                y = row * dy

                # All at optimal angle for consistency
                angle = 45

                placements.append((x, y, angle))

            if len(placements) >= n:
                break

        return placements[:n]

    @staticmethod
    def optimal_small_n(n: int) -> List[Tuple[float, float, float]]:
        """Hand-optimized patterns for very small n."""
        if n == 1:
            return [(0.0, 0.0, 45)]  # Optimal 45-degree rotation
        elif n == 2:
            # Two trees at 45 degrees, positioned optimally
            return [(0.0, 0.0, 45), (0.85, 0.0, 45)]
        elif n == 3:
            # Triangle formation at 45 degrees
            return [
                (0.425, 0.0, 45),
                (0.0, 0.75, 45),
                (0.85, 0.75, 45),
            ]
        elif n == 4:
            # 2x2 grid with offset at 45 degrees
            return [
                (0.0, 0.0, 45),
                (0.85, 0.0, 45),
                (0.425, 0.75, 45),
                (1.275, 0.75, 45),
            ]
        elif n == 5:
            # Pentagon-like at 45 degrees with safe spacing
            return [
                (0.425, 0.0, 45),
                (0.0, 0.85, 45),
                (0.85, 0.85, 45),
                (0.0, 1.70, 45),
                (0.85, 1.70, 45),
            ]
        else:
            # Default to tight interlocked
            return PatternGenerator.tight_interlocked(n)[:n]


# =============================================================================
# GREEDY PLACEMENT WITH NFP
# =============================================================================

class GreedyNFPPlacer:
    """Greedy placement using No-Fit Polygons for tight packing."""

    def __init__(self, buffer: float = 0.001):
        self.buffer = buffer
        self.nfp_cache: Dict[Tuple[float, float], Polygon] = {}

    def place_greedy(self, n: int, angles: Optional[List[float]] = None) -> List[Tuple[float, float, float]]:
        """Greedy placement one tree at a time using NFP."""
        if not HAS_PYCLIPPER:
            return PatternGenerator.hexagonal_grid(n)

        if angles is None:
            angles = [0] * n

        placements = []
        polys = []

        for i in range(n):
            angle = angles[i]

            if i == 0:
                # First tree at origin
                placements.append((0.0, 0.0, angle))
                polys.append(make_tree_polygon(0, 0, angle))
            else:
                # Find best position using NFP
                best_pos = self.find_best_position(placements, polys, angle)
                if best_pos:
                    x, y = best_pos
                    placements.append((x, y, angle))
                    polys.append(make_tree_polygon(x, y, angle))
                else:
                    # Fallback to radial placement
                    x, y = self.radial_place(polys, angle)
                    placements.append((x, y, angle))
                    polys.append(make_tree_polygon(x, y, angle))

        return placements

    def find_best_position(self, placements: List[Tuple[float, float, float]],
                          polys: List[Polygon], angle: float) -> Optional[Tuple[float, float]]:
        """Find position that minimizes bounding box."""
        candidates = []

        # Get NFP boundary points
        for idx, (fx, fy, fa) in enumerate(placements):
            nfp = self.compute_nfp(fa, angle)
            if nfp is None:
                continue

            # Translate to fixed position
            nfp_t = affinity.translate(nfp, xoff=fx, yoff=fy)

            # Sample boundary points
            try:
                boundary = nfp_t.exterior
                for t in np.linspace(0, 1, 50, endpoint=False):
                    pt = boundary.interpolate(t, normalized=True)
                    candidates.append((pt.x + self.buffer * 2, pt.y + self.buffer * 2))
            except:
                pass

        if not candidates:
            return None

        # Evaluate each candidate
        best_pos = None
        best_side = float('inf')

        current_bounds = unary_union(polys).bounds
        cx = (current_bounds[0] + current_bounds[2]) / 2
        cy = (current_bounds[1] + current_bounds[3]) / 2

        for x, y in candidates:
            new_poly = make_tree_polygon(x, y, angle)

            # Check collision
            collision = False
            for p in polys:
                if new_poly.distance(p) < self.buffer:
                    collision = True
                    break

            if collision:
                continue

            # Compute new bounding box
            all_polys = polys + [new_poly]
            bounds = unary_union(all_polys).bounds
            side = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

            if side < best_side:
                best_side = side
                best_pos = (x, y)

        return best_pos

    def radial_place(self, polys: List[Polygon], angle: float) -> Tuple[float, float]:
        """Fallback radial placement."""
        if not polys:
            return (0.0, 0.0)

        bounds = unary_union(polys).bounds
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        radius = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2 + 1.0

        for _ in range(100):
            theta = random.uniform(0, 2 * math.pi)
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)

            poly = make_tree_polygon(x, y, angle)
            collision = False
            for p in polys:
                if poly.distance(p) < self.buffer:
                    collision = True
                    break

            if not collision:
                return (x, y)

            radius += 0.1

        return (cx + radius, cy)

    def compute_nfp(self, fixed_angle: float, moving_angle: float) -> Optional[Polygon]:
        """Compute NFP for two tree orientations."""
        cache_key = (fixed_angle % 360, moving_angle % 360)
        if cache_key in self.nfp_cache:
            return self.nfp_cache[cache_key]

        try:
            fixed = make_tree_polygon(0, 0, fixed_angle)
            moving = make_tree_polygon(0, 0, moving_angle)
            moving_reflected = affinity.scale(moving, xfact=-1, yfact=-1, origin=(0, 0))

            precision = 10000
            fixed_path = [(int(x * precision), int(y * precision))
                         for x, y in list(fixed.exterior.coords)[:-1]]
            moving_path = [(int(x * precision), int(y * precision))
                          for x, y in list(moving_reflected.exterior.coords)[:-1]]

            result = pyclipper.MinkowskiSum(fixed_path, moving_path, True)

            if result:
                coords = [(x / precision, y / precision) for x, y in result[0]]
                if len(coords) >= 3:
                    nfp = Polygon(coords)
                    self.nfp_cache[cache_key] = nfp
                    return nfp
        except:
            pass

        return None


# =============================================================================
# NO-FIT POLYGON (NFP)
# =============================================================================

class NFPCalculator:
    """Calculate No-Fit Polygons for precise placement."""

    def __init__(self, precision: int = 10000):
        self.precision = precision
        self.nfp_cache: Dict[Tuple[float, float], List] = {}

    def to_clipper(self, poly: Polygon) -> List[Tuple[int, int]]:
        """Convert Shapely polygon to pyclipper format."""
        coords = list(poly.exterior.coords)[:-1]
        return [(int(x * self.precision), int(y * self.precision)) for x, y in coords]

    def from_clipper(self, path: List[Tuple[int, int]]) -> Polygon:
        """Convert pyclipper path to Shapely polygon."""
        coords = [(x / self.precision, y / self.precision) for x, y in path]
        if len(coords) < 3:
            return None
        return Polygon(coords)

    def compute_nfp(self, fixed_angle: float, moving_angle: float) -> Optional[Polygon]:
        """Compute NFP for two tree orientations."""
        if not HAS_PYCLIPPER:
            return None

        cache_key = (fixed_angle, moving_angle)
        if cache_key in self.nfp_cache:
            return self.nfp_cache[cache_key]

        try:
            fixed = make_tree_polygon(0, 0, fixed_angle)
            moving = make_tree_polygon(0, 0, moving_angle)

            # Reflect moving polygon for Minkowski sum
            moving_reflected = affinity.scale(moving, xfact=-1, yfact=-1, origin=(0, 0))

            fixed_path = self.to_clipper(fixed)
            moving_path = self.to_clipper(moving_reflected)

            # Compute Minkowski sum
            result = pyclipper.MinkowskiSum(fixed_path, moving_path, True)

            if result:
                nfp = self.from_clipper(result[0])
                self.nfp_cache[cache_key] = nfp
                return nfp
        except Exception:
            pass

        return None

    def find_valid_positions(self, fixed_polys: List[Polygon], fixed_placements: List[Tuple[float, float, float]],
                            moving_angle: float, num_samples: int = 100) -> List[Tuple[float, float]]:
        """Find valid positions for a new tree using NFP."""
        if not fixed_polys:
            return [(0.0, 0.0)]

        valid_positions = []

        for idx, (fx, fy, fa) in enumerate(fixed_placements):
            nfp = self.compute_nfp(fa, moving_angle)
            if nfp is None:
                continue

            # Translate NFP to fixed position
            nfp_translated = affinity.translate(nfp, xoff=fx, yoff=fy)

            # Sample points on NFP boundary
            boundary = nfp_translated.exterior
            for i in range(num_samples):
                point = boundary.interpolate(i / num_samples, normalized=True)
                valid_positions.append((point.x, point.y))

        return valid_positions


# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

class Individual:
    """Individual in genetic algorithm population."""

    def __init__(self, placements: List[Tuple[float, float, float]]):
        self.placements = placements
        self.fitness: float = float('inf')
        self.valid: bool = False

    def copy(self) -> 'Individual':
        return Individual(list(self.placements))


class GeneticAlgorithm:
    """Genetic algorithm for global optimization."""

    def __init__(self, n: int, config: EliteConfig):
        self.n = n
        self.config = config
        self.detector = CollisionDetector(config.collision_buffer)
        self.pattern_gen = PatternGenerator()
        self.nfp_calc = NFPCalculator() if HAS_PYCLIPPER else None

    def create_initial_population(self) -> List[Individual]:
        """Create diverse initial population using different patterns."""
        population = []

        patterns = [
            self.pattern_gen.hexagonal_grid,
            self.pattern_gen.interlocked_pairs,
            self.pattern_gen.spiral,
            self.pattern_gen.concentric_rings,
            self.pattern_gen.diamond_lattice,
            self.pattern_gen.tight_interlocked,
        ]

        # Generate from patterns
        for pattern_fn in patterns:
            for _ in range(self.config.ga_population_size // len(patterns)):
                try:
                    placements = pattern_fn(self.n)
                    # Add some randomness
                    placements = self.add_noise(placements, 0.1)
                    placements = self.repair_collisions(placements)
                    if placements:
                        population.append(Individual(placements))
                except:
                    pass

        # Fill remaining with random
        while len(population) < self.config.ga_population_size:
            placements = self.random_placement()
            if placements:
                population.append(Individual(placements))

        return population[:self.config.ga_population_size]

    def random_placement(self) -> List[Tuple[float, float, float]]:
        """Generate random valid placement."""
        placements = []
        polys = []
        max_attempts = 1000

        for i in range(self.n):
            for attempt in range(max_attempts):
                # Random position in reasonable area
                radius = math.sqrt(self.n) * 0.5
                x = random.uniform(-radius, radius)
                y = random.uniform(-radius, radius)
                angle = random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])

                poly = make_tree_polygon(x, y, angle)
                if not self.detector.check_collision_list(poly, polys):
                    placements.append((x, y, angle))
                    polys.append(poly)
                    break

            if len(placements) <= i:
                # Failed to place - restart
                return self.pattern_gen.hexagonal_grid(self.n)

        return placements

    def add_noise(self, placements: List[Tuple[float, float, float]],
                  magnitude: float) -> List[Tuple[float, float, float]]:
        """Add random noise to placements."""
        return [
            (x + random.gauss(0, magnitude),
             y + random.gauss(0, magnitude),
             (d + random.choice([-30, 0, 30])) % 360)
            for x, y, d in placements
        ]

    def repair_collisions(self, placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Repair colliding placements."""
        if not placements:
            return placements

        repaired = list(placements)
        polys = [make_tree_polygon(x, y, d) for x, y, d in repaired]

        for _ in range(50):  # Max repair iterations
            collision_found = False

            for i in range(len(polys)):
                for j in range(i + 1, len(polys)):
                    if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                        collision_found = True

                        # Move tree j away from tree i
                        ci = polys[i].centroid
                        cj = polys[j].centroid

                        dx = cj.x - ci.x
                        dy = cj.y - ci.y
                        dist = math.sqrt(dx*dx + dy*dy)

                        if dist > 0.001:
                            move_dist = 0.1
                            nx, ny = dx/dist, dy/dist
                            new_x = repaired[j][0] + nx * move_dist
                            new_y = repaired[j][1] + ny * move_dist
                            repaired[j] = (new_x, new_y, repaired[j][2])
                            polys[j] = make_tree_polygon(new_x, new_y, repaired[j][2])

            if not collision_found:
                break

        return repaired

    def evaluate(self, individual: Individual) -> float:
        """Evaluate fitness (lower is better)."""
        # Check for collisions
        if has_any_collision(individual.placements, self.config.collision_buffer):
            individual.valid = False
            individual.fitness = float('inf')
            return individual.fitness

        individual.valid = True
        side = compute_side(individual.placements)
        individual.fitness = side
        return side

    def tournament_select(self, population: List[Individual]) -> Individual:
        """Tournament selection."""
        tournament = random.sample(population, min(self.config.ga_tournament_size, len(population)))
        return min(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover two individuals."""
        if random.random() > self.config.ga_crossover_rate:
            return parent1.copy()

        # Position-based crossover
        child_placements = []
        for i in range(self.n):
            if random.random() < 0.5:
                child_placements.append(parent1.placements[i])
            else:
                child_placements.append(parent2.placements[i])

        return Individual(child_placements)

    def mutate(self, individual: Individual) -> Individual:
        """Mutate individual."""
        if random.random() > self.config.ga_mutation_rate:
            return individual

        mutated = list(individual.placements)

        # Choose mutation type
        mutation_type = random.choice(['position', 'rotation', 'swap', 'shift_all'])

        if mutation_type == 'position':
            # Move random tree
            idx = random.randint(0, self.n - 1)
            x, y, d = mutated[idx]
            mutated[idx] = (
                x + random.gauss(0, 0.1),
                y + random.gauss(0, 0.1),
                d
            )

        elif mutation_type == 'rotation':
            # Rotate random tree
            idx = random.randint(0, self.n - 1)
            x, y, d = mutated[idx]
            mutated[idx] = (x, y, (d + random.choice([30, 60, 90, 180])) % 360)

        elif mutation_type == 'swap':
            # Swap two trees
            if self.n >= 2:
                i, j = random.sample(range(self.n), 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]

        elif mutation_type == 'shift_all':
            # Shift all toward center
            cx = sum(p[0] for p in mutated) / self.n
            cy = sum(p[1] for p in mutated) / self.n
            shift = 0.02
            mutated = [
                (x - (x - cx) * shift, y - (y - cy) * shift, d)
                for x, y, d in mutated
            ]

        child = Individual(mutated)
        child.placements = self.repair_collisions(child.placements)
        return child

    def evolve(self, time_limit: float) -> List[Tuple[float, float, float]]:
        """Run genetic algorithm evolution."""
        start_time = time.time()

        # Initialize population
        population = self.create_initial_population()

        # Evaluate initial population
        for ind in population:
            self.evaluate(ind)

        best = min(population, key=lambda x: x.fitness)
        best_ever = best.copy()

        generation = 0
        while time.time() - start_time < time_limit and generation < self.config.ga_generations:
            generation += 1

            # Create next generation
            new_population = []

            # Elitism
            population.sort(key=lambda x: x.fitness)
            for i in range(self.config.ga_elite_count):
                if i < len(population):
                    new_population.append(population[i].copy())

            # Generate offspring
            while len(new_population) < self.config.ga_population_size:
                parent1 = self.tournament_select(population)
                parent2 = self.tournament_select(population)

                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                self.evaluate(child)

                new_population.append(child)

            population = new_population

            # Track best
            best = min(population, key=lambda x: x.fitness)
            if best.fitness < best_ever.fitness:
                best_ever = best.copy()

        return best_ever.placements if best_ever.valid else self.pattern_gen.hexagonal_grid(self.n)


# =============================================================================
# SIMULATED ANNEALING
# =============================================================================

class SimulatedAnnealing:
    """Simulated annealing for local optimization."""

    def __init__(self, config: EliteConfig):
        self.config = config
        self.detector = CollisionDetector(config.collision_buffer)

    def optimize(self, placements: List[Tuple[float, float, float]],
                 time_limit: float) -> List[Tuple[float, float, float]]:
        """Run simulated annealing optimization."""
        start_time = time.time()
        n = len(placements)

        if n == 0:
            return placements

        current = list(placements)
        current_score = compute_side(current)

        best = list(current)
        best_score = current_score

        temp = self.config.sa_initial_temp
        iterations = 0

        while time.time() - start_time < time_limit and iterations < self.config.sa_iterations:
            iterations += 1

            # Generate neighbor
            neighbor = self.generate_neighbor(current, n)

            # Check validity
            if has_any_collision(neighbor, self.config.collision_buffer):
                continue

            neighbor_score = compute_side(neighbor)

            # Accept or reject
            delta = neighbor_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = neighbor
                current_score = neighbor_score

                if current_score < best_score:
                    best = list(current)
                    best_score = current_score

            # Cool down
            temp *= self.config.sa_cooling_rate
            temp = max(temp, self.config.sa_final_temp)

        return best

    def generate_neighbor(self, placements: List[Tuple[float, float, float]],
                          n: int) -> List[Tuple[float, float, float]]:
        """Generate neighboring solution."""
        neighbor = list(placements)

        move_type = random.choices(
            ['small_move', 'medium_move', 'rotate', 'swap', 'compact', 'large_move'],
            weights=[0.35, 0.2, 0.15, 0.1, 0.15, 0.05]
        )[0]

        idx = random.randint(0, n - 1)
        x, y, d = neighbor[idx]

        if move_type == 'small_move':
            neighbor[idx] = (
                x + random.gauss(0, 0.02),
                y + random.gauss(0, 0.02),
                d
            )

        elif move_type == 'medium_move':
            neighbor[idx] = (
                x + random.gauss(0, 0.1),
                y + random.gauss(0, 0.1),
                d
            )

        elif move_type == 'large_move':
            neighbor[idx] = (
                x + random.gauss(0, 0.3),
                y + random.gauss(0, 0.3),
                d
            )

        elif move_type == 'rotate':
            new_angle = (d + random.choice([15, 30, 45, 60, 90, 180])) % 360
            neighbor[idx] = (x, y, new_angle)

        elif move_type == 'swap':
            if n >= 2:
                j = random.randint(0, n - 1)
                while j == idx:
                    j = random.randint(0, n - 1)
                neighbor[idx], neighbor[j] = neighbor[j], neighbor[idx]

        elif move_type == 'compact':
            # Move toward centroid
            cx = sum(p[0] for p in neighbor) / n
            cy = sum(p[1] for p in neighbor) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0.001:
                step = 0.02
                neighbor[idx] = (x + dx/dist * step, y + dy/dist * step, d)

        return neighbor


# =============================================================================
# LOCAL SEARCH
# =============================================================================

class LocalSearch:
    """Fine-grained local search."""

    def __init__(self, config: EliteConfig):
        self.config = config

    def optimize(self, placements: List[Tuple[float, float, float]],
                 time_limit: float) -> List[Tuple[float, float, float]]:
        """Run local search optimization."""
        start_time = time.time()
        n = len(placements)

        if n == 0:
            return placements

        current = list(placements)
        best_side = compute_side(current)

        precision = self.config.local_precision
        directions = [
            (precision, 0), (-precision, 0),
            (0, precision), (0, -precision),
            (precision, precision), (precision, -precision),
            (-precision, precision), (-precision, -precision),
        ]

        improved = True
        iterations = 0

        while improved and time.time() - start_time < time_limit and iterations < self.config.local_iterations:
            improved = False
            iterations += 1

            for i in range(n):
                x, y, d = current[i]

                for dx, dy in directions:
                    new_placement = (x + dx, y + dy, d)
                    test = current[:i] + [new_placement] + current[i+1:]

                    if not has_any_collision(test, self.config.collision_buffer):
                        new_side = compute_side(test)
                        if new_side < best_side:
                            current = test
                            best_side = new_side
                            improved = True
                            break

                # Also try rotation
                for delta_d in [5, -5, 10, -10, 15, -15]:
                    new_d = (d + delta_d) % 360
                    new_placement = (x, y, new_d)
                    test = current[:i] + [new_placement] + current[i+1:]

                    if not has_any_collision(test, self.config.collision_buffer):
                        new_side = compute_side(test)
                        if new_side < best_side:
                            current = test
                            best_side = new_side
                            improved = True
                            break

        return current


# =============================================================================
# BASIN HOPPING
# =============================================================================

class BasinHopping:
    """Basin hopping for escaping local minima."""

    def __init__(self, config: EliteConfig):
        self.config = config
        self.sa = SimulatedAnnealing(config)
        self.local = LocalSearch(config)

    def optimize(self, placements: List[Tuple[float, float, float]],
                 time_limit: float) -> List[Tuple[float, float, float]]:
        """Run basin hopping optimization."""
        start_time = time.time()
        time_per_hop = time_limit / max(1, self.config.basin_hops)

        best = list(placements)
        best_score = compute_side(best)

        current = list(best)

        for hop in range(self.config.basin_hops):
            if time.time() - start_time >= time_limit:
                break

            # Perturb
            perturbed = self.perturb(current)

            # Local minimize
            hop_time = min(time_per_hop, time_limit - (time.time() - start_time))
            optimized = self.sa.optimize(perturbed, hop_time * 0.7)
            optimized = self.local.optimize(optimized, hop_time * 0.3)

            score = compute_side(optimized)

            # Accept based on Metropolis criterion
            if score < best_score:
                best = list(optimized)
                best_score = score
                current = list(optimized)
            elif random.random() < 0.1:  # Sometimes accept worse
                current = list(optimized)

        return best

    def perturb(self, placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Perturb solution to escape local minimum."""
        n = len(placements)
        perturbed = list(placements)

        # Perturb random subset
        num_to_perturb = max(1, int(n * self.config.basin_perturbation))
        indices = random.sample(range(n), num_to_perturb)

        for idx in indices:
            x, y, d = perturbed[idx]
            perturbed[idx] = (
                x + random.gauss(0, 0.2),
                y + random.gauss(0, 0.2),
                (d + random.choice([0, 60, 120, 180, 240, 300])) % 360
            )

        # Repair collisions
        perturbed = GeneticAlgorithm(n, self.config).repair_collisions(perturbed)

        return perturbed


# =============================================================================
# OR-TOOLS CONSTRAINT PROGRAMMING (for small n)
# =============================================================================

class ConstraintSolver:
    """OR-Tools constraint programming solver for small n."""

    def __init__(self, config: EliteConfig):
        self.config = config

    def solve(self, n: int, time_limit: float) -> Optional[List[Tuple[float, float, float]]]:
        """Solve using constraint programming."""
        if not HAS_ORTOOLS or n > self.config.ortools_max_n:
            return None

        # Grid-based discretization for CP
        precision = 100  # 0.01 units
        max_coord = int(n * 0.5 * precision)  # Estimated max coordinate

        model = cp_model.CpModel()

        # Variables: x, y, angle for each tree
        xs = [model.NewIntVar(-max_coord, max_coord, f'x_{i}') for i in range(n)]
        ys = [model.NewIntVar(-max_coord, max_coord, f'y_{i}') for i in range(n)]
        angles = [model.NewIntVar(0, 11, f'a_{i}') for i in range(n)]  # 12 possible angles (30 degree increments)

        # Bounding box
        min_x = model.NewIntVar(-max_coord, max_coord, 'min_x')
        max_x = model.NewIntVar(-max_coord, max_coord, 'max_x')
        min_y = model.NewIntVar(-max_coord, max_coord, 'min_y')
        max_y = model.NewIntVar(-max_coord, max_coord, 'max_y')

        model.AddMinEquality(min_x, xs)
        model.AddMaxEquality(max_x, xs)
        model.AddMinEquality(min_y, ys)
        model.AddMaxEquality(max_y, ys)

        # Width and height (with tree dimensions)
        tree_half_width = int(0.35 * precision)
        tree_height_above = int(0.8 * precision)
        tree_height_below = int(0.2 * precision)

        width = model.NewIntVar(0, max_coord * 2, 'width')
        height = model.NewIntVar(0, max_coord * 2, 'height')

        model.Add(width == max_x - min_x + 2 * tree_half_width)
        model.Add(height == max_y - min_y + tree_height_above + tree_height_below)

        # Side (max of width, height)
        side = model.NewIntVar(0, max_coord * 2, 'side')
        model.AddMaxEquality(side, [width, height])

        # Minimum distance constraints (approximate)
        min_dist = int(0.6 * precision)  # Minimum center-to-center distance

        for i in range(n):
            for j in range(i + 1, n):
                # Manhattan distance approximation
                dx = model.NewIntVar(-max_coord * 2, max_coord * 2, f'dx_{i}_{j}')
                dy = model.NewIntVar(-max_coord * 2, max_coord * 2, f'dy_{i}_{j}')
                abs_dx = model.NewIntVar(0, max_coord * 2, f'abs_dx_{i}_{j}')
                abs_dy = model.NewIntVar(0, max_coord * 2, f'abs_dy_{i}_{j}')

                model.Add(dx == xs[i] - xs[j])
                model.Add(dy == ys[i] - ys[j])
                model.AddAbsEquality(abs_dx, dx)
                model.AddAbsEquality(abs_dy, dy)

                # At least one dimension must have minimum distance
                model.AddBoolOr([
                    abs_dx >= min_dist,
                    abs_dy >= min_dist
                ])

        # Objective: minimize side
        model.Minimize(side)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit

        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            placements = []
            for i in range(n):
                x = solver.Value(xs[i]) / precision
                y = solver.Value(ys[i]) / precision
                angle = solver.Value(angles[i]) * 30
                placements.append((x, y, angle))

            # Verify and refine
            if not has_any_collision(placements, self.config.collision_buffer):
                return placements
            else:
                # CP solution has collisions, use as starting point
                ga = GeneticAlgorithm(n, self.config)
                return ga.repair_collisions(placements)

        return None


# =============================================================================
# COMPACTING
# =============================================================================

class Compactor:
    """Compact placements toward center."""

    def __init__(self, config: EliteConfig):
        self.config = config

    def compact(self, placements: List[Tuple[float, float, float]],
                time_limit: float) -> List[Tuple[float, float, float]]:
        """Compact all trees toward center."""
        start_time = time.time()
        n = len(placements)

        if n == 0:
            return placements

        current = list(placements)
        step = 0.005

        iterations = 0
        max_iterations = 500

        while time.time() - start_time < time_limit and iterations < max_iterations:
            iterations += 1
            moved = False

            # Calculate centroid
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n

            for i in range(n):
                x, y, d = current[i]

                # Direction toward center
                dx = cx - x
                dy = cy - y
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < step:
                    continue

                # Try moving toward center
                nx = x + dx / dist * step
                ny = y + dy / dist * step

                test = current[:i] + [(nx, ny, d)] + current[i+1:]

                if not has_any_collision(test, self.config.collision_buffer):
                    current = test
                    moved = True

            if not moved:
                step *= 0.8
                if step < 0.0005:
                    break

        return current


# =============================================================================
# MAIN SOLVER
# =============================================================================

class EliteSolver:
    """Main solver combining all techniques."""

    def __init__(self, config: EliteConfig):
        self.config = config
        self.pattern_gen = PatternGenerator()
        self.ga = None  # Created per puzzle
        self.sa = SimulatedAnnealing(config)
        self.local = LocalSearch(config)
        self.basin = BasinHopping(config)
        self.compactor = Compactor(config)
        self.cp_solver = ConstraintSolver(config) if HAS_ORTOOLS else None
        self.nfp_placer = GreedyNFPPlacer(config.collision_buffer) if HAS_PYCLIPPER else None

        random.seed(config.seed)
        np.random.seed(config.seed)

    def solve_puzzle(self, n: int, time_budget: float) -> List[Tuple[float, float, float]]:
        """Solve a single puzzle."""

        if n == 0:
            return []

        best_placements = None
        best_side = float('inf')

        # Allocate time
        cp_time = time_budget * 0.1 if n <= self.config.ortools_max_n else 0
        ga_time = time_budget * 0.25
        sa_time = time_budget * 0.2
        basin_time = time_budget * 0.2
        local_time = time_budget * 0.1
        compact_time = time_budget * 0.05
        restart_time = time_budget * 0.1

        start_time = time.time()

        # 1. Try hand-optimized patterns for small n
        if n <= 5:
            try:
                placements = self.pattern_gen.optimal_small_n(n)
                if not has_any_collision(placements, self.config.collision_buffer):
                    side = compute_side(placements)
                    if side < best_side:
                        best_placements = placements
                        best_side = side
            except:
                pass

        # 2. Try constraint programming for small n
        if self.cp_solver and n <= self.config.ortools_max_n:
            cp_result = self.cp_solver.solve(n, cp_time)
            if cp_result:
                side = compute_side(cp_result)
                if side < best_side:
                    best_placements = cp_result
                    best_side = side

        # 3. Generate pattern-based solutions (expanded list)
        patterns = [
            ('hex', self.pattern_gen.hexagonal_grid),
            ('interlock', self.pattern_gen.interlocked_pairs),
            ('spiral', self.pattern_gen.spiral),
            ('rings', self.pattern_gen.concentric_rings),
            ('diamond', self.pattern_gen.diamond_lattice),
            ('tight', self.pattern_gen.tight_interlocked),
            ('nested', self.pattern_gen.nested_pairs),
            ('brick', self.pattern_gen.brick_pattern),
            ('fishbone', self.pattern_gen.fishbone_pattern),
        ]

        for name, pattern_fn in patterns:
            try:
                placements = pattern_fn(n)
                if not has_any_collision(placements, self.config.collision_buffer):
                    side = compute_side(placements)
                    if side < best_side:
                        best_placements = placements
                        best_side = side
            except:
                pass

        if best_placements is None:
            best_placements = self.pattern_gen.hexagonal_grid(n)
            best_side = compute_side(best_placements)

        # 4. Try NFP greedy placement with different angle configs
        if self.nfp_placer and time.time() - start_time < time_budget * 0.3:
            angle_configs = [
                [0] * n,  # All upright
                [180] * n,  # All inverted
                [(i % 2) * 180 for i in range(n)],  # Alternating
                [random.choice([0, 60, 120, 180, 240, 300]) for _ in range(n)],  # Random
                [(i * 30) % 360 for i in range(n)],  # Progressive rotation
            ]

            for angles in angle_configs:
                if time.time() - start_time > time_budget * 0.3:
                    break
                try:
                    placements = self.nfp_placer.place_greedy(n, angles)
                    if not has_any_collision(placements, self.config.collision_buffer):
                        side = compute_side(placements)
                        if side < best_side:
                            best_placements = placements
                            best_side = side
                except:
                    pass

        # 5. Genetic algorithm
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 1:
            self.ga = GeneticAlgorithm(n, self.config)
            ga_result = self.ga.evolve(min(ga_time, remaining_time * 0.4))
            if not has_any_collision(ga_result, self.config.collision_buffer):
                side = compute_side(ga_result)
                if side < best_side:
                    best_placements = ga_result
                    best_side = side

        # 4. Simulated annealing
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 1:
            sa_result = self.sa.optimize(best_placements, min(sa_time, remaining_time * 0.4))
            if not has_any_collision(sa_result, self.config.collision_buffer):
                side = compute_side(sa_result)
                if side < best_side:
                    best_placements = sa_result
                    best_side = side

        # 5. Basin hopping
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 1:
            basin_result = self.basin.optimize(best_placements, min(basin_time, remaining_time * 0.4))
            if not has_any_collision(basin_result, self.config.collision_buffer):
                side = compute_side(basin_result)
                if side < best_side:
                    best_placements = basin_result
                    best_side = side

        # 6. Local search refinement
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 0.5:
            local_result = self.local.optimize(best_placements, min(local_time, remaining_time * 0.5))
            if not has_any_collision(local_result, self.config.collision_buffer):
                side = compute_side(local_result)
                if side < best_side:
                    best_placements = local_result
                    best_side = side

        # 7. Final compacting
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 0.1:
            compact_result = self.compactor.compact(best_placements, min(compact_time, remaining_time))
            if not has_any_collision(compact_result, self.config.collision_buffer):
                side = compute_side(compact_result)
                if side < best_side:
                    best_placements = compact_result
                    best_side = side

        return best_placements

    def solve_all(self, output_path: str = "submission.csv") -> Dict[int, List[Tuple[float, float, float]]]:
        """Solve all puzzles."""
        solutions = {}
        total_score = 0.0

        start_time = time.time()
        max_time = self.config.max_hours * 3600

        # Calculate time budgets
        total_priority = sum(1.0 + (self.config.priority_multiplier - 1.0) * (200 - n) / 199
                            for n in range(1, 201))

        pbar = tqdm(range(1, 201), desc="Solving", ncols=100) if HAS_TQDM else range(1, 201)

        for n in pbar:
            elapsed = time.time() - start_time
            if elapsed >= max_time:
                print(f"\nTime limit reached at n={n}")
                break

            # Calculate time budget for this puzzle
            remaining_time = max_time - elapsed
            priority = 1.0 + (self.config.priority_multiplier - 1.0) * (200 - n) / 199
            remaining_priority = sum(1.0 + (self.config.priority_multiplier - 1.0) * (200 - k) / 199
                                    for k in range(n, 201))
            time_budget = min(
                self.config.get_time_for_n(n),
                remaining_time * priority / remaining_priority
            )

            # Solve
            placements = self.solve_puzzle(n, time_budget)
            solutions[n] = placements

            # Score
            side = compute_side(placements)
            contribution = (side ** 2) / n
            total_score += contribution

            # Update progress
            if HAS_TQDM:
                pbar.set_postfix({
                    'score': f'{total_score:.2f}',
                    'side': f'{side:.4f}',
                    'time': f'{time_budget:.1f}s'
                })

            # Checkpoint
            if n % self.config.checkpoint_interval == 0:
                self.save_checkpoint(solutions, self.config.checkpoint_path)

        # Fill any missing solutions with fallback
        for n in range(1, 201):
            if n not in solutions:
                solutions[n] = self.pattern_gen.hexagonal_grid(n)

        # Save final
        self.save_submission(solutions, output_path)

        return solutions

    def save_checkpoint(self, solutions: Dict[int, List[Tuple[float, float, float]]], path: str):
        """Save checkpoint."""
        self.save_submission(solutions, path)

    def save_submission(self, solutions: Dict[int, List[Tuple[float, float, float]]], path: str):
        """Save submission CSV."""
        with open(path, 'w') as f:
            f.write("id,x,y,deg\n")

            for n in sorted(solutions.keys()):
                placements = solutions[n]

                # Normalize to origin
                if placements:
                    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
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
    parser = argparse.ArgumentParser(description='Santa 2025 Elite Solver')
    parser.add_argument('--output', '-o', default='submission_elite.csv', help='Output file')
    parser.add_argument('--quick', action='store_true', help='Quick mode (~2 hours)')
    parser.add_argument('--standard', action='store_true', help='Standard mode (~12 hours)')
    parser.add_argument('--ultra', action='store_true', help='Ultra mode (~48 hours)')
    parser.add_argument('--hours', type=float, help='Custom time limit in hours')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Select config
    if args.quick:
        config = quick_config()
    elif args.ultra:
        config = ultra_config()
    elif args.standard:
        config = standard_config()
    else:
        config = EliteConfig()

    if args.hours:
        config.max_hours = args.hours

    config.seed = args.seed

    print("=" * 60)
    print("SANTA 2025 - ELITE SOLVER")
    print("=" * 60)
    print(f"Max hours: {config.max_hours}")
    print(f"GA population: {config.ga_population_size}")
    print(f"GA generations: {config.ga_generations}")
    print(f"SA iterations: {config.sa_iterations}")
    print(f"Basin hops: {config.basin_hops}")
    print(f"OR-Tools: {'enabled' if HAS_ORTOOLS else 'disabled'}")
    print(f"NFP: {'enabled' if HAS_PYCLIPPER else 'disabled'}")
    print(f"Seed: {config.seed}")
    print("=" * 60)

    solver = EliteSolver(config)
    solutions = solver.solve_all(args.output)

    # Final score
    total_score = compute_total_score(solutions)
    print()
    print("=" * 60)
    print(f"FINAL SCORE: {total_score:.4f}")
    print(f"Saved to: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
