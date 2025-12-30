#!/usr/bin/env python3
"""
Santa 2025 - ULTRA Optimization Solver
========================================

Ultra-aggressive optimization targeting score < 55:
- Genetic Algorithm with population evolution
- 100+ restarts per puzzle
- Multi-phase simulated annealing (coarse-to-fine)
- Aggressive compacting phase
- Multiple placement strategies
- Parallel execution where possible

Usage:
    python ultra_solver.py [--output submission.csv] [--seed 42] [--workers 4]
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from shapely.geometry import Polygon, Point
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapely.prepared import prep

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UltraConfig:
    """Configuration for ultra optimization."""
    # Multi-restart - MUCH more aggressive
    num_restarts: int = 100  # 100 restarts per puzzle

    # Genetic Algorithm
    population_size: int = 30  # Population for GA
    ga_generations: int = 50  # Generations per restart
    ga_elite_count: int = 5  # Top solutions to keep
    ga_mutation_rate: float = 0.3  # Mutation probability
    ga_crossover_rate: float = 0.7  # Crossover probability

    # Greedy placement
    num_placement_attempts: int = 500  # Many more attempts
    start_radius: float = 20.0
    step_in: float = 0.05  # Much finer step
    step_out: float = 0.002  # Ultra-fine backup step

    # Collision buffer
    collision_buffer: float = 0.015  # Smaller buffer for tighter packing

    # Rotation angles - test all 72
    num_rotation_angles: int = 72  # Every 5 degrees

    # Multi-phase Simulated Annealing
    sa_phases: int = 3  # Coarse -> Medium -> Fine
    sa_iterations_per_phase: List[int] = field(default_factory=lambda: [20000, 30000, 20000])
    sa_temp_initial: List[float] = field(default_factory=lambda: [3.0, 1.0, 0.2])
    sa_temp_final: List[float] = field(default_factory=lambda: [0.5, 0.05, 0.0001])

    # Local search
    local_search_precision: float = 0.002
    local_search_iterations: int = 1000

    # Compacting phase
    compacting_passes: int = 5
    compacting_step: float = 0.001

    # Time budget (seconds per puzzle based on n)
    time_base: float = 30.0  # Base time
    time_scale: float = 0.5  # Additional time per n

    # Parallel workers
    num_workers: int = 4

    seed: int = 42


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

# Pre-compute all 72 rotations
ROTATED_POLYGONS = {}
for deg in range(0, 360, 5):
    ROTATED_POLYGONS[deg] = affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))


def make_tree_polygon(x: float, y: float, angle_deg: float) -> Polygon:
    """Create tree polygon at position with rotation."""
    snapped = int(round(angle_deg / 5) * 5) % 360
    poly = ROTATED_POLYGONS[snapped]
    if x != 0 or y != 0:
        return affinity.translate(poly, xoff=x, yoff=y)
    return poly


def has_collision(tree_poly: Polygon, other_polys: List[Polygon], buffer: float = 0.0) -> bool:
    """Check if tree_poly collides with any other polygon."""
    for poly in other_polys:
        if buffer > 0:
            if tree_poly.distance(poly) < buffer:
                return True
        else:
            if tree_poly.intersects(poly) and not tree_poly.touches(poly):
                return True
    return False


def has_collision_fast(tree_poly: Polygon, tree_index: STRtree,
                       all_polys: List[Polygon], buffer: float = 0.0) -> bool:
    """Fast collision check using spatial index."""
    if buffer > 0:
        query_poly = tree_poly.buffer(buffer)
    else:
        query_poly = tree_poly

    candidates = tree_index.query(query_poly)
    for idx in candidates:
        if buffer > 0:
            if tree_poly.distance(all_polys[idx]) < buffer:
                return True
        else:
            if tree_poly.intersects(all_polys[idx]) and not tree_poly.touches(all_polys[idx]):
                return True
    return False


def compute_bounding_square(placements: List[Tuple[float, float, float]]) -> float:
    """Compute bounding square side from placements."""
    if not placements:
        return 0.0
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return max(width, height)


def bounding_square_from_polys(polygons: List[Polygon]) -> float:
    """Compute bounding square side from polygon list."""
    if not polygons:
        return 0.0
    bounds = unary_union(polygons).bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return max(width, height)


def center_placements(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Center placements around origin."""
    if not placements:
        return placements
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    bounds = unary_union(polys).bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]


def check_overlaps(placements: List[Tuple[float, float, float]], buffer: float = 0.0) -> List[Tuple[int, int]]:
    """Find all overlapping pairs."""
    polys = [make_tree_polygon(x, y, d) for x, y, d in placements]
    overlaps = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if buffer > 0:
                if polys[i].distance(polys[j]) < buffer:
                    overlaps.append((i, j))
            else:
                if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                    overlaps.append((i, j))
    return overlaps


# =============================================================================
# PLACEMENT STRATEGIES
# =============================================================================

def generate_weighted_angle() -> float:
    """Generate random angle weighted toward corners."""
    while True:
        angle = random.uniform(0, 2 * math.pi)
        if random.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle


def place_tree_radial(base_polygon: Polygon, existing_polys: List[Polygon],
                      config: UltraConfig) -> Tuple[float, float]:
    """Place tree using radial greedy approach."""
    if not existing_polys:
        return 0.0, 0.0

    buffer = config.collision_buffer
    tree_index = STRtree(existing_polys)

    best_x, best_y = None, None
    min_dist = float('inf')

    for _ in range(config.num_placement_attempts):
        angle = generate_weighted_angle()
        vx = math.cos(angle)
        vy = math.sin(angle)

        radius = config.start_radius
        collision_found = False

        while radius >= 0:
            px = radius * vx
            py = radius * vy
            candidate_poly = affinity.translate(base_polygon, xoff=px, yoff=py)

            if has_collision_fast(candidate_poly, tree_index, existing_polys, buffer):
                collision_found = True
                break

            radius -= config.step_in

        if collision_found:
            while radius < config.start_radius * 2:
                radius += config.step_out
                px = radius * vx
                py = radius * vy
                candidate_poly = affinity.translate(base_polygon, xoff=px, yoff=py)

                if not has_collision_fast(candidate_poly, tree_index, existing_polys, buffer):
                    break
        else:
            radius = 0
            px, py = 0.0, 0.0

        dist = px*px + py*py
        if dist < min_dist:
            min_dist = dist
            best_x, best_y = px, py

    return best_x if best_x is not None else 0.0, best_y if best_y is not None else 0.0


def place_tree_spiral(base_polygon: Polygon, existing_polys: List[Polygon],
                      config: UltraConfig) -> Tuple[float, float]:
    """Place tree using spiral search from center."""
    if not existing_polys:
        return 0.0, 0.0

    buffer = config.collision_buffer
    tree_index = STRtree(existing_polys)

    best_x, best_y = None, None
    min_dist = float('inf')

    # Spiral outward from center
    for r in np.arange(0, config.start_radius, 0.1):
        num_angles = max(8, int(2 * math.pi * r / 0.3))
        for i in range(num_angles):
            angle = 2 * math.pi * i / num_angles + random.uniform(-0.1, 0.1)
            px = r * math.cos(angle)
            py = r * math.sin(angle)

            candidate_poly = affinity.translate(base_polygon, xoff=px, yoff=py)

            if not has_collision_fast(candidate_poly, tree_index, existing_polys, buffer):
                dist = px*px + py*py
                if dist < min_dist:
                    min_dist = dist
                    best_x, best_y = px, py
                    if r < 0.5:  # Very close to center, good enough
                        return best_x, best_y

    return best_x if best_x is not None else 0.0, best_y if best_y is not None else 0.0


def place_tree_grid_search(base_polygon: Polygon, existing_polys: List[Polygon],
                           config: UltraConfig) -> Tuple[float, float]:
    """Place tree using dense grid search."""
    if not existing_polys:
        return 0.0, 0.0

    buffer = config.collision_buffer
    tree_index = STRtree(existing_polys)

    # Get current bounds
    bounds = unary_union(existing_polys).bounds
    padding = 2.0

    best_x, best_y = None, None
    min_dist = float('inf')

    # Dense grid search
    step = 0.2
    for x in np.arange(bounds[0] - padding, bounds[2] + padding, step):
        for y in np.arange(bounds[1] - padding, bounds[3] + padding, step):
            candidate_poly = affinity.translate(base_polygon, xoff=x, yoff=y)

            if not has_collision_fast(candidate_poly, tree_index, existing_polys, buffer):
                dist = x*x + y*y
                if dist < min_dist:
                    min_dist = dist
                    best_x, best_y = x, y

    return best_x if best_x is not None else 0.0, best_y if best_y is not None else 0.0


# =============================================================================
# GENETIC ALGORITHM
# =============================================================================

class Individual:
    """Individual in genetic algorithm population."""

    def __init__(self, placements: List[Tuple[float, float, float]]):
        self.placements = list(placements)
        self._fitness = None
        self._polys = None

    @property
    def fitness(self) -> float:
        if self._fitness is None:
            self._fitness = compute_bounding_square(self.placements)
        return self._fitness

    def invalidate(self):
        self._fitness = None
        self._polys = None

    def copy(self) -> 'Individual':
        ind = Individual(list(self.placements))
        ind._fitness = self._fitness
        return ind


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    """Single-point crossover."""
    n = len(parent1.placements)
    if n <= 1:
        return parent1.copy()

    point = random.randint(1, n - 1)
    child_placements = parent1.placements[:point] + parent2.placements[point:]
    return Individual(child_placements)


def mutate(individual: Individual, config: UltraConfig, current_score: float) -> Individual:
    """Mutate individual."""
    n = len(individual.placements)
    if n == 0:
        return individual

    ind = individual.copy()

    # Mutation strength based on current score
    shift_scale = 0.02 * current_score

    # Mutate random subset of trees
    num_to_mutate = max(1, int(n * config.ga_mutation_rate))
    indices = random.sample(range(n), num_to_mutate)

    for i in indices:
        x, y, d = ind.placements[i]

        mutation_type = random.random()
        if mutation_type < 0.5:
            # Position mutation
            x += random.gauss(0, shift_scale)
            y += random.gauss(0, shift_scale)
        elif mutation_type < 0.8:
            # Rotation mutation
            d = (d + random.choice([-45, -30, -15, -5, 5, 15, 30, 45])) % 360
        else:
            # Combined mutation
            x += random.gauss(0, shift_scale * 0.5)
            y += random.gauss(0, shift_scale * 0.5)
            d = (d + random.choice([-15, -5, 5, 15])) % 360

        ind.placements[i] = (x, y, d)

    ind.invalidate()
    return ind


def evolve_population(population: List[Individual], config: UltraConfig,
                      buffer: float) -> List[Individual]:
    """Evolve population for one generation."""
    # Sort by fitness (lower is better)
    population.sort(key=lambda x: x.fitness)

    new_population = []

    # Elitism - keep best individuals
    for i in range(config.ga_elite_count):
        new_population.append(population[i].copy())

    current_best = population[0].fitness

    # Generate rest through crossover and mutation
    while len(new_population) < config.population_size:
        # Tournament selection
        tournament_size = 3
        parents = []
        for _ in range(2):
            candidates = random.sample(population, min(tournament_size, len(population)))
            winner = min(candidates, key=lambda x: x.fitness)
            parents.append(winner)

        # Crossover
        if random.random() < config.ga_crossover_rate:
            child = crossover(parents[0], parents[1])
        else:
            child = parents[0].copy()

        # Mutation
        if random.random() < config.ga_mutation_rate:
            child = mutate(child, config, current_best)

        # Validate - no overlaps
        overlaps = check_overlaps(child.placements, buffer)
        if not overlaps:
            new_population.append(child)
        else:
            # Try to repair by small adjustments
            repaired = repair_overlaps(child.placements, buffer)
            if repaired:
                new_population.append(Individual(repaired))
            else:
                # Just copy a parent
                new_population.append(parents[0].copy())

    return new_population


def repair_overlaps(placements: List[Tuple[float, float, float]],
                    buffer: float, max_attempts: int = 50) -> Optional[List[Tuple[float, float, float]]]:
    """Try to repair overlapping placements."""
    current = list(placements)

    for _ in range(max_attempts):
        overlaps = check_overlaps(current, buffer)
        if not overlaps:
            return current

        # Fix first overlap
        i, j = overlaps[0]
        polys = [make_tree_polygon(x, y, d) for x, y, d in current]

        # Move one tree away
        x_i, y_i, d_i = current[i]
        x_j, y_j, d_j = current[j]

        # Direction from j to i
        dx = x_i - x_j
        dy = y_i - y_j
        dist = math.sqrt(dx*dx + dy*dy) + 0.001

        # Move i away from j
        move_dist = buffer + 0.01
        new_x = x_i + move_dist * dx / dist
        new_y = y_i + move_dist * dy / dist
        current[i] = (new_x, new_y, d_i)

    return None


# =============================================================================
# SIMULATED ANNEALING - MULTI-PHASE
# =============================================================================

def simulated_annealing_phase(placements: List[Tuple[float, float, float]],
                               config: UltraConfig,
                               phase: int,
                               time_limit: float) -> List[Tuple[float, float, float]]:
    """Run one phase of simulated annealing."""
    n = len(placements)
    if n <= 1:
        return placements

    buffer = config.collision_buffer
    start_time = time.time()

    current = list(placements)
    polys = [make_tree_polygon(x, y, d) for x, y, d in current]

    current_score = bounding_square_from_polys(polys)
    best_score = current_score
    best = list(current)

    # Phase-specific parameters
    iterations = config.sa_iterations_per_phase[phase]
    T = config.sa_temp_initial[phase]
    T_final = config.sa_temp_final[phase]

    # Move scales based on phase
    if phase == 0:  # Coarse
        max_shift = 0.15 * current_score
        max_rotate = 45.0
    elif phase == 1:  # Medium
        max_shift = 0.05 * current_score
        max_rotate = 20.0
    else:  # Fine
        max_shift = 0.01 * current_score
        max_rotate = 10.0

    iter_count = 0
    accepted = 0

    while iter_count < iterations and time.time() - start_time < time_limit:
        iter_count += 1

        # Pick random tree
        i = random.randrange(n)
        x, y, deg = current[i]

        # Random move
        move_type = random.random()
        if move_type < 0.4:
            # Small translation
            new_x = x + random.gauss(0, max_shift)
            new_y = y + random.gauss(0, max_shift)
            new_deg = deg
        elif move_type < 0.6:
            # Large translation (escape)
            new_x = x + random.uniform(-max_shift * 3, max_shift * 3)
            new_y = y + random.uniform(-max_shift * 3, max_shift * 3)
            new_deg = deg
        elif move_type < 0.8:
            # Rotation
            new_x, new_y = x, y
            new_deg = (deg + random.choice([-5, 5, -10, 10, -15, 15, -30, 30])) % 360
        else:
            # Center-seeking
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx = cx - x
            dy = cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 0.001
            step = random.uniform(0.01, 0.1) * current_score
            new_x = x + step * dx / dist
            new_y = y + step * dy / dist
            new_deg = deg

        # Check collision
        new_poly = make_tree_polygon(new_x, new_y, new_deg)
        others = polys[:i] + polys[i+1:]

        if has_collision(new_poly, others, buffer):
            continue

        # Evaluate
        old_poly = polys[i]
        polys[i] = new_poly
        new_score = bounding_square_from_polys(polys)

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

        # Cool down
        progress = iter_count / iterations
        T = config.sa_temp_initial[phase] * ((T_final / config.sa_temp_initial[phase]) ** progress)

        # Adaptive step size
        if iter_count % 500 == 0:
            accept_rate = accepted / iter_count
            if accept_rate > 0.4:
                max_shift *= 1.1
            elif accept_rate < 0.1:
                max_shift *= 0.9

    return best


def multi_phase_sa(placements: List[Tuple[float, float, float]],
                   config: UltraConfig,
                   time_limit: float) -> List[Tuple[float, float, float]]:
    """Run multi-phase simulated annealing."""
    current = placements
    time_per_phase = time_limit / config.sa_phases

    for phase in range(config.sa_phases):
        current = simulated_annealing_phase(current, config, phase, time_per_phase)

    return current


# =============================================================================
# LOCAL SEARCH
# =============================================================================

def local_search(placements: List[Tuple[float, float, float]],
                 config: UltraConfig) -> List[Tuple[float, float, float]]:
    """Fine-grained local search."""
    n = len(placements)
    if n <= 1:
        return placements

    buffer = config.collision_buffer
    precision = config.local_search_precision

    current = list(placements)
    polys = [make_tree_polygon(x, y, d) for x, y, d in current]
    current_score = bounding_square_from_polys(polys)

    improved = True
    iteration = 0

    while improved and iteration < config.local_search_iterations:
        improved = False
        iteration += 1

        for i in range(n):
            x, y, deg = current[i]

            # Try small moves
            for dx, dy in [(-precision, 0), (precision, 0),
                           (0, -precision), (0, precision),
                           (-precision, -precision), (precision, -precision),
                           (-precision, precision), (precision, precision)]:
                new_x, new_y = x + dx, y + dy
                new_poly = make_tree_polygon(new_x, new_y, deg)

                others = polys[:i] + polys[i+1:]
                if not has_collision(new_poly, others, buffer):
                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bounding_square_from_polys(polys)

                    if new_score < current_score - 0.0001:
                        current[i] = (new_x, new_y, deg)
                        current_score = new_score
                        x, y = new_x, new_y
                        improved = True
                    else:
                        polys[i] = old_poly

            # Try rotation
            for d_deg in [-5, 5]:
                new_deg = (deg + d_deg) % 360
                new_poly = make_tree_polygon(x, y, new_deg)

                others = polys[:i] + polys[i+1:]
                if not has_collision(new_poly, others, buffer):
                    old_poly = polys[i]
                    polys[i] = new_poly
                    new_score = bounding_square_from_polys(polys)

                    if new_score < current_score - 0.0001:
                        current[i] = (x, y, new_deg)
                        current_score = new_score
                        improved = True
                    else:
                        polys[i] = old_poly

    return current


# =============================================================================
# COMPACTING PHASE
# =============================================================================

def compact_solution(placements: List[Tuple[float, float, float]],
                     config: UltraConfig) -> List[Tuple[float, float, float]]:
    """Aggressively compact solution toward center."""
    n = len(placements)
    if n <= 1:
        return placements

    buffer = config.collision_buffer
    step = config.compacting_step

    current = list(placements)

    for _ in range(config.compacting_passes):
        polys = [make_tree_polygon(x, y, d) for x, y, d in current]

        # Compute centroid
        cx = sum(p[0] for p in current) / n
        cy = sum(p[1] for p in current) / n

        # Sort by distance from centroid (farthest first)
        distances = [(i, math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)) for i, p in enumerate(current)]
        distances.sort(key=lambda x: -x[1])

        for idx, dist in distances:
            if dist < 0.1:
                continue

            x, y, deg = current[idx]

            # Direction toward centroid
            dx = cx - x
            dy = cy - y
            d = math.sqrt(dx*dx + dy*dy) + 0.001

            # Try moving toward center
            best_x, best_y = x, y
            best_dist = dist

            for mult in [1.0, 0.5, 0.2, 0.1]:
                test_step = step * mult * dist
                new_x = x + test_step * dx / d
                new_y = y + test_step * dy / d

                new_poly = make_tree_polygon(new_x, new_y, deg)
                others = polys[:idx] + polys[idx+1:]

                if not has_collision(new_poly, others, buffer):
                    new_dist = math.sqrt((new_x-cx)**2 + (new_y-cy)**2)
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_x, best_y = new_x, new_y

            if best_x != x or best_y != y:
                current[idx] = (best_x, best_y, deg)
                polys[idx] = make_tree_polygon(best_x, best_y, deg)

    return current


# =============================================================================
# MAIN SOLVER
# =============================================================================

class UltraSolver:
    """Ultra optimization solver."""

    def __init__(self, config: Optional[UltraConfig] = None, seed: int = 42):
        self.config = config or UltraConfig()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.scores: Dict[int, float] = {}

    def solve_single(self, n: int, prev_solution: List[Tuple[float, float, float]],
                     verbose: bool = False) -> Tuple[List[Tuple[float, float, float]], float]:
        """Solve for n trees with ultra optimization."""
        if n <= 0:
            return [], 0.0

        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            return sol, compute_bounding_square(sol)

        # Time budget
        time_budget = self.config.time_base + self.config.time_scale * n
        start_time = time.time()

        prev_polys = [make_tree_polygon(x, y, d) for x, y, d in prev_solution]

        best_solution = None
        best_score = float('inf')

        # Get all rotation angles
        rotation_angles = [i * 5.0 for i in range(72)]

        for restart in range(self.config.num_restarts):
            if time.time() - start_time >= time_budget * 0.95:
                break

            # Vary seed
            random.seed(self.seed + restart * 1000 + n)

            # Try different rotation angle for new tree
            angle_idx = restart % len(rotation_angles)
            new_angle = rotation_angles[angle_idx]
            new_base = ROTATED_POLYGONS[int(new_angle) % 360]

            # Try different placement strategies
            strategy = restart % 3
            if strategy == 0:
                best_x, best_y = place_tree_radial(new_base, prev_polys, self.config)
            elif strategy == 1:
                best_x, best_y = place_tree_spiral(new_base, prev_polys, self.config)
            else:
                best_x, best_y = place_tree_grid_search(new_base, prev_polys, self.config)

            # Build initial solution
            solution = prev_solution + [(best_x, best_y, new_angle)]

            # Perturb for restarts > 0
            if restart > 0 and best_solution is not None and restart % 5 != 0:
                # Start from best and perturb
                solution = list(best_solution)
                num_to_perturb = max(1, int(n * 0.15))
                indices = random.sample(range(n), num_to_perturb)

                for idx in indices:
                    x, y, d = solution[idx]
                    solution[idx] = (
                        x + random.gauss(0, 0.03 * best_score),
                        y + random.gauss(0, 0.03 * best_score),
                        (d + random.uniform(-10, 10)) % 360
                    )

            # Validate initial solution
            overlaps = check_overlaps(solution, self.config.collision_buffer)
            if overlaps:
                repaired = repair_overlaps(solution, self.config.collision_buffer)
                if repaired:
                    solution = repaired
                else:
                    continue

            # Calculate remaining time for this restart
            elapsed = time.time() - start_time
            remaining = (time_budget - elapsed) / max(1, self.config.num_restarts - restart)

            # Multi-phase simulated annealing
            if remaining > 0.5:
                solution = multi_phase_sa(solution, self.config, remaining * 0.6)

            # Local search
            solution = local_search(solution, self.config)

            # Compacting
            solution = compact_solution(solution, self.config)

            # Final local search
            solution = local_search(solution, self.config)

            # Evaluate
            score = compute_bounding_square(solution)

            # Validate
            overlaps = check_overlaps(solution, self.config.collision_buffer)
            if not overlaps and score < best_score:
                best_score = score
                best_solution = solution

        # Final optimization on best solution
        if best_solution is not None:
            # One more compacting pass
            best_solution = compact_solution(best_solution, self.config)
            best_solution = local_search(best_solution, self.config)
            best_score = compute_bounding_square(best_solution)
        else:
            # Fallback
            new_angle = random.uniform(0, 360)
            new_base = affinity.rotate(BASE_POLYGON, new_angle, origin=(0, 0))
            best_x, best_y = place_tree_radial(new_base, prev_polys, self.config)
            best_solution = prev_solution + [(best_x, best_y, new_angle)]
            best_score = compute_bounding_square(best_solution)

        return center_placements(best_solution), best_score

    def solve_all(self, max_n: int = 200, verbose: bool = True) -> Dict[int, List[Tuple[float, float, float]]]:
        """Solve for all n from 1 to max_n."""
        total_start = time.time()

        if verbose:
            print("=" * 70)
            print("SANTA 2025 - ULTRA OPTIMIZATION SOLVER")
            print("=" * 70)
            print(f"Restarts per puzzle: {self.config.num_restarts}")
            print(f"Population size: {self.config.population_size}")
            print(f"SA phases: {self.config.sa_phases}")
            print(f"Placement attempts: {self.config.num_placement_attempts}")
            print(f"Collision buffer: {self.config.collision_buffer}")
            print()

        for n in range(1, max_n + 1):
            n_start = time.time()

            if n == 1:
                prev_solution = []
            else:
                prev_solution = self.solutions[n - 1]

            solution, score = self.solve_single(n, prev_solution, verbose)

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
            side = self.scores.get(n, compute_bounding_square(sol))
            total += (side ** 2) / n
        return total


# =============================================================================
# VALIDATION AND I/O
# =============================================================================

def validate_solution(solution: List[Tuple[float, float, float]], n: int) -> Tuple[bool, str]:
    """Validate a single solution."""
    if len(solution) != n:
        return False, f"Expected {n} trees, got {len(solution)}"

    if n == 0:
        return True, "OK"

    overlaps = check_overlaps(solution)
    if overlaps:
        return False, f"{len(overlaps)} overlapping pair(s)"

    return True, "OK"


def validate_all(solutions: Dict[int, List[Tuple[float, float, float]]],
                 max_n: int = 200, verbose: bool = True) -> bool:
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


def create_submission(solutions: Dict[int, List[Tuple[float, float, float]]],
                      output_path: str = "submission.csv") -> str:
    """Create submission CSV."""
    with open(output_path, "w") as f:
        f.write("id,x,y,deg\n")

        for n in range(1, 201):
            if n not in solutions:
                raise ValueError(f"Missing solution for n={n}")

            positions = solutions[n]
            if len(positions) != n:
                raise ValueError(f"Wrong count for n={n}")

            polys = [make_tree_polygon(x, y, d) for x, y, d in positions]
            bounds = unary_union(polys).bounds
            min_x, min_y = bounds[0], bounds[1]

            for idx, (x, y, deg) in enumerate(positions):
                X = x - min_x
                Y = y - min_y
                f.write(f"{n:03d}_{idx},s{X:.6f},s{Y:.6f},s{deg:.6f}\n")

    return output_path


def print_score_summary(solutions: Dict[int, List[Tuple[float, float, float]]], max_n: int = 200):
    """Print detailed score summary."""
    print("=" * 60)
    print("SCORE SUMMARY")
    print("=" * 60)

    sides = {}
    contributions = {}

    for n in range(1, max_n + 1):
        if n not in solutions:
            continue
        side = compute_bounding_square(solutions[n])
        sides[n] = side
        contributions[n] = (side ** 2) / n

    total = sum(contributions.values())
    baseline = 157.08

    print(f"Total score: {total:.4f}")
    print(f"Baseline: {baseline}")
    print(f"Improvement: {(baseline - total) / baseline * 100:.1f}%")
    print()

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
    parser = argparse.ArgumentParser(description="Santa 2025 Ultra Optimization Solver")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-n", type=int, default=200, help="Maximum n to solve")
    parser.add_argument("--restarts", type=int, default=100, help="Restarts per puzzle")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    args = parser.parse_args()

    config = UltraConfig()
    config.seed = args.seed
    config.num_restarts = args.restarts

    if args.quick:
        config.num_restarts = 20
        config.num_placement_attempts = 100
        config.sa_iterations_per_phase = [5000, 10000, 5000]
        config.time_base = 10.0
        config.time_scale = 0.2

    print("Configuration:")
    print(f"  Restarts: {config.num_restarts}")
    print(f"  Placement attempts: {config.num_placement_attempts}")
    print(f"  SA phases: {config.sa_phases}")
    print(f"  Collision buffer: {config.collision_buffer}")
    print()

    solver = UltraSolver(config=config, seed=args.seed)
    solutions = solver.solve_all(max_n=args.max_n, verbose=True)

    print("\nValidating...")
    validate_all(solutions, max_n=args.max_n, verbose=True)

    print()
    print_score_summary(solutions, max_n=args.max_n)

    print(f"\nCreating submission: {args.output}")
    create_submission(solutions, args.output)
    print(f"Saved: {args.output}")

    total_score = solver.compute_total_score()
    print(f"\nFinal Score: {total_score:.4f}")

    return solutions


if __name__ == "__main__":
    main()
