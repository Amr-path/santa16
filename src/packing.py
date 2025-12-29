"""
packing.py - Optimized packing solver for Santa 2025

Combines techniques from:
1. Official Kaggle starter (greedy radial placement with weighted angles)
2. Simulated Annealing for refinement
3. Multiple initialization strategies
"""
import math
import random
from decimal import Decimal
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

from .geometry import (
    make_tree_polygon, has_collision, has_collision_strtree,
    compute_bounding_square_side, bounding_square_side_from_polys,
    center_placements, check_all_overlaps,
    SIMPLE_BASE_POLYGON, TREE_WIDTH, TREE_HEIGHT
)

Placement = Tuple[float, float, float]
Solution = List[Placement]


@dataclass
class SolverConfig:
    """Configuration for the solver."""
    # Greedy placement settings
    num_placement_attempts: int = 20      # More attempts = better placement (official uses 10)
    start_radius: float = 25.0            # Starting distance for greedy placement
    step_in: float = 0.3                  # Step size moving toward center
    step_out: float = 0.03                # Fine step backing up after collision

    # Collision buffer - minimum separation between trees
    collision_buffer: float = 0.02        # Required minimum separation

    # Simulated Annealing settings
    sa_enabled: bool = True
    sa_iterations_base: int = 3000        # Base SA iterations per n
    sa_iterations_scale: float = 1.5      # Scale factor for larger n
    sa_temp_initial: float = 1.0
    sa_temp_final: float = 0.0001
    sa_max_shift: float = 0.2             # Initial max translation
    sa_max_rotate: float = 45.0           # Initial max rotation
    sa_decay: float = 0.9995              # Step size decay

    # Multi-restart
    num_restarts: int = 1

    seed: int = 42
    
    @classmethod
    def quick_mode(cls):
        return cls(
            num_placement_attempts=15,
            sa_iterations_base=1500,
            num_restarts=1,
            collision_buffer=0.02
        )

    @classmethod
    def standard_mode(cls):
        return cls(
            num_placement_attempts=25,
            sa_iterations_base=4000,
            num_restarts=2,
            collision_buffer=0.02
        )

    @classmethod
    def aggressive_mode(cls):
        return cls(
            num_placement_attempts=40,
            sa_iterations_base=8000,
            sa_iterations_scale=2.0,
            num_restarts=3,
            collision_buffer=0.02
        )

    @classmethod
    def maximum_mode(cls):
        return cls(
            num_placement_attempts=60,
            sa_iterations_base=15000,
            sa_iterations_scale=2.5,
            sa_temp_final=0.00001,
            num_restarts=5,
            collision_buffer=0.02
        )

    @classmethod
    def optimized_mode(cls):
        """Optimized mode for sub-60 score with strong collision buffer."""
        return cls(
            num_placement_attempts=100,
            start_radius=15.0,
            step_in=0.1,
            step_out=0.01,
            collision_buffer=0.025,  # Slightly larger buffer for safety
            sa_enabled=True,
            sa_iterations_base=20000,
            sa_iterations_scale=3.0,
            sa_temp_initial=2.0,
            sa_temp_final=0.00001,
            sa_max_shift=0.15,
            sa_max_rotate=60.0,
            sa_decay=0.9998,
            num_restarts=10
        )


def generate_weighted_angle() -> float:
    """
    Generate random angle weighted by abs(sin(2*angle)).
    This favors placing trees in corners (diagonal directions).
    From official Kaggle code.
    """
    while True:
        angle = random.uniform(0, 2 * math.pi)
        if random.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle


def place_tree_greedy(
    base_polygon: Polygon,
    existing_polys: List[Polygon],
    tree_index: Optional[STRtree],
    config: SolverConfig
) -> Tuple[float, float]:
    """
    Place a new tree using greedy radial approach from official code.

    Strategy:
    1. Start at radius=25 from center at a weighted random angle
    2. Move toward center in steps until collision
    3. Back up in fine steps until no collision (with buffer)
    4. Try multiple attempts, keep the one closest to center
    """
    if not existing_polys:
        return 0.0, 0.0

    buffer = config.collision_buffer
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

            # Check collision with buffer
            if tree_index:
                has_coll = has_collision_strtree(candidate_poly, tree_index, existing_polys, buffer)
            else:
                has_coll = has_collision(candidate_poly, existing_polys, buffer)

            if has_coll:
                collision_found = True
                break

            radius -= config.step_in

        # Back up until no collision (with buffer)
        if collision_found:
            while True:
                radius += config.step_out
                px = radius * vx
                py = radius * vy

                candidate_poly = affinity.translate(base_polygon, xoff=px, yoff=py)

                if tree_index:
                    has_coll = has_collision_strtree(candidate_poly, tree_index, existing_polys, buffer)
                else:
                    has_coll = has_collision(candidate_poly, existing_polys, buffer)

                if not has_coll:
                    break

                if radius > config.start_radius * 2:  # Safety limit
                    break
        else:
            # No collision found - place at center
            radius = 0
            px, py = 0.0, 0.0

        # Keep best (closest to center)
        if radius < min_radius:
            min_radius = radius
            best_x, best_y = px, py

    return best_x if best_x is not None else 0.0, best_y if best_y is not None else 0.0


def simulated_annealing(
    placements: Solution,
    config: SolverConfig,
    n_iterations: int
) -> Solution:
    """
    Optimize placements using Simulated Annealing with collision buffer.
    """
    n = len(placements)
    if n <= 1:
        return placements

    buffer = config.collision_buffer
    current = list(placements)
    polys = [make_tree_polygon(x, y, d) for x, y, d in current]

    current_score = bounding_square_side_from_polys(polys)
    best_score = current_score
    best = list(current)

    # Temperature schedule
    T = config.sa_temp_initial
    cooling = (config.sa_temp_final / config.sa_temp_initial) ** (1.0 / n_iterations)

    max_shift = config.sa_max_shift * current_score
    max_rotate = config.sa_max_rotate

    for _ in range(n_iterations):
        # Pick random tree
        i = random.randrange(n)
        x, y, deg = current[i]

        # Random move
        if random.random() < 0.6:
            # Translation
            new_x = x + random.uniform(-max_shift, max_shift)
            new_y = y + random.uniform(-max_shift, max_shift)
            new_deg = deg
        elif random.random() < 0.85:
            # Rotation
            new_x, new_y = x, y
            new_deg = (deg + random.uniform(-max_rotate, max_rotate)) % 360
        else:
            # Combined
            new_x = x + random.uniform(-max_shift * 0.5, max_shift * 0.5)
            new_y = y + random.uniform(-max_shift * 0.5, max_shift * 0.5)
            new_deg = (deg + random.uniform(-max_rotate * 0.5, max_rotate * 0.5)) % 360

        # Create new polygon
        new_poly = make_tree_polygon(new_x, new_y, new_deg)
        others = polys[:i] + polys[i+1:]

        # Check collision with buffer
        if has_collision(new_poly, others, buffer):
            T *= cooling
            max_shift = max(0.005, max_shift * config.sa_decay)
            max_rotate = max(0.5, max_rotate * config.sa_decay)
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

            if current_score < best_score:
                best_score = current_score
                best = list(current)
        else:
            polys[i] = old_poly

        T *= cooling
        max_shift = max(0.005, max_shift * config.sa_decay)
        max_rotate = max(0.5, max_rotate * config.sa_decay)

    return best


class PackingSolver:
    """
    Main solver combining greedy placement + Simulated Annealing.
    """
    
    def __init__(self, config: Optional[SolverConfig] = None, seed: int = 42):
        self.config = config or SolverConfig.standard_mode()
        self.seed = seed
        random.seed(seed)
        
        self.solutions: Dict[int, Solution] = {}
        self.scores: Dict[int, float] = {}
    
    def solve_single(self, n: int, verbose: bool = False) -> Solution:
        """Solve for n trees."""
        
        if n <= 0:
            return []
        
        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            self.solutions[1] = sol
            self.scores[1] = compute_bounding_square_side(sol)
            return sol
        
        # Get previous solution
        if n - 1 not in self.solutions:
            self.solve_single(n - 1, verbose=False)
        
        prev_solution = self.solutions[n - 1]
        
        # Build polygons and index for previous trees
        prev_polys = [make_tree_polygon(x, y, d) for x, y, d in prev_solution]
        tree_index = STRtree(prev_polys) if prev_polys else None
        
        # Create base polygon for new tree with random rotation
        new_angle = random.uniform(0, 360)
        new_base = affinity.rotate(SIMPLE_BASE_POLYGON, new_angle, origin=(0, 0))
        
        # Find best placement using greedy approach
        best_x, best_y = place_tree_greedy(new_base, prev_polys, tree_index, self.config)
        
        # Build new solution
        new_placement = (best_x, best_y, new_angle)
        solution = prev_solution + [new_placement]
        
        # Apply Simulated Annealing refinement
        if self.config.sa_enabled:
            n_iter = int(self.config.sa_iterations_base * (1 + n * self.config.sa_iterations_scale / 200))
            
            best_solution = solution
            best_score = compute_bounding_square_side(solution)
            
            for restart in range(self.config.num_restarts):
                if restart == 0:
                    start = list(solution)
                else:
                    # Perturb best solution
                    start = list(best_solution)
                    for i in range(len(start)):
                        if random.random() < 0.3:
                            x, y, d = start[i]
                            start[i] = (
                                x + random.uniform(-0.1, 0.1),
                                y + random.uniform(-0.1, 0.1),
                                (d + random.uniform(-30, 30)) % 360
                            )
                
                optimized = simulated_annealing(start, self.config, n_iter)
                score = compute_bounding_square_side(optimized)
                
                if score < best_score:
                    best_score = score
                    best_solution = optimized
            
            solution = best_solution
        
        # Center the solution
        solution = center_placements(solution)

        # Validate with collision buffer
        overlaps = check_all_overlaps(solution, self.config.collision_buffer)
        if overlaps:
            # Something went wrong - retry with fallback placement
            # Find valid placement without optimization
            new_angle = random.uniform(0, 360)
            new_base = affinity.rotate(SIMPLE_BASE_POLYGON, new_angle, origin=(0, 0))
            best_x, best_y = place_tree_greedy(new_base, prev_polys, tree_index, self.config)
            solution = center_placements(prev_solution + [(best_x, best_y, new_angle)])

            # Check again
            overlaps = check_all_overlaps(solution, self.config.collision_buffer)
            if overlaps:
                # If still overlapping, use larger separation
                config_backup = SolverConfig(
                    num_placement_attempts=200,
                    collision_buffer=0.05,
                    step_out=0.005
                )
                best_x, best_y = place_tree_greedy(new_base, prev_polys, tree_index, config_backup)
                solution = center_placements(prev_solution + [(best_x, best_y, new_angle)])
        
        self.solutions[n] = solution
        self.scores[n] = compute_bounding_square_side(solution)
        
        if verbose and (n % 10 == 0 or n <= 10):
            print(f"  n={n}: side={self.scores[n]:.4f}")
        
        return solution
    
    def solve_all(self, max_n: int = 200, verbose: bool = True) -> Dict[int, Solution]:
        """Solve for all n from 1 to max_n."""
        
        if verbose:
            print(f"Solving n=1 to {max_n}...")
            print(f"Config: {self.config.sa_iterations_base} SA iterations, "
                  f"{self.config.num_placement_attempts} placement attempts, "
                  f"{self.config.num_restarts} restarts")
        
        for n in range(1, max_n + 1):
            self.solve_single(n, verbose=verbose)
        
        if verbose:
            total = self.compute_total_score()
            print(f"\nTotal score: {total:.4f}")
        
        return self.solutions
    
    def compute_total_score(self) -> float:
        """Compute competition score: sum of (side^2 / n)."""
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, compute_bounding_square_side(sol))
            total += (side ** 2) / n
        return total


# Convenience alias
OptimizationConfig = SolverConfig
