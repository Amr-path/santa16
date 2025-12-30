#!/usr/bin/env python3
"""
Santa 2025 - MEGA Optimization Solver
======================================

The most aggressive optimization targeting score < 55:
- 200+ restarts per puzzle
- Hexagonal packing-inspired placement
- Extreme compacting with iterative refinement
- Basin hopping for escaping local minima
- Multiple diverse mutation strategies
- Continuous improvement loops

Usage:
    python mega_solver.py [--output submission.csv] [--seed 42]
"""

import os
import sys
import math
import time
import random
import argparse
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import heapq

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MegaConfig:
    """Configuration for mega optimization."""
    # Ultra aggressive restarts
    num_restarts: int = 200

    # Placement
    num_placement_attempts: int = 1000
    start_radius: float = 15.0
    step_in: float = 0.02
    step_out: float = 0.001

    # Collision - tighter packing
    collision_buffer: float = 0.01

    # Simulated Annealing - very aggressive
    sa_iterations: int = 100000
    sa_temp_initial: float = 5.0
    sa_temp_final: float = 0.00001

    # Local search - ultra fine
    local_precision: float = 0.001
    local_iterations: int = 2000

    # Compacting - many passes
    compact_passes: int = 20
    compact_step: float = 0.0005

    # Basin hopping
    basin_hops: int = 10
    basin_perturbation: float = 0.1

    # Time per puzzle
    time_per_n: float = 45.0

    seed: int = 42


# =============================================================================
# GEOMETRY
# =============================================================================

TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

BASE_POLYGON = Polygon(TREE_COORDS)
ROTATED_POLYGONS = {deg: affinity.rotate(BASE_POLYGON, deg, origin=(0, 0)) for deg in range(0, 360, 5)}


def make_tree(x: float, y: float, deg: float) -> Polygon:
    snapped = int(round(deg / 5) * 5) % 360
    poly = ROTATED_POLYGONS[snapped]
    return affinity.translate(poly, xoff=x, yoff=y) if x != 0 or y != 0 else poly


def collides(poly: Polygon, others: List[Polygon], buf: float = 0.0) -> bool:
    for o in others:
        if buf > 0:
            if poly.distance(o) < buf:
                return True
        elif poly.intersects(o) and not poly.touches(o):
            return True
    return False


def collides_fast(poly: Polygon, tree_idx: STRtree, all_polys: List[Polygon], buf: float = 0.0) -> bool:
    query = poly.buffer(buf) if buf > 0 else poly
    for idx in tree_idx.query(query):
        if buf > 0:
            if poly.distance(all_polys[idx]) < buf:
                return True
        elif poly.intersects(all_polys[idx]) and not poly.touches(all_polys[idx]):
            return True
    return False


def bbox_side(placements: List[Tuple[float, float, float]]) -> float:
    if not placements:
        return 0.0
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def bbox_side_polys(polys: List[Polygon]) -> float:
    if not polys:
        return 0.0
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def center(placements: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    if not placements:
        return placements
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]


def check_overlaps(placements: List[Tuple[float, float, float]], buf: float = 0.0) -> bool:
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
    while True:
        a = random.uniform(0, 2 * math.pi)
        if random.random() < abs(math.sin(2 * a)):
            return a


def place_radial(base: Polygon, existing: List[Polygon], cfg: MegaConfig) -> Tuple[float, float]:
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


def place_hexagonal(base: Polygon, existing: List[Polygon], cfg: MegaConfig) -> Tuple[float, float]:
    """Place using hexagonal grid pattern - optimal for circle packing."""
    if not existing:
        return 0.0, 0.0

    idx = STRtree(existing)
    best = (None, None)
    min_dist = float('inf')

    # Hexagonal grid spacing
    dx = 0.8  # Approximate tree width
    dy = dx * math.sqrt(3) / 2

    # Search in hexagonal grid pattern from center outward
    for ring in range(30):
        for i in range(-ring, ring + 1):
            for j in range(-ring, ring + 1):
                if abs(i) != ring and abs(j) != ring:
                    continue

                # Hexagonal offset
                x = i * dx + (j % 2) * dx / 2
                y = j * dy

                cand = affinity.translate(base, xoff=x, yoff=y)
                if not collides_fast(cand, idx, existing, cfg.collision_buffer):
                    dist = x*x + y*y
                    if dist < min_dist:
                        min_dist = dist
                        best = (x, y)

    return best[0] if best[0] is not None else 0.0, best[1] if best[1] is not None else 0.0


def place_boundary(base: Polygon, existing: List[Polygon], cfg: MegaConfig) -> Tuple[float, float]:
    """Place on the boundary of existing polygon union."""
    if not existing:
        return 0.0, 0.0

    idx = STRtree(existing)
    hull = unary_union(existing).convex_hull
    boundary = hull.boundary

    best = (None, None)
    min_dist = float('inf')

    # Sample points along boundary
    length = boundary.length
    for i in range(200):
        t = i / 200.0
        pt = boundary.interpolate(t * length)

        # Try positions around this point
        for offset in np.arange(-0.5, 0.5, 0.1):
            for angle in range(0, 360, 30):
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


# =============================================================================
# SIMULATED ANNEALING - EXTREME
# =============================================================================

def sa_extreme(placements: List[Tuple[float, float, float]], cfg: MegaConfig,
               time_limit: float) -> List[Tuple[float, float, float]]:
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
    iters = 0
    accepted = 0

    # Adaptive parameters
    shift = 0.1 * cur_score
    rot = 30.0

    while time.time() - start < time_limit and iters < cfg.sa_iterations:
        iters += 1

        i = random.randrange(n)
        x, y, d = current[i]

        # Move type selection with more variety
        mt = random.random()
        if mt < 0.25:
            # Small move
            nx = x + random.gauss(0, shift * 0.3)
            ny = y + random.gauss(0, shift * 0.3)
            nd = d
        elif mt < 0.45:
            # Medium move
            nx = x + random.gauss(0, shift)
            ny = y + random.gauss(0, shift)
            nd = d
        elif mt < 0.55:
            # Large escape
            nx = x + random.uniform(-shift * 5, shift * 5)
            ny = y + random.uniform(-shift * 5, shift * 5)
            nd = d
        elif mt < 0.70:
            # Rotation only
            nx, ny = x, y
            nd = (d + random.choice([-5, 5, -10, 10, -15, 15, -30, 30, -45, 45, -90, 90])) % 360
        elif mt < 0.85:
            # Center seeking
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 0.001
            step = random.uniform(0.02, 0.15) * cur_score
            nx = x + step * dx / dist
            ny = y + step * dy / dist
            nd = d
        else:
            # Swap with random other tree
            if n >= 2:
                j = random.randrange(n)
                while j == i:
                    j = random.randrange(n)
                ox, oy, od = current[j]

                # Try swapping positions
                new_poly_i = make_tree(ox, oy, d)
                new_poly_j = make_tree(x, y, od)

                others_i = polys[:i] + polys[i+1:]
                if i < j:
                    others_i = others_i[:j-1] + others_i[j:]
                else:
                    others_i = others_i[:j] + others_i[j+1:]

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

                # Update temperature and continue
                progress = iters / cfg.sa_iterations
                T = cfg.sa_temp_initial * ((cfg.sa_temp_final / cfg.sa_temp_initial) ** progress)
                continue

            nx, ny, nd = x, y, d

        # Regular move
        new_poly = make_tree(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if collides(new_poly, others, buf):
            progress = iters / cfg.sa_iterations
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
        progress = iters / cfg.sa_iterations
        T = cfg.sa_temp_initial * ((cfg.sa_temp_final / cfg.sa_temp_initial) ** progress)

        # Adaptive step sizes
        if iters % 1000 == 0:
            rate = accepted / iters
            if rate > 0.35:
                shift *= 1.1
                rot = min(90, rot * 1.1)
            elif rate < 0.15:
                shift *= 0.9
                rot = max(5, rot * 0.9)

    return best


# =============================================================================
# LOCAL SEARCH - ULTRA FINE
# =============================================================================

def local_search_ultra(placements: List[Tuple[float, float, float]], cfg: MegaConfig) -> List[Tuple[float, float, float]]:
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

        # Random order to avoid bias
        order = list(range(n))
        random.shuffle(order)

        for i in order:
            x, y, d = current[i]
            best_move = None
            best_improvement = 0

            # Try moves in all directions
            moves = [
                (-prec, 0), (prec, 0), (0, -prec), (0, prec),
                (-prec, -prec), (-prec, prec), (prec, -prec), (prec, prec),
                (-prec*2, 0), (prec*2, 0), (0, -prec*2), (0, prec*2),
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
            for dd in [-5, 5, -10, 10]:
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

            if best_move is not None and best_improvement > 0.00001:
                current[i] = (best_move[0], best_move[1], best_move[2])
                polys[i] = best_move[3]
                cur_score -= best_improvement
                improved = True

    return current


# =============================================================================
# COMPACTING - AGGRESSIVE
# =============================================================================

def compact_aggressive(placements: List[Tuple[float, float, float]], cfg: MegaConfig) -> List[Tuple[float, float, float]]:
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
            if dist < 0.05:
                continue

            x, y, d = current[idx]
            dx, dy = cx - x, cy - y
            norm = math.sqrt(dx*dx + dy*dy) + 0.001

            # Try multiple step sizes
            for mult in [1.0, 0.5, 0.25, 0.1, 0.05]:
                move = step * mult * dist * 5  # More aggressive
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

def basin_hop(placements: List[Tuple[float, float, float]], cfg: MegaConfig,
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
        indices = random.sample(range(n), num_perturb)

        for i in indices:
            x, y, d = perturbed[i]
            perturbed[i] = (
                x + random.gauss(0, 0.1 * cur_score),
                y + random.gauss(0, 0.1 * cur_score),
                (d + random.uniform(-30, 30)) % 360
            )

        # Check validity
        if check_overlaps(perturbed, buf):
            continue

        # Local minimization
        optimized = sa_extreme(perturbed, cfg, time_per_hop * 0.5)
        optimized = local_search_ultra(optimized, cfg)
        optimized = compact_aggressive(optimized, cfg)

        opt_score = bbox_side(optimized)

        # Accept if better
        if opt_score < best_score and not check_overlaps(optimized, buf):
            best = optimized
            best_score = opt_score
            current = optimized
            cur_score = opt_score
        elif opt_score < cur_score * 1.1:  # Accept slightly worse for exploration
            current = optimized
            cur_score = opt_score

    return best


# =============================================================================
# MAIN SOLVER
# =============================================================================

class MegaSolver:
    def __init__(self, config: Optional[MegaConfig] = None, seed: int = 42):
        self.cfg = config or MegaConfig()
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.solutions: Dict[int, List[Tuple[float, float, float]]] = {}
        self.scores: Dict[int, float] = {}

    def solve_single(self, n: int, prev: List[Tuple[float, float, float]],
                     verbose: bool = False) -> Tuple[List[Tuple[float, float, float]], float]:
        if n <= 0:
            return [], 0.0

        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
            return sol, bbox_side(sol)

        start = time.time()
        time_budget = self.cfg.time_per_n

        prev_polys = [make_tree(x, y, d) for x, y, d in prev]

        best_sol = None
        best_score = float('inf')

        angles = [i * 5.0 for i in range(72)]

        for restart in range(self.cfg.num_restarts):
            if time.time() - start >= time_budget * 0.95:
                break

            random.seed(self.seed + restart * 1000 + n)

            # Rotation for new tree
            angle = angles[restart % 72]
            base = ROTATED_POLYGONS[int(angle) % 360]

            # Placement strategy rotation
            strat = restart % 4
            if strat == 0:
                bx, by = place_radial(base, prev_polys, self.cfg)
            elif strat == 1:
                bx, by = place_hexagonal(base, prev_polys, self.cfg)
            elif strat == 2:
                bx, by = place_boundary(base, prev_polys, self.cfg)
            else:
                # Combined: try all and pick best
                candidates = []
                for place_fn in [place_radial, place_hexagonal, place_boundary]:
                    px, py = place_fn(base, prev_polys, self.cfg)
                    candidates.append((px*px + py*py, px, py))
                candidates.sort()
                bx, by = candidates[0][1], candidates[0][2]

            sol = prev + [(bx, by, angle)]

            # Perturb from best on later restarts
            if restart > 0 and best_sol is not None and restart % 3 != 0:
                sol = list(best_sol)
                num_p = max(1, int(n * 0.1))
                for idx in random.sample(range(n), num_p):
                    x, y, d = sol[idx]
                    sol[idx] = (
                        x + random.gauss(0, 0.02 * best_score),
                        y + random.gauss(0, 0.02 * best_score),
                        (d + random.uniform(-10, 10)) % 360
                    )

            if check_overlaps(sol, self.cfg.collision_buffer):
                continue

            # Optimization pipeline
            remaining = time_budget - (time.time() - start)
            per_restart = remaining / max(1, self.cfg.num_restarts - restart)

            if per_restart > 0.3:
                sol = sa_extreme(sol, self.cfg, per_restart * 0.5)
                sol = local_search_ultra(sol, self.cfg)
                sol = compact_aggressive(sol, self.cfg)
                sol = local_search_ultra(sol, self.cfg)

            score = bbox_side(sol)
            if not check_overlaps(sol, self.cfg.collision_buffer) and score < best_score:
                best_score = score
                best_sol = sol

        # Final basin hopping
        if best_sol is not None and time.time() - start < time_budget * 0.9:
            remaining = time_budget - (time.time() - start)
            best_sol = basin_hop(best_sol, self.cfg, remaining * 0.5)
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

    def solve_all(self, max_n: int = 200, verbose: bool = True) -> Dict[int, List[Tuple[float, float, float]]]:
        total_start = time.time()

        if verbose:
            print("=" * 70)
            print("SANTA 2025 - MEGA OPTIMIZATION SOLVER")
            print("=" * 70)
            print(f"Restarts: {self.cfg.num_restarts}")
            print(f"Placement attempts: {self.cfg.num_placement_attempts}")
            print(f"SA iterations: {self.cfg.sa_iterations}")
            print(f"Collision buffer: {self.cfg.collision_buffer}")
            print(f"Compact passes: {self.cfg.compact_passes}")
            print(f"Basin hops: {self.cfg.basin_hops}")
            print()

        for n in range(1, max_n + 1):
            n_start = time.time()

            prev = self.solutions[n - 1] if n > 1 else []
            sol, score = self.solve_single(n, prev, verbose)

            self.solutions[n] = sol
            self.scores[n] = score

            elapsed = time.time() - n_start

            if verbose and (n <= 10 or n % 10 == 0 or n == max_n):
                total = self.total_score()
                print(f"n={n:3d}: side={score:.4f}, time={elapsed:.1f}s, total={total:.2f}")

        if verbose:
            total_time = time.time() - total_start
            total = self.total_score()
            print()
            print("=" * 70)
            print(f"COMPLETE - Time: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"Final Score: {total:.4f}")
            print("=" * 70)

        return self.solutions

    def total_score(self) -> float:
        total = 0.0
        for n, sol in self.solutions.items():
            side = self.scores.get(n, bbox_side(sol))
            total += (side ** 2) / n
        return total


# =============================================================================
# I/O
# =============================================================================

def create_submission(solutions: Dict[int, List[Tuple[float, float, float]]],
                      output: str = "submission.csv") -> str:
    with open(output, "w") as f:
        f.write("id,x,y,deg\n")
        for n in range(1, 201):
            if n not in solutions:
                raise ValueError(f"Missing n={n}")
            pos = solutions[n]
            polys = [make_tree(x, y, d) for x, y, d in pos]
            b = unary_union(polys).bounds
            for idx, (x, y, d) in enumerate(pos):
                f.write(f"{n:03d}_{idx},s{x - b[0]:.6f},s{y - b[1]:.6f},s{d:.6f}\n")
    return output


def validate_all(solutions: Dict[int, List[Tuple[float, float, float]]]) -> bool:
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
    print("=" * 60)
    sides = {n: bbox_side(sol) for n, sol in solutions.items()}
    contribs = {n: (s ** 2) / n for n, s in sides.items()}
    total = sum(contribs.values())

    print(f"Total score: {total:.4f}")
    print(f"Baseline: 157.08")
    print(f"Improvement: {(157.08 - total) / 157.08 * 100:.1f}%")

    worst = sorted(contribs.items(), key=lambda x: -x[1])[:10]
    print("\nTop 10 worst contributions:")
    for n, c in worst:
        print(f"  n={n:3d}: side={sides[n]:.4f}, contrib={c:.4f}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Santa 2025 Mega Solver")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-n", type=int, default=200)
    parser.add_argument("--restarts", type=int, default=200)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = MegaConfig()
    cfg.seed = args.seed
    cfg.num_restarts = args.restarts

    if args.quick:
        cfg.num_restarts = 30
        cfg.num_placement_attempts = 200
        cfg.sa_iterations = 20000
        cfg.time_per_n = 10.0
        cfg.basin_hops = 3

    solver = MegaSolver(config=cfg, seed=args.seed)
    solutions = solver.solve_all(max_n=args.max_n, verbose=True)

    print("\nValidating...")
    if validate_all(solutions):
        print("All solutions valid!")

    print()
    print_summary(solutions)

    print(f"\nSaving: {args.output}")
    create_submission(solutions, args.output)
    print(f"Final Score: {solver.total_score():.4f}")


if __name__ == "__main__":
    main()
