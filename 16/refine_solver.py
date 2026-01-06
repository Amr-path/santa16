#!/usr/bin/env python3
"""
Santa 2025 - Refinement Solver
==============================

Specialized solver for refining existing solutions. Instead of building
from scratch, this focuses on improving each puzzle individually.

Key insight: Score = sum(side^2 / n)
- n=1 to n=10 contribute roughly 50% of the score
- Focus most effort on these small puzzles

Strategy:
1. Load existing solution
2. Identify puzzles with highest score contribution
3. Focus optimization time on those puzzles
4. Run continuous improvement loops

Usage:
    # Refine an existing submission
    python refine_solver.py --input submission.csv --output refined.csv

    # Focus on specific puzzles
    python refine_solver.py --input submission.csv --focus 1-20 --output refined.csv

    # Run indefinitely (Ctrl+C to stop)
    python refine_solver.py --input submission.csv --continuous --output refined.csv
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

import numpy as np
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# =============================================================================
# GEOMETRY (same as ultimate_solver.py)
# =============================================================================

TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]

BASE_POLYGON = Polygon(TREE_COORDS)
ROTATED_POLYGONS = {deg: affinity.rotate(BASE_POLYGON, deg, origin=(0, 0))
                    for deg in range(0, 360, 1)}


def make_tree(x: float, y: float, deg: float) -> Polygon:
    snapped = int(round(deg)) % 360
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
# REFINEMENT CONFIG
# =============================================================================

@dataclass
class RefineConfig:
    """Configuration for refinement."""
    # Collision buffer
    collision_buffer: float = 0.01

    # SA parameters
    sa_temp_initial: float = 2.0
    sa_temp_final: float = 0.0000001

    # Time budgets
    time_per_iteration: float = 60.0  # Per puzzle per iteration

    # Local search
    local_precision: float = 0.0003
    local_iterations: int = 3000

    # Compact
    compact_passes: int = 30
    compact_step: float = 0.0001

    seed: int = 42


# =============================================================================
# REFINEMENT ALGORITHMS
# =============================================================================

def refine_sa(placements: List[Tuple[float, float, float]], cfg: RefineConfig,
              time_limit: float, seed: int = 42) -> List[Tuple[float, float, float]]:
    """Simulated Annealing for refinement."""
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
    iters = 0
    accepted = 0

    shift = 0.1 * cur_score
    rot = 30.0

    while time.time() - start < time_limit:
        iters += 1

        i = random.randrange(n)
        x, y, d = current[i]

        # Diverse move types
        mt = random.random()

        if mt < 0.15:
            # Very precise move
            nx = x + random.gauss(0, shift * 0.05)
            ny = y + random.gauss(0, shift * 0.05)
            nd = d
        elif mt < 0.35:
            # Small move
            nx = x + random.gauss(0, shift * 0.2)
            ny = y + random.gauss(0, shift * 0.2)
            nd = d
        elif mt < 0.50:
            # Medium move
            nx = x + random.gauss(0, shift * 0.5)
            ny = y + random.gauss(0, shift * 0.5)
            nd = d
        elif mt < 0.60:
            # Large escape
            nx = x + random.uniform(-shift * 2, shift * 2)
            ny = y + random.uniform(-shift * 2, shift * 2)
            nd = d
        elif mt < 0.75:
            # Fine rotation
            nx, ny = x, y
            nd = (d + random.choice([-1, 1, -2, 2, -3, 3])) % 360
        elif mt < 0.85:
            # Larger rotation
            nx, ny = x, y
            nd = (d + random.choice([-5, 5, -10, 10, -15, 15, -30, 30])) % 360
        elif mt < 0.93:
            # Center seeking
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 0.001
            step = random.uniform(0.005, 0.05) * cur_score
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

        # Adaptive
        if iters % 2000 == 0:
            rate = accepted / iters
            if rate > 0.35:
                shift = min(cur_score * 0.3, shift * 1.1)
            elif rate < 0.1:
                shift = max(0.0005, shift * 0.9)

    return best


def refine_local(placements: List[Tuple[float, float, float]],
                 cfg: RefineConfig) -> List[Tuple[float, float, float]]:
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

            # Very fine moves
            moves = []
            for scale in [0.25, 0.5, 1.0, 2.0]:
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
            for dd in [-1, 1, -2, 2, -3, 3]:
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

            if best_move is not None and best_improvement > 1e-7:
                current[i] = (best_move[0], best_move[1], best_move[2])
                polys[i] = best_move[3]
                cur_score -= best_improvement
                improved = True

    return current


def refine_compact(placements: List[Tuple[float, float, float]],
                   cfg: RefineConfig) -> List[Tuple[float, float, float]]:
    """Compact trees toward center."""
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
            norm = math.sqrt(dx*dx + dy*dy) + 0.001

            for mult in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]:
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


def refine_puzzle(placements: List[Tuple[float, float, float]], cfg: RefineConfig,
                  time_limit: float, seed: int = 42) -> Tuple[List[Tuple[float, float, float]], float]:
    """Full refinement pipeline for a single puzzle."""
    best = list(placements)
    best_score = bbox_side(best)

    # Multiple seeds
    for s in range(3):
        actual_seed = seed + s * 1000

        remaining = time_limit * (1 - s * 0.25)
        if remaining < 1:
            break

        # SA
        result = refine_sa(best, cfg, remaining * 0.6, actual_seed)
        result = refine_local(result, cfg)
        result = refine_compact(result, cfg)
        result = refine_local(result, cfg)

        score = bbox_side(result)
        if not check_overlaps(result, cfg.collision_buffer) and score < best_score:
            best_score = score
            best = result

    return center(best), best_score


# =============================================================================
# I/O
# =============================================================================

def load_submission(path: str) -> Dict[int, List[Tuple[float, float, float]]]:
    """Load solutions from CSV."""
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
    """Save solutions to CSV."""
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
# MAIN REFINER
# =============================================================================

class Refiner:
    """Iterative solution refiner."""

    def __init__(self, solutions: Dict[int, List[Tuple[float, float, float]]],
                 config: Optional[RefineConfig] = None, verbose: bool = True):
        self.solutions = {n: list(sol) for n, sol in solutions.items()}
        self.cfg = config or RefineConfig()
        self.verbose = verbose
        self.running = True

        # Track best score
        self.best_score = compute_score(self.solutions)
        self.improvements = []

        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nReceived stop signal. Finishing current puzzle and saving...")
        self.running = False

    def refine_single(self, n: int, time_limit: float = 60.0) -> bool:
        """Refine a single puzzle. Returns True if improved."""
        if n not in self.solutions:
            return False

        old_score = bbox_side(self.solutions[n])
        old_contrib = (old_score ** 2) / n

        refined, new_score = refine_puzzle(
            self.solutions[n], self.cfg, time_limit,
            seed=self.cfg.seed + n
        )

        new_contrib = (new_score ** 2) / n

        if new_contrib < old_contrib and not check_overlaps(refined, self.cfg.collision_buffer):
            self.solutions[n] = refined
            improvement = old_contrib - new_contrib
            self.improvements.append((n, improvement))
            return True

        return False

    def refine_priority(self, focus_range: Optional[Tuple[int, int]] = None,
                        iterations: int = 1, output_path: Optional[str] = None):
        """
        Refine puzzles by priority (highest contribution first).

        Args:
            focus_range: Optional (start, end) to focus on specific n values
            iterations: Number of full passes
            output_path: Path to save after each iteration
        """
        start_time = time.time()
        start_score = compute_score(self.solutions)

        if self.verbose:
            print("=" * 70)
            print("REFINEMENT SOLVER")
            print("=" * 70)
            print(f"Starting score: {start_score:.4f}")
            print()

        for iteration in range(iterations):
            if not self.running:
                break

            iter_start = time.time()
            iter_improvements = 0

            # Get contributions and sort by priority
            contribs = get_contributions(self.solutions)

            if focus_range:
                puzzles = [(n, c) for n, c in contribs.items()
                           if focus_range[0] <= n <= focus_range[1]]
            else:
                puzzles = list(contribs.items())

            # Sort by contribution (highest first = most impact)
            puzzles.sort(key=lambda x: -x[1])

            if self.verbose:
                print(f"Iteration {iteration + 1}/{iterations}")
                print(f"Puzzles to refine: {len(puzzles)}")

            for idx, (n, contrib) in enumerate(puzzles):
                if not self.running:
                    break

                # More time for high-contribution puzzles
                priority = 1.0 + (contrib / max(c for _, c in puzzles)) * 2.0
                time_limit = self.cfg.time_per_iteration * priority

                improved = self.refine_single(n, time_limit)

                if improved:
                    iter_improvements += 1

                if self.verbose and (idx % 10 == 0 or improved):
                    current = compute_score(self.solutions)
                    elapsed = time.time() - iter_start
                    status = "IMPROVED" if improved else "        "
                    print(f"  n={n:3d}: {status} | score={current:.4f} | "
                          f"elapsed={elapsed:.0f}s")

            # Save after each iteration
            if output_path:
                save_submission(self.solutions, output_path)

            iter_time = time.time() - iter_start
            current_score = compute_score(self.solutions)

            if self.verbose:
                print()
                print(f"Iteration {iteration + 1} complete:")
                print(f"  Improvements: {iter_improvements}")
                print(f"  Score: {current_score:.4f}")
                print(f"  Time: {iter_time:.0f}s")
                print()

        total_time = time.time() - start_time
        final_score = compute_score(self.solutions)

        if self.verbose:
            print("=" * 70)
            print(f"REFINEMENT COMPLETE")
            print(f"  Start score: {start_score:.4f}")
            print(f"  Final score: {final_score:.4f}")
            print(f"  Improvement: {start_score - final_score:.4f} ({(start_score - final_score) / start_score * 100:.2f}%)")
            print(f"  Total time: {total_time / 3600:.2f} hours")
            print("=" * 70)

    def refine_continuous(self, output_path: str):
        """Run continuous refinement until stopped (Ctrl+C)."""
        print("Running continuous refinement. Press Ctrl+C to stop...")
        print()

        iteration = 0
        while self.running:
            iteration += 1
            self.refine_priority(iterations=1, output_path=output_path)

            if self.running:
                score = compute_score(self.solutions)
                print(f"\n--- After iteration {iteration}: score = {score:.4f} ---\n")


def main():
    parser = argparse.ArgumentParser(
        description="Refine existing Santa 2025 submissions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic refinement
  python refine_solver.py --input submission.csv --output refined.csv

  # Focus on puzzles 1-20 (highest impact)
  python refine_solver.py --input submission.csv --focus 1-20 --output refined.csv

  # Run continuously until Ctrl+C
  python refine_solver.py --input submission.csv --continuous --output refined.csv

  # Multiple iterations
  python refine_solver.py --input submission.csv --iterations 5 --output refined.csv
"""
    )
    parser.add_argument("--input", required=True, help="Input submission CSV")
    parser.add_argument("--output", default="refined.csv", help="Output CSV path")
    parser.add_argument("--focus", type=str, help="Focus range, e.g., '1-20'")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--continuous", action="store_true", help="Run until Ctrl+C")
    parser.add_argument("--time", type=float, default=60.0, help="Time per puzzle (seconds)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load existing solution
    print(f"Loading: {args.input}")
    solutions = load_submission(args.input)

    if not solutions:
        print("Error: Could not load solutions")
        return

    initial_score = compute_score(solutions)
    print(f"Loaded {len(solutions)} puzzles, score: {initial_score:.4f}")

    # Parse focus range
    focus_range = None
    if args.focus:
        parts = args.focus.split("-")
        focus_range = (int(parts[0]), int(parts[1]))
        print(f"Focusing on n={focus_range[0]} to n={focus_range[1]}")

    # Config
    cfg = RefineConfig()
    cfg.time_per_iteration = args.time
    cfg.seed = args.seed

    # Create refiner
    refiner = Refiner(solutions, config=cfg, verbose=True)

    if args.continuous:
        refiner.refine_continuous(args.output)
    else:
        refiner.refine_priority(
            focus_range=focus_range,
            iterations=args.iterations,
            output_path=args.output
        )

    # Final save
    save_submission(refiner.solutions, args.output)
    final_score = compute_score(refiner.solutions)
    print(f"\nSaved: {args.output}")
    print(f"Final score: {final_score:.4f}")


if __name__ == "__main__":
    main()
