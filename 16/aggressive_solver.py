#!/usr/bin/env python3
"""
Santa 2025 - AGGRESSIVE Solver
==============================
Rebuilds solutions from scratch with heavy optimization.
Targets score < 65.
"""

import math
import time
import random
import argparse
from typing import List, Tuple, Dict
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
from shapely.ops import unary_union

# Tree polygon
TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5), (0.2, 0.25), (0.1, 0.25),
    (0.35, 0.0), (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2), (-0.075, 0.0),
    (-0.35, 0.0), (-0.1, 0.25), (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5),
]
BASE_POLYGON = Polygon(TREE_COORDS)
ROTATED = {deg: affinity.rotate(BASE_POLYGON, deg, origin=(0, 0)) for deg in range(0, 360, 5)}


def make_tree(x, y, deg):
    snapped = int(round(deg / 5) * 5) % 360
    return affinity.translate(ROTATED[snapped], xoff=x, yoff=y) if x != 0 or y != 0 else ROTATED[snapped]


def bbox_side(placements):
    if not placements:
        return 0.0
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    return max(b[2] - b[0], b[3] - b[1])


def has_collision(poly, others, buf=0.01):
    for o in others:
        if poly.distance(o) < buf:
            return True
    return False


def center_solution(placements):
    if not placements:
        return placements
    polys = [make_tree(x, y, d) for x, y, d in placements]
    b = unary_union(polys).bounds
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return [(x - cx, y - cy, d) for x, y, d in placements]


def place_tree_aggressive(base_poly, existing, attempts=100):
    """Aggressive placement with many attempts."""
    if not existing:
        return 0.0, 0.0

    idx = STRtree(existing)
    best_x, best_y = 0.0, 0.0
    best_r = float('inf')

    for _ in range(attempts):
        while True:
            angle = random.uniform(0, 2 * math.pi)
            if random.random() < abs(math.sin(2 * angle)):
                break

        vx, vy = math.cos(angle), math.sin(angle)

        r = 12.0
        while r >= 0:
            px, py = r * vx, r * vy
            cand = affinity.translate(base_poly, xoff=px, yoff=py)

            collides = False
            for i in idx.query(cand):
                if cand.distance(existing[i]) < 0.01:
                    collides = True
                    break

            if collides:
                for r2 in [r + 0.01 * i for i in range(1, 200)]:
                    px2, py2 = r2 * vx, r2 * vy
                    cand2 = affinity.translate(base_poly, xoff=px2, yoff=py2)
                    ok = True
                    for i in idx.query(cand2):
                        if cand2.distance(existing[i]) < 0.01:
                            ok = False
                            break
                    if ok:
                        if r2 < best_r:
                            best_r = r2
                            best_x, best_y = px2, py2
                        break
                break
            r -= 0.15

    return best_x, best_y


def simulated_annealing_aggressive(placements, time_limit=30, temp_init=2.0):
    """Aggressive SA with time limit."""
    n = len(placements)
    if n <= 1:
        return placements

    start = time.time()
    current = list(placements)
    polys = [make_tree(x, y, d) for x, y, d in current]

    cur_score = bbox_side(current)
    best = list(current)
    best_score = cur_score

    T = temp_init
    iters = 0

    while time.time() - start < time_limit:
        iters += 1
        i = random.randrange(n)
        x, y, d = current[i]

        r = random.random()
        if r < 0.4:
            nx = x + random.gauss(0, 0.05 * cur_score)
            ny = y + random.gauss(0, 0.05 * cur_score)
            nd = d
        elif r < 0.6:
            nx = x + random.gauss(0, 0.15 * cur_score)
            ny = y + random.gauss(0, 0.15 * cur_score)
            nd = d
        elif r < 0.8:
            nx, ny = x, y
            nd = (d + random.choice([-5, 5, -10, 10, -15, 15, -30, 30])) % 360
        else:
            cx = sum(p[0] for p in current) / n
            cy = sum(p[1] for p in current) / n
            dx, dy = cx - x, cy - y
            dist = math.sqrt(dx*dx + dy*dy) + 0.001
            step = random.uniform(0.01, 0.08) * cur_score
            nx = x + step * dx / dist
            ny = y + step * dy / dist
            nd = d

        new_poly = make_tree(nx, ny, nd)
        others = polys[:i] + polys[i+1:]

        if has_collision(new_poly, others):
            T *= 0.9999
            continue

        old_poly = polys[i]
        polys[i] = new_poly

        test_placements = list(current)
        test_placements[i] = (nx, ny, nd)
        new_score = bbox_side(test_placements)

        delta = new_score - cur_score
        if delta < 0 or (T > 0.001 and random.random() < math.exp(-delta / T)):
            current[i] = (nx, ny, nd)
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best = list(current)
        else:
            polys[i] = old_poly

        T *= 0.9999

    return best


def compact_solution(placements, passes=20):
    """Compact trees toward center."""
    n = len(placements)
    if n <= 1:
        return placements

    current = list(placements)

    for _ in range(passes):
        polys = [make_tree(x, y, d) for x, y, d in current]
        cx = sum(p[0] for p in current) / n
        cy = sum(p[1] for p in current) / n

        dists = [(i, math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)) for i, p in enumerate(current)]
        dists.sort(key=lambda x: -x[1])

        for idx, dist in dists:
            if dist < 0.02:
                continue
            x, y, d = current[idx]
            dx, dy = cx - x, cy - y
            norm = math.sqrt(dx*dx + dy*dy) + 0.001

            for mult in [0.1, 0.05, 0.02, 0.01, 0.005]:
                move = mult * dist
                nx = x + move * dx / norm
                ny = y + move * dy / norm

                new_poly = make_tree(nx, ny, d)
                others = polys[:idx] + polys[idx+1:]

                if not has_collision(new_poly, others):
                    current[idx] = (nx, ny, d)
                    polys[idx] = new_poly
                    break

    return current


def solve_single(n, prev_solution, sa_time=30, attempts=100):
    """Solve a single puzzle."""
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    prev_polys = [make_tree(x, y, d) for x, y, d in prev_solution]

    best_sol = None
    best_score = float('inf')

    # Try multiple angles
    for angle in range(0, 360, 15):
        base = ROTATED[angle]
        px, py = place_tree_aggressive(base, prev_polys, attempts)
        sol = prev_solution + [(px, py, float(angle))]

        # SA refinement
        sol = simulated_annealing_aggressive(sol, time_limit=sa_time/24)
        sol = compact_solution(sol, passes=10)

        score = bbox_side(sol)
        if score < best_score:
            best_score = score
            best_sol = sol

    # Full SA on best
    if best_sol:
        best_sol = simulated_annealing_aggressive(best_sol, time_limit=sa_time*0.7)
        best_sol = compact_solution(best_sol, passes=20)
        best_sol = simulated_annealing_aggressive(best_sol, time_limit=sa_time*0.2)

    return center_solution(best_sol) if best_sol else prev_solution + [(0.0, 0.0, 0.0)]


def load_submission(path):
    solutions = {}
    with open(path, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            n = int(parts[0].split('_')[0])
            x = float(parts[1].lstrip('s'))
            y = float(parts[2].lstrip('s'))
            deg = float(parts[3].lstrip('s'))
            if n not in solutions:
                solutions[n] = []
            solutions[n].append((x, y, deg))
    return solutions


def save_submission(solutions, output):
    with open(output, 'w') as f:
        f.write("id,x,y,deg\n")
        for n in sorted(solutions.keys()):
            pos = solutions[n]
            polys = [make_tree(x, y, d) for x, y, d in pos]
            b = unary_union(polys).bounds
            for idx, (x, y, d) in enumerate(pos):
                f.write(f"{n:03d}_{idx},s{x - b[0]:.6f},s{y - b[1]:.6f},s{d:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Aggressive Santa Solver")
    parser.add_argument("--input", help="Input submission to improve")
    parser.add_argument("--output", default="aggressive_output.csv")
    parser.add_argument("--max-n", type=int, default=200)
    parser.add_argument("--sa-time", type=float, default=60, help="SA time per puzzle (seconds)")
    parser.add_argument("--attempts", type=int, default=150, help="Placement attempts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("AGGRESSIVE SOLVER")
    print("=" * 60)
    print(f"SA time per puzzle: {args.sa_time}s")
    print(f"Placement attempts: {args.attempts}")
    print()

    # Load existing or start fresh
    if args.input:
        existing = load_submission(args.input)
        existing_score = sum((bbox_side(s)**2)/n for n, s in existing.items())
        print(f"Loaded {args.input} with score: {existing_score:.4f}")
    else:
        existing = {}

    solutions = {}
    total_start = time.time()

    for n in range(1, args.max_n + 1):
        n_start = time.time()

        if n == 1:
            sol = [(0.0, 0.0, 0.0)]
        else:
            prev = solutions[n - 1]
            sol = solve_single(n, prev, args.sa_time, args.attempts)

        # Compare with existing
        if n in existing:
            existing_side = bbox_side(existing[n])
            new_side = bbox_side(sol)
            if existing_side < new_side:
                sol = existing[n]

        solutions[n] = sol
        elapsed = time.time() - n_start

        if n <= 10 or n % 10 == 0 or n == args.max_n:
            side = bbox_side(sol)
            total = sum((bbox_side(solutions[i])**2)/i for i in range(1, n+1))
            print(f"n={n:3d}: side={side:.4f}, time={elapsed:.1f}s, total={total:.2f}")

        # Save periodically
        if n % 20 == 0:
            save_submission(solutions, args.output)

    total_time = time.time() - total_start
    total_score = sum((bbox_side(s)**2)/n for n, s in solutions.items())

    print()
    print("=" * 60)
    print(f"DONE in {total_time/60:.1f} min")
    print(f"Final Score: {total_score:.4f}")
    print("=" * 60)

    save_submission(solutions, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
