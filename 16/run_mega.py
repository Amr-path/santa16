#!/usr/bin/env python3
"""
Runner for MEGA Solver - Targets score < 55

Usage:
    python run_mega.py                    # Full optimization (~2-3 hours)
    python run_mega.py --quick            # Quick test (~30 min)
    python run_mega.py --turbo            # Maximum optimization (~4-6 hours)
"""

import argparse
import time
from mega_solver import MegaSolver, MegaConfig, create_submission, validate_all, print_summary


def main():
    parser = argparse.ArgumentParser(description="MEGA Solver Runner")
    parser.add_argument("--output", default="mega_submission.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Quick mode (~30 min)")
    parser.add_argument("--turbo", action="store_true", help="Turbo mode (~4-6 hours)")
    args = parser.parse_args()

    cfg = MegaConfig()
    cfg.seed = args.seed

    if args.quick:
        print("Mode: QUICK (targeting ~30 minutes)")
        cfg.num_restarts = 50
        cfg.num_placement_attempts = 300
        cfg.sa_iterations = 30000
        cfg.time_per_n = 8.0
        cfg.basin_hops = 3
        cfg.compact_passes = 10
        cfg.local_iterations = 500
    elif args.turbo:
        print("Mode: TURBO (targeting 4-6 hours)")
        cfg.num_restarts = 500
        cfg.num_placement_attempts = 2000
        cfg.sa_iterations = 200000
        cfg.time_per_n = 120.0
        cfg.basin_hops = 20
        cfg.compact_passes = 50
        cfg.local_iterations = 5000
        cfg.collision_buffer = 0.008  # Tighter packing
    else:
        print("Mode: STANDARD (targeting 2-3 hours)")
        cfg.num_restarts = 200
        cfg.num_placement_attempts = 1000
        cfg.sa_iterations = 100000
        cfg.time_per_n = 45.0
        cfg.basin_hops = 10
        cfg.compact_passes = 20

    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print()

    start = time.time()

    solver = MegaSolver(config=cfg, seed=args.seed)
    solutions = solver.solve_all(max_n=200, verbose=True)

    print("\nValidating...")
    if validate_all(solutions):
        print("All solutions valid!")
    else:
        print("WARNING: Some solutions have issues!")

    print()
    print_summary(solutions)

    print(f"\nSaving: {args.output}")
    create_submission(solutions, args.output)

    total_time = time.time() - start
    final_score = solver.total_score()

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Score: {final_score:.4f}")
    print(f"Time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Output: {args.output}")

    if final_score < 55:
        print("\n*** TARGET ACHIEVED: Score < 55! ***")
    elif final_score < 60:
        print("\n** GOOD: Score < 60 **")
    elif final_score < 80:
        print("\n* DECENT: Score < 80 *")
    else:
        print("\nNeeds more optimization...")


if __name__ == "__main__":
    main()
