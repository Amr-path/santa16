#!/usr/bin/env python3
"""
run.py - Santa 2025 Solver v3

Usage:
    python run.py [--mode quick|standard|aggressive|maximum] [--seed 42]

Modes:
    quick:      ~15-25 min, score ~110-130
    standard:   ~45-75 min, score ~90-110
    aggressive: ~2-3 hours, score ~75-90
    maximum:    ~5-6 hours, score ~65-75
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.packing import PackingSolver, SolverConfig
from src.validate import validate_all_solutions, compute_score, print_score_summary
from src.io_utils import create_submission, get_output_path, validate_submission_format


def get_config(mode: str) -> SolverConfig:
    if mode == "quick":
        return SolverConfig.quick_mode()
    elif mode == "standard":
        return SolverConfig.standard_mode()
    elif mode == "aggressive":
        return SolverConfig.aggressive_mode()
    elif mode == "maximum":
        return SolverConfig.maximum_mode()
    return SolverConfig.standard_mode()


def main(mode: str = "standard", seed: int = 42, max_n: int = 200):
    print("=" * 70)
    print("SANTA 2025 CHRISTMAS TREE PACKING SOLVER v3.0")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Seed: {seed}")
    print(f"Max N: {max_n}")
    print()
    
    config = get_config(mode)
    config.seed = seed
    
    print("Configuration:")
    print(f"  Placement attempts: {config.num_placement_attempts}")
    print(f"  SA iterations: {config.sa_iterations_base}")
    print(f"  SA restarts: {config.num_restarts}")
    print()
    
    start_time = time.time()
    
    solver = PackingSolver(config=config, seed=seed)
    solutions = solver.solve_all(max_n=max_n, verbose=True)
    
    solve_time = time.time() - start_time
    print(f"\nSolving time: {solve_time:.1f}s ({solve_time/60:.1f} min)")
    
    # Validate
    print("\nValidating...")
    validate_all_solutions(solutions, max_n=max_n, verbose=True)
    
    # Score summary
    print()
    print_score_summary(solutions, max_n=max_n)
    
    # Create submission
    print("\nCreating submission...")
    output_path = get_output_path("submission.csv")
    
    try:
        created = create_submission(solutions, output_path)
        print(f"✓ Saved: {created}")
        
        valid, error = validate_submission_format(created)
        if valid:
            print("✓ Format OK")
        else:
            print(f"⚠ Format issue: {error}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Final summary
    total_time = time.time() - start_time
    total_score = compute_score(solutions, max_n)
    
    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Score: {total_score:.4f}")
    print(f"Baseline: 157.08")
    print(f"Improvement: {(157.08 - total_score) / 157.08 * 100:.1f}%")
    print(f"Output: {output_path}")
    
    return solutions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Santa 2025 Solver v3")
    parser.add_argument("--mode", choices=["quick", "standard", "aggressive", "maximum"], default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-n", type=int, default=200)
    args = parser.parse_args()
    
    main(mode=args.mode, seed=args.seed, max_n=args.max_n)
