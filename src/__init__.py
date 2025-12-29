"""
Santa 2025 Christmas Tree Packing Solver v3

Combines:
- Official Kaggle greedy radial placement with weighted angles
- Simulated Annealing optimization
- Multiple restart capability
- High-precision geometry
"""

from .geometry import (
    make_tree_polygon,
    transform_tree,
    compute_bounding_square_side,
    check_all_overlaps,
    center_placements,
)

from .packing import (
    PackingSolver,
    SolverConfig,
    OptimizationConfig,
)

from .validate import (
    validate_solution,
    validate_all_solutions,
    compute_score,
    print_score_summary,
)

from .io_utils import (
    create_submission,
    print_solution_summary,
    get_output_path,
)

__version__ = "3.0.0"
