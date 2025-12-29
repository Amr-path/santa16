# Santa 2025 Christmas Tree Packing Solver v3.0

Highly optimized solver combining the official Kaggle approach with Simulated Annealing.

## Key Features

1. **Official Greedy Placement**: Uses weighted random angles (favoring corners) from official code
2. **Simulated Annealing**: Refines placements after greedy initialization  
3. **Multi-Restart**: Multiple optimization attempts to escape local minima
4. **Exact Tree Polygon**: Uses official 15-vertex tree shape with high precision

## Quick Start

```bash
# Quick mode (~20 min, score ~110-130)
python run.py --mode quick

# Standard mode (~60 min, score ~90-110)
python run.py --mode standard

# Aggressive mode (~2-3 hours, score ~75-90)
python run.py --mode aggressive

# Maximum mode (~5-6 hours, score ~65-75)
python run.py --mode maximum
```

## On Kaggle

1. Upload all files
2. Open `notebooks/main.ipynb`
3. Set `MODE = "standard"` (or other)
4. Run all cells

## Expected Performance

| Mode       | Time      | Score     | Improvement |
|------------|-----------|-----------|-------------|
| quick      | ~20 min   | 110-130   | 17-30%      |
| standard   | ~60 min   | 90-110    | 30-43%      |
| aggressive | ~2-3 hrs  | 75-90     | 43-52%      |
| maximum    | ~5-6 hrs  | 65-75     | 52-59%      |

Baseline: 157.08

## Algorithm

### Phase 1: Greedy Placement (from official code)
- Start new tree at radius=25 from center
- Use weighted random angle favoring corners: `sin(2*angle)`
- Move toward center until collision
- Back up in fine steps until no collision
- Try 20-60 attempts, keep closest to center

### Phase 2: Simulated Annealing
- Random moves: translation + rotation
- Accept worse moves with probability exp(-Δ/T)
- Exponential cooling schedule
- 3000-15000 iterations per n

### Phase 3: Multi-Restart
- Run SA multiple times
- Perturb best solution between runs
- Keep overall best

## Files

```
santa_solver_v3/
├── run.py              # CLI entry point
├── README.md
├── requirements.txt
├── notebooks/
│   └── main.ipynb      # Kaggle notebook
└── src/
    ├── __init__.py
    ├── geometry.py     # Tree polygon (official format)
    ├── packing.py      # Main solver
    ├── validate.py     # Validation
    └── io_utils.py     # I/O (official format)
```

## Requirements

```
numpy>=1.21.0
shapely>=2.0.0
```
