# Strategy: Getting from 112 to Below 65

## Score Analysis

The score formula is: `score = Σ(side² / n)` for n=1 to 200

**Key insight**: Smaller puzzles contribute MORE to the score!
- n=1: contributes `side²` (huge impact)
- n=2: contributes `side² / 2`
- n=10: contributes `side² / 10`
- n=200: contributes `side² / 200` (minimal impact)

### Score Distribution Example
If your current score is ~112, approximately:
- n=1 to n=10: ~35-40% of total score
- n=11 to n=50: ~30-35%
- n=51 to n=100: ~15-20%
- n=101 to n=200: ~10-15%

## Recommended Strategy

### Phase 1: Quick Test (2-3 hours)
First, verify the solver works with a quick run:

```bash
cd /home/user/santa16/16
chmod +x run_vps.sh
./run_vps.sh quick
```

This should produce a score around 85-95.

### Phase 2: Initial Long Run (24-48 hours)
Run the standard solver for a full day:

```bash
./run_vps.sh standard
```

This runs in background. Monitor with:
```bash
tail -f output/log_standard_*.txt
```

Expected score: 65-75

### Phase 3: Refinement (focus on low-n puzzles)
Once you have a solution around 70-75, refine it:

```bash
# Focus on puzzles 1-20 (highest impact)
python3 refine_solver.py --input output/submission_standard_*.csv \
    --focus 1-20 --iterations 5 --output refined.csv
```

### Phase 4: Ultra Mode (if needed)
If still above 65, run ultra mode:

```bash
./run_vps.sh ultra
```

This can run for 3-7 days.

### Phase 5: Continuous Improvement
Run continuous refinement until you reach your goal:

```bash
./run_vps.sh continuous refined.csv
```

This runs until you stop it (Ctrl+C or `kill <PID>`).

## Optimization Tips

### 1. Focus on Small Puzzles First
The puzzles n=1 to n=10 contribute ~40% of your score.
For a target of 65:
- n=1 should have side < 1.0 (contributes ~1.0 to score)
- n=2 should have side < 1.2 (contributes ~0.7)
- n=5 should have side < 1.8 (contributes ~0.65)
- n=10 should have side < 2.5 (contributes ~0.6)

### 2. Monitor Progress
Check your score breakdown:

```python
python3 -c "
from ultimate_solver import load_submission, bbox_side

sols = load_submission('submission.csv')
for n in range(1, 21):
    side = bbox_side(sols[n])
    contrib = (side ** 2) / n
    print(f'n={n:2d}: side={side:.4f}, contrib={contrib:.4f}')

total = sum((bbox_side(s) ** 2) / n for n, s in sols.items())
print(f'Total: {total:.4f}')
"
```

### 3. Multiple Seeds
Run multiple times with different seeds for diversity:

```bash
# Run 3 parallel instances with different seeds
nohup python3 ultimate_solver.py --seed 42 --output sub_42.csv &
nohup python3 ultimate_solver.py --seed 123 --output sub_123.csv &
nohup python3 ultimate_solver.py --seed 456 --output sub_456.csv &
```

Then pick the best one.

### 4. VPS Resource Management
- Use `htop` to monitor CPU usage
- The solver auto-detects cores and uses all but one
- If you have other processes, limit workers: `--workers 8`

## Expected Timeline

| Phase | Duration | Expected Score |
|-------|----------|----------------|
| Quick test | 2-3 hours | 85-95 |
| Standard | 24-48 hours | 65-75 |
| Refinement | 12-24 hours | 60-68 |
| Ultra | 3-7 days | 55-63 |
| Continuous | Until stopped | < 55 possible |

## Troubleshooting

### Process died?
Check if the process is still running:
```bash
./run_vps.sh status
```

### Out of memory?
Reduce workers:
```bash
python3 ultimate_solver.py --workers 4 --output submission.csv
```

### Score not improving?
1. Check the log for errors
2. Try different seeds
3. Focus refinement on specific puzzle ranges:
   ```bash
   python3 refine_solver.py --input submission.csv --focus 1-10 --output refined.csv
   ```

## Quick Commands Reference

```bash
# Start quick test
./run_vps.sh quick

# Start standard (background)
./run_vps.sh standard

# Start ultra (background)
./run_vps.sh ultra

# Check status
./run_vps.sh status

# Monitor logs
tail -f output/log_*.txt

# Stop a background process
kill $(cat output/pid_standard.txt)

# Check score of a submission
python3 -c "
from ultimate_solver import load_submission, bbox_side
sols = load_submission('submission.csv')
print(sum((bbox_side(s)**2)/n for n,s in sols.items()))
"
```

Good luck reaching below 65!
