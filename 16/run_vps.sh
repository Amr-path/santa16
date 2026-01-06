#!/bin/bash
#
# Santa 2025 - VPS Run Script
# ===========================
#
# Usage:
#   ./run_vps.sh quick        # Quick test (2-3 hours)
#   ./run_vps.sh standard     # Standard run (24-48 hours)
#   ./run_vps.sh ultra        # Ultra mode (3-7 days)
#   ./run_vps.sh refine       # Refine existing submission.csv
#   ./run_vps.sh continuous   # Run continuous refinement
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Output directory
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

# Timestamp for unique filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check Python dependencies
check_deps() {
    log "Checking dependencies..."
    python3 -c "import numpy, shapely" 2>/dev/null || {
        log "Installing dependencies..."
        pip3 install numpy shapely --quiet
    }
}

# Run quick mode
run_quick() {
    OUTPUT="$OUTPUT_DIR/submission_quick_${TIMESTAMP}.csv"
    log "Starting QUICK mode..."
    log "Output: $OUTPUT"
    log "Estimated time: 2-3 hours"
    echo ""

    python3 ultimate_solver.py --quick --output "$OUTPUT"

    log "Done! Output: $OUTPUT"
}

# Run standard mode
run_standard() {
    OUTPUT="$OUTPUT_DIR/submission_standard_${TIMESTAMP}.csv"
    log "Starting STANDARD mode..."
    log "Output: $OUTPUT"
    log "Estimated time: 24-48 hours"
    echo ""

    nohup python3 ultimate_solver.py --output "$OUTPUT" > "$OUTPUT_DIR/log_standard_${TIMESTAMP}.txt" 2>&1 &
    PID=$!

    log "Running in background with PID: $PID"
    log "Log: $OUTPUT_DIR/log_standard_${TIMESTAMP}.txt"
    log "Monitor with: tail -f $OUTPUT_DIR/log_standard_${TIMESTAMP}.txt"
    echo "$PID" > "$OUTPUT_DIR/pid_standard.txt"
}

# Run ultra mode
run_ultra() {
    OUTPUT="$OUTPUT_DIR/submission_ultra_${TIMESTAMP}.csv"
    log "Starting ULTRA mode..."
    log "Output: $OUTPUT"
    log "Estimated time: 3-7 days"
    echo ""

    nohup python3 ultimate_solver.py --ultra --output "$OUTPUT" > "$OUTPUT_DIR/log_ultra_${TIMESTAMP}.txt" 2>&1 &
    PID=$!

    log "Running in background with PID: $PID"
    log "Log: $OUTPUT_DIR/log_ultra_${TIMESTAMP}.txt"
    log "Monitor with: tail -f $OUTPUT_DIR/log_ultra_${TIMESTAMP}.txt"
    echo "$PID" > "$OUTPUT_DIR/pid_ultra.txt"
}

# Refine existing solution
run_refine() {
    INPUT="${2:-submission.csv}"
    OUTPUT="$OUTPUT_DIR/refined_${TIMESTAMP}.csv"

    if [ ! -f "$INPUT" ]; then
        log "Error: Input file not found: $INPUT"
        log "Usage: ./run_vps.sh refine [input.csv]"
        exit 1
    fi

    log "Starting REFINE mode..."
    log "Input: $INPUT"
    log "Output: $OUTPUT"
    echo ""

    nohup python3 refine_solver.py --input "$INPUT" --output "$OUTPUT" --iterations 10 > "$OUTPUT_DIR/log_refine_${TIMESTAMP}.txt" 2>&1 &
    PID=$!

    log "Running in background with PID: $PID"
    log "Monitor with: tail -f $OUTPUT_DIR/log_refine_${TIMESTAMP}.txt"
    echo "$PID" > "$OUTPUT_DIR/pid_refine.txt"
}

# Continuous refinement
run_continuous() {
    INPUT="${2:-submission.csv}"
    OUTPUT="$OUTPUT_DIR/continuous_${TIMESTAMP}.csv"

    if [ ! -f "$INPUT" ]; then
        log "Error: Input file not found: $INPUT"
        log "Usage: ./run_vps.sh continuous [input.csv]"
        exit 1
    fi

    log "Starting CONTINUOUS refinement..."
    log "Input: $INPUT"
    log "Output: $OUTPUT"
    log "Will run until stopped (Ctrl+C or kill)"
    echo ""

    nohup python3 refine_solver.py --input "$INPUT" --output "$OUTPUT" --continuous > "$OUTPUT_DIR/log_continuous_${TIMESTAMP}.txt" 2>&1 &
    PID=$!

    log "Running in background with PID: $PID"
    log "Monitor with: tail -f $OUTPUT_DIR/log_continuous_${TIMESTAMP}.txt"
    log "Stop with: kill $PID"
    echo "$PID" > "$OUTPUT_DIR/pid_continuous.txt"
}

# Show status
show_status() {
    log "Checking running processes..."
    echo ""

    for pidfile in "$OUTPUT_DIR"/pid_*.txt; do
        if [ -f "$pidfile" ]; then
            PID=$(cat "$pidfile")
            NAME=$(basename "$pidfile" .txt | sed 's/pid_//')
            if ps -p $PID > /dev/null 2>&1; then
                echo "  $NAME: Running (PID $PID)"
            else
                echo "  $NAME: Stopped (PID $PID was)"
                rm -f "$pidfile"
            fi
        fi
    done

    echo ""
    log "Latest outputs:"
    ls -lt "$OUTPUT_DIR"/*.csv 2>/dev/null | head -5 || echo "  No outputs yet"
}

# Show help
show_help() {
    echo "Santa 2025 VPS Runner"
    echo ""
    echo "Usage: ./run_vps.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  quick       Run quick test mode (2-3 hours, foreground)"
    echo "  standard    Run standard mode (24-48 hours, background)"
    echo "  ultra       Run ultra mode (3-7 days, background)"
    echo "  refine      Refine existing solution (background)"
    echo "  continuous  Run continuous refinement (background)"
    echo "  status      Show running processes"
    echo "  help        Show this help"
    echo ""
    echo "Examples:"
    echo "  ./run_vps.sh quick                    # Quick test"
    echo "  ./run_vps.sh standard                 # Standard long run"
    echo "  ./run_vps.sh ultra                    # Ultra long run"
    echo "  ./run_vps.sh refine submission.csv    # Refine existing"
    echo "  ./run_vps.sh continuous submission.csv # Continuous improvement"
    echo "  ./run_vps.sh status                   # Check status"
}

# Main
check_deps

case "${1:-help}" in
    quick)
        run_quick
        ;;
    standard)
        run_standard
        ;;
    ultra)
        run_ultra
        ;;
    refine)
        run_refine "$@"
        ;;
    continuous)
        run_continuous "$@"
        ;;
    status)
        show_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
