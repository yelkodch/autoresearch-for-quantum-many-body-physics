#!/usr/bin/env bash
# Run the J1-J2 automated recipe-search campaign.
#
# Quick run:   ./run_campaign.sh
# Long run:    ./run_campaign.sh --iterations 20 --max-steps 3000 --max-wall-time 8h
# Resume:      ./run_campaign.sh --resume
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate local venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "No .venv found. Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Force CPU (faster than MPS for this workload on M-series Macs)
export TV_DEVICE=cpu

# Load .env if present (API keys)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Check API keys
if [ -z "${GEMINI_API_KEY:-}" ] && [ -z "${GROQ_API_KEY:-}" ]; then
    echo "ERROR: No API keys found. Set GEMINI_API_KEY or GROQ_API_KEY in .env"
    exit 1
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║  J1-J2 Heisenberg 2D Recipe Search              ║"
echo "║  Lattice: ${TV_LX:-4}×${TV_LY:-4}, Device: $TV_DEVICE              ║"
echo "╚══════════════════════════════════════════════════╝"

# Default: quick mode (5 iterations, 1000 steps per delta, max 4h)
python controller/run_agentic_loop.py \
    --iterations "${ITERATIONS:-5}" \
    --max-steps "${MAX_STEPS:-1000}" \
    --max-wall-time "${MAX_WALL_TIME:-4h}" \
    "$@"
