#!/usr/bin/env bash
# ============================================================================
# FramerAI - From-scratch training pipeline
#
# This script:
#   1. Installs Python dependencies
#   2. Builds, trains, and exports the model from your local data
#   3. Starts the backend + frontend
#
# Usage:
#   chmod +x train.sh
#   ./train.sh                       # Full pipeline (tiny model, ./data corpus)
#   ./train.sh --size small          # Larger model
#   ./train.sh --train-modalities    # Also train image and audio generators
#   ./train.sh --help                # Show all options
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# ---------------------------------------------------------------------------
# Defaults (override via CLI flags or environment variables)
# ---------------------------------------------------------------------------
MODEL_SIZE="${MODEL_SIZE:-tiny}"
MAX_STEPS="${MAX_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
DATA_DIR="${DATA_DIR:-data}"
DEVICE="${DEVICE:-auto}"
ROPE_SCALING="${ROPE_SCALING:-}"
TRAIN_MODALITIES="${TRAIN_MODALITIES:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_SERVE="${SKIP_SERVE:-0}"

print_help() {
    cat <<'HELP'
FramerAI Training Pipeline

Usage: ./train.sh [OPTIONS]

Model options:
  --size SIZE            Model size: tiny, small, medium, large (default: tiny)
  --device DEVICE        Device: auto, cpu, cuda, mps (default: auto)
  --rope-scaling F       RoPE scaling factor for extended context

Training options:
  --max-steps N          Training steps (default: 1000)
  --batch-size N         Batch size (default: 8)
  --lr RATE              Learning rate (default: 3e-4)
  --data-dir DIR         Local training data directory (default: data)
  --train-modalities     Also train the image and audio generators on caption pairs

Paths:
  --output-dir DIR       Checkpoint directory (default: checkpoints)

Skip stages:
  --skip-install         Skip pip install
  --skip-train           Skip build/train/export
  --skip-serve           Skip starting backend/frontend

  --help                 Show this help
HELP
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size)             MODEL_SIZE="$2"; shift 2 ;;
        --device)           DEVICE="$2"; shift 2 ;;
        --rope-scaling)     ROPE_SCALING="$2"; shift 2 ;;
        --max-steps)        MAX_STEPS="$2"; shift 2 ;;
        --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
        --lr)               LEARNING_RATE="$2"; shift 2 ;;
        --data-dir)         DATA_DIR="$2"; shift 2 ;;
        --train-modalities) TRAIN_MODALITIES=1; shift ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --skip-install)     SKIP_INSTALL=1; shift ;;
        --skip-train)       SKIP_TRAIN=1; shift ;;
        --skip-serve)       SKIP_SERVE=1; shift ;;
        --help|-h)          print_help ;;
        *) echo "Unknown option: $1"; echo "Run ./train.sh --help for usage"; exit 1 ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

step_num=0
log_step() {
    step_num=$((step_num + 1))
    echo ""
    echo -e "${CYAN}${BOLD}============================================================${NC}"
    echo -e "${CYAN}${BOLD}  STEP ${step_num}: $1${NC}"
    echo -e "${CYAN}${BOLD}============================================================${NC}"
    echo ""
}

log_info()   { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()   { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_config() { echo -e "  ${BOLD}$1:${NC} $2"; }

echo ""
echo -e "${BOLD}FramerAI Training Pipeline${NC}"
echo "=========================================="
log_config "Model size"    "$MODEL_SIZE"
log_config "Max steps"     "$MAX_STEPS"
log_config "Batch size"    "$BATCH_SIZE"
log_config "Learning rate" "$LEARNING_RATE"
log_config "Data dir"      "$DATA_DIR"
log_config "Output dir"    "$OUTPUT_DIR"
log_config "Device"        "$DEVICE"
if [[ "$TRAIN_MODALITIES" == "1" ]]; then
    log_config "Modalities" "image + audio generators"
fi
echo "=========================================="

# ---------------------------------------------------------------------------
# STEP 1: Install dependencies
# ---------------------------------------------------------------------------
if [[ "$SKIP_INSTALL" == "0" ]]; then
    log_step "Installing Python dependencies"
    if [[ ! -d "venv" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    log_info "Dependencies installed."
else
    log_info "Skipping install (--skip-install)"
    if [[ -d "venv" ]]; then source venv/bin/activate; fi
fi

if [[ -d "venv" ]] && [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source venv/bin/activate
fi

# ---------------------------------------------------------------------------
# STEP 2: Build, train, and export the model
# ---------------------------------------------------------------------------
if [[ "$SKIP_TRAIN" == "0" ]]; then
    log_step "Building and training FramerAI (size=$MODEL_SIZE) from '$DATA_DIR'"

    TRAIN_CMD="python build.py --mode all --size $MODEL_SIZE"
    TRAIN_CMD+=" --max-steps $MAX_STEPS"
    TRAIN_CMD+=" --batch-size $BATCH_SIZE"
    TRAIN_CMD+=" --lr $LEARNING_RATE"
    TRAIN_CMD+=" --data-dir $DATA_DIR"
    TRAIN_CMD+=" --output-dir $OUTPUT_DIR"
    if [[ "$DEVICE" != "auto" ]]; then TRAIN_CMD+=" --device $DEVICE"; fi
    if [[ -n "$ROPE_SCALING" ]]; then TRAIN_CMD+=" --rope-scaling $ROPE_SCALING"; fi
    if [[ "$TRAIN_MODALITIES" == "1" ]]; then TRAIN_CMD+=" --train-modalities"; fi

    log_info "Running: $TRAIN_CMD"
    eval "$TRAIN_CMD"
    log_info "Model built, trained, and exported to $OUTPUT_DIR/export/"

    if [[ -f "$OUTPUT_DIR/export/model_info.json" ]]; then
        python3 -c "
import json
with open('$OUTPUT_DIR/export/model_info.json') as f:
    info = json.load(f)
print(f'  Name: {info[\"model_name\"]}')
print(f'  Parameters: {info[\"parameters\"]:,}')
print(f'  Modalities: {\", \".join(info[\"modalities\"])}')
"
    fi
else
    log_info "Skipping training (--skip-train)"
fi

# ---------------------------------------------------------------------------
# STEP 3: Start backend + frontend
# ---------------------------------------------------------------------------
if [[ "$SKIP_SERVE" == "0" ]]; then
    log_step "Starting backend and frontend"

    if [[ -d "backend" ]]; then
        log_info "Installing backend dependencies..."
        (cd backend && npm install --silent 2>/dev/null)
        if [[ -f "backend/.env.example" ]] && [[ ! -f "backend/.env" ]]; then
            cp backend/.env.example backend/.env
            log_info "Created backend/.env from .env.example"
        fi
        log_info "Starting backend on port 3001..."
        (cd backend && npm run dev &) 2>/dev/null
    fi

    if [[ -d "website" ]]; then
        log_info "Installing frontend dependencies..."
        (cd website && npm install --silent 2>/dev/null)
        log_info "Starting frontend on port 5173..."
        (cd website && npm run dev &) 2>/dev/null
    fi

    echo ""
    echo -e "${GREEN}${BOLD}============================================================${NC}"
    echo -e "${GREEN}${BOLD}  FramerAI is ready!${NC}"
    echo -e "${GREEN}${BOLD}============================================================${NC}"
    echo ""
    echo -e "  Backend API:  ${CYAN}http://localhost:3001${NC}"
    echo -e "  Frontend:     ${CYAN}http://localhost:5173${NC}"
    echo -e "  Model:        ${CYAN}$OUTPUT_DIR/export/framerai_model.pt${NC}"
    echo ""
    echo "  Press Ctrl+C to stop all servers."
    echo ""
    wait
else
    log_info "Skipping serve (--skip-serve)"
    echo ""
    echo -e "  Model:  ${CYAN}$OUTPUT_DIR/export/framerai_model.pt${NC}"
    echo "  To start serving:  cd backend && npm run dev &   |   cd website && npm run dev &"
    echo ""
fi
