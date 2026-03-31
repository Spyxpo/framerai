#!/usr/bin/env bash
# ============================================================================
# FramerAI — Full Training Pipeline
#
# This script automates the entire process:
#   1. Install Python dependencies
#   2. Build the base FramerAI model
#   3. Generate training data from all open-source teacher models
#   4. Distill knowledge from all teachers into FramerAI
#   5. Export the final model for serving
#   6. Start the backend + frontend
#
# Usage:
#   chmod +x train.sh
#   ./train.sh                  # Full pipeline with defaults (large model, all teachers)
#   ./train.sh --size small     # Quick run with smaller model
#   ./train.sh --help           # Show all options
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Load .env file if it exists (for HF_TOKEN, etc.)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# ---------------------------------------------------------------------------
# Defaults (override via CLI flags or environment variables)
# ---------------------------------------------------------------------------
MODEL_SIZE="${MODEL_SIZE:-large}"
TEACHER_MODELS="${TEACHER_MODELS:-all}"
QUANTIZE="${QUANTIZE:-4bit}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
NUM_CONVERSATIONS="${NUM_CONVERSATIONS:-500}"
DISTILL_STEPS="${DISTILL_STEPS:-200000}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
DATA_DIR="${DATA_DIR:-data/distillation}"
DEVICE="${DEVICE:-auto}"
TEACHER_DEVICE="${TEACHER_DEVICE:-auto}"
TEACHER_DTYPE="${TEACHER_DTYPE:-auto}"
ROPE_SCALING="${ROPE_SCALING:-}"
TARGET_SEQ_LEN="${TARGET_SEQ_LEN:-}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_DISTILL="${SKIP_DISTILL:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_SERVE="${SKIP_SERVE:-0}"
HF_TOKEN="${HF_TOKEN:-}"

# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
print_help() {
    cat <<'HELP'
FramerAI Training Pipeline

Usage: ./train.sh [OPTIONS]

Model options:
  --size SIZE              Model size: tiny, small, medium, large (default: large)
  --teacher-model MODELS   Teacher(s): all, all-small, all-medium, all-large,
                           or comma-separated list (default: all)
  --quantize Q             Teacher quantization: 4bit, 8bit, none (default: 4bit)
  --device DEVICE          Student device: auto, cpu, cuda, mps (default: auto)
  --teacher-device DEVICE  Teacher device (default: auto)
  --teacher-dtype DTYPE    Teacher dtype: auto, float16, bfloat16 (default: auto)

Training options:
  --num-samples N          Training samples to generate (default: 10000)
  --num-conversations N    Conversations to generate (default: 500)
  --pretrain-steps N       Pretrain steps before distillation (default: 5000)
  --distill-steps N        Distillation training steps (default: 200000)
  --batch-size N           Batch size (default: 4)
  --lr RATE                Learning rate (default: 1e-4)
  --rope-scaling F         RoPE scaling factor for extended context
  --target-seq-len N       Target sequence length (use with --rope-scaling)

Paths:
  --output-dir DIR         Checkpoint directory (default: checkpoints)
  --data-dir DIR           Training data directory (default: data/distillation)
  --hf-token TOKEN         HuggingFace token (or set HF_TOKEN env var)

Skip stages:
  --skip-install           Skip pip install
  --skip-build             Skip model build
  --skip-generate          Skip data generation
  --skip-distill           Skip distillation training
  --skip-export            Skip model export
  --skip-serve             Skip starting backend/frontend

  --help                   Show this help
HELP
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size)             MODEL_SIZE="$2"; shift 2 ;;
        --teacher-model)    TEACHER_MODELS="$2"; shift 2 ;;
        --quantize)         QUANTIZE="$2"; shift 2 ;;
        --device)           DEVICE="$2"; shift 2 ;;
        --teacher-device)   TEACHER_DEVICE="$2"; shift 2 ;;
        --teacher-dtype)    TEACHER_DTYPE="$2"; shift 2 ;;
        --num-samples)      NUM_SAMPLES="$2"; shift 2 ;;
        --num-conversations) NUM_CONVERSATIONS="$2"; shift 2 ;;
        --pretrain-steps)   PRETRAIN_STEPS="$2"; shift 2 ;;
        --distill-steps)    DISTILL_STEPS="$2"; shift 2 ;;
        --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
        --lr)               LEARNING_RATE="$2"; shift 2 ;;
        --rope-scaling)     ROPE_SCALING="$2"; shift 2 ;;
        --target-seq-len)   TARGET_SEQ_LEN="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --data-dir)         DATA_DIR="$2"; shift 2 ;;
        --hf-token)         HF_TOKEN="$2"; shift 2 ;;
        --skip-install)     SKIP_INSTALL=1; shift ;;
        --skip-build)       SKIP_BUILD=1; shift ;;
        --skip-generate)    SKIP_GENERATE=1; shift ;;
        --skip-distill)     SKIP_DISTILL=1; shift ;;
        --skip-export)      SKIP_EXPORT=1; shift ;;
        --skip-serve)       SKIP_SERVE=1; shift ;;
        --help|-h)          print_help ;;
        *) echo "Unknown option: $1"; echo "Run ./train.sh --help for usage"; exit 1 ;;
    esac
done

# Export HF_TOKEN if provided
if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN
fi

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
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

log_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_config()  { echo -e "  ${BOLD}$1:${NC} $2"; }

# ---------------------------------------------------------------------------
# Print configuration
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}FramerAI Training Pipeline${NC}"
echo "=========================================="
log_config "Model size"       "$MODEL_SIZE"
log_config "Teacher models"   "$TEACHER_MODELS"
log_config "Quantization"     "$QUANTIZE"
log_config "Num samples"      "$NUM_SAMPLES"
log_config "Conversations"    "$NUM_CONVERSATIONS"
log_config "Pretrain steps"   "$PRETRAIN_STEPS"
log_config "Distill steps"    "$DISTILL_STEPS"
log_config "Batch size"       "$BATCH_SIZE"
log_config "Learning rate"    "$LEARNING_RATE"
log_config "Device"           "$DEVICE"
log_config "Teacher device"   "$TEACHER_DEVICE"
log_config "Output dir"       "$OUTPUT_DIR"
log_config "Data dir"         "$DATA_DIR"
if [[ -n "$ROPE_SCALING" ]]; then
    log_config "RoPE scaling" "$ROPE_SCALING"
fi
if [[ -n "$TARGET_SEQ_LEN" ]]; then
    log_config "Target seq len" "$TARGET_SEQ_LEN"
fi
if [[ -n "$HF_TOKEN" ]]; then
    log_config "HF Token"     "set (${#HF_TOKEN} chars)"
else
    log_warn "HF_TOKEN not set — gated models (Llama) will fail. Set with --hf-token or export HF_TOKEN=hf_..."
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

    log_info "Activating virtual environment..."
    source venv/bin/activate

    log_info "Upgrading pip..."
    pip install --upgrade pip --quiet

    log_info "Installing requirements..."
    pip install -r requirements.txt --quiet

    log_info "Dependencies installed."
else
    log_info "Skipping install (--skip-install)"
    if [[ -d "venv" ]]; then
        source venv/bin/activate
    fi
fi

# Ensure venv is active for remaining steps
if [[ -d "venv" ]] && [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source venv/bin/activate
fi

# ---------------------------------------------------------------------------
# STEP 2: Build the base model
# ---------------------------------------------------------------------------
if [[ "$SKIP_BUILD" == "0" ]]; then
    log_step "Building FramerAI base model (size=$MODEL_SIZE)"

    BUILD_CMD="python build.py --mode all --size $MODEL_SIZE"
    BUILD_CMD+=" --max-steps $PRETRAIN_STEPS"
    BUILD_CMD+=" --output-dir $OUTPUT_DIR"
    if [[ "$DEVICE" != "auto" ]]; then
        BUILD_CMD+=" --device $DEVICE"
    fi

    log_info "Running: $BUILD_CMD"
    eval "$BUILD_CMD"
    log_info "Base model built and pretrained."
else
    log_info "Skipping build (--skip-build)"
fi

# ---------------------------------------------------------------------------
# STEP 3: Generate training data from all teacher models
# ---------------------------------------------------------------------------
if [[ "$SKIP_GENERATE" == "0" ]]; then
    log_step "Generating training data (teachers=$TEACHER_MODELS, samples=$NUM_SAMPLES)"

    GEN_CMD="python build.py --mode generate-data"
    GEN_CMD+=" --teacher-model $TEACHER_MODELS"
    GEN_CMD+=" --num-samples $NUM_SAMPLES"
    GEN_CMD+=" --conversations"
    GEN_CMD+=" --num-conversations $NUM_CONVERSATIONS"
    GEN_CMD+=" --data-dir $DATA_DIR"
    if [[ "$QUANTIZE" != "none" ]]; then
        GEN_CMD+=" --quantize $QUANTIZE"
    fi
    if [[ "$TEACHER_DEVICE" != "auto" ]]; then
        GEN_CMD+=" --teacher-device $TEACHER_DEVICE"
    fi
    if [[ "$TEACHER_DTYPE" != "auto" ]]; then
        GEN_CMD+=" --teacher-dtype $TEACHER_DTYPE"
    fi

    log_info "Running: $GEN_CMD"
    eval "$GEN_CMD"
    log_info "Training data generated."

    # Show data stats
    DATA_FILE="$DATA_DIR/distillation_data.jsonl"
    if [[ -f "$DATA_FILE" ]]; then
        LINES=$(wc -l < "$DATA_FILE" | tr -d ' ')
        SIZE=$(du -sh "$DATA_FILE" | cut -f1)
        log_info "Dataset: $LINES samples, $SIZE"
    fi
else
    log_info "Skipping data generation (--skip-generate)"
fi

# ---------------------------------------------------------------------------
# STEP 4: Distill from all teacher models into FramerAI
# ---------------------------------------------------------------------------
if [[ "$SKIP_DISTILL" == "0" ]]; then
    log_step "Distilling knowledge from teachers into FramerAI (steps=$DISTILL_STEPS)"

    DISTILL_CMD="python build.py --mode distill --size $MODEL_SIZE"
    DISTILL_CMD+=" --teacher-model $TEACHER_MODELS"
    DISTILL_CMD+=" --max-steps $DISTILL_STEPS"
    DISTILL_CMD+=" --batch-size $BATCH_SIZE"
    DISTILL_CMD+=" --lr $LEARNING_RATE"
    DISTILL_CMD+=" --output-dir $OUTPUT_DIR"
    DISTILL_CMD+=" --distill-data $DATA_DIR/distillation_data.jsonl"
    if [[ "$QUANTIZE" != "none" ]]; then
        DISTILL_CMD+=" --quantize $QUANTIZE"
    fi
    if [[ "$DEVICE" != "auto" ]]; then
        DISTILL_CMD+=" --device $DEVICE"
    fi
    if [[ "$TEACHER_DEVICE" != "auto" ]]; then
        DISTILL_CMD+=" --teacher-device $TEACHER_DEVICE"
    fi
    if [[ -n "$ROPE_SCALING" ]]; then
        DISTILL_CMD+=" --rope-scaling $ROPE_SCALING"
    fi
    if [[ -n "$TARGET_SEQ_LEN" ]]; then
        DISTILL_CMD+=" --target-seq-len $TARGET_SEQ_LEN"
    fi

    log_info "Running: $DISTILL_CMD"
    eval "$DISTILL_CMD"
    log_info "Distillation complete."
else
    log_info "Skipping distillation (--skip-distill)"
fi

# ---------------------------------------------------------------------------
# STEP 5: Export the final model
# ---------------------------------------------------------------------------
if [[ "$SKIP_EXPORT" == "0" ]]; then
    log_step "Exporting final model"

    EXPORT_CMD="python build.py --mode export --size $MODEL_SIZE"
    EXPORT_CMD+=" --output-dir $OUTPUT_DIR"

    log_info "Running: $EXPORT_CMD"
    eval "$EXPORT_CMD"
    log_info "Model exported to $OUTPUT_DIR/export/"

    # Show export info
    if [[ -f "$OUTPUT_DIR/export/model_info.json" ]]; then
        log_info "Model info:"
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
    log_info "Skipping export (--skip-export)"
fi

# ---------------------------------------------------------------------------
# STEP 6: Start backend + frontend
# ---------------------------------------------------------------------------
if [[ "$SKIP_SERVE" == "0" ]]; then
    log_step "Starting backend and frontend"

    # Install backend dependencies
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

    # Install frontend dependencies
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

    # Wait for background processes
    wait
else
    log_info "Skipping serve (--skip-serve)"

    echo ""
    echo -e "${GREEN}${BOLD}============================================================${NC}"
    echo -e "${GREEN}${BOLD}  FramerAI training complete!${NC}"
    echo -e "${GREEN}${BOLD}============================================================${NC}"
    echo ""
    echo -e "  Model:  ${CYAN}$OUTPUT_DIR/export/framerai_model.pt${NC}"
    echo ""
    echo "  To start serving:"
    echo "    cd backend && npm run dev &"
    echo "    cd website && npm run dev &"
    echo ""
fi
