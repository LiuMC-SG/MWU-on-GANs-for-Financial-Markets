#!/bin/bash
#SBATCH --job-name=gan_master_workflow
#SBATCH --output=logs/master_%j.out
#SBATCH --error=logs/master_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

###############################################################################
# Master Workflow Script for GAN-based Anomaly Detection
# 
# This script orchestrates the complete workflow:
# 1. Model hyperparameter tuning (Optuna)
# 2. Trading signal hyperparameter tuning
# 3. Final backtesting with optimized parameters
#
# Usage:
#   sbatch slurm/master_workflow.sh <model_type> <feature_type> <ticker>
#
# Arguments:
#   model_type: sequential or parallel
#   feature_type: price or ohlcav
#   ticker: Stock ticker symbol (e.g., AAPL)
###############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Parse arguments
MODEL_TYPE=${1:-paper_gan_price}

# Configuration
PROJECT_ROOT="${HOME}/FYP"
MODEL_DIR="${MODEL_TYPE}"
DATA_DIR="${PROJECT_ROOT}/data"
OUTPUT_BASE="${PROJECT_ROOT}/results/${MODEL_DIR}"
LOGS_DIR="${PROJECT_ROOT}/logs/${MODEL_DIR}"

TICKER="SP500"

# Create directories
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${LOGS_DIR}"

echo "========================================"
echo "GAN Anomaly Detection Workflow"
echo "========================================"
echo "Project Root: ${PROJECT_ROOT}"
echo "Model Type:   ${MODEL_TYPE}"
echo "Model Dir:    ${PROJECT_ROOT}/${MODEL_DIR}"
echo "Output Dir:   ${OUTPUT_BASE}"
echo "Logs Dir:     ${LOGS_DIR}"
echo "========================================"

# Load conda environment
VENV="${PROJECT_ROOT}/venv"
source "${VENV}/bin/activate"

# Ensure we are in the project root so that `-m` imports use FYP as root
cd "${PROJECT_ROOT}"

# Temporarily extend PYTHONPATH so that imports like `from common import data_utils`
# and `from paper_gan_price import ...` work correctly.
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Using PYTHONPATH=${PYTHONPATH}"
echo "Current directory: $(pwd)"

###############################################################################
# STAGE 1: Model Hyperparameter Optimization (Optuna)
###############################################################################
echo ""
echo "========== STAGE 1: Model Hyperparameter Tuning =========="
echo "Starting at: $(date)"

STAGE1_OUTPUT="${OUTPUT_BASE}/stage1_model_optimization"
mkdir -p "${STAGE1_OUTPUT}"

python -m "${MODEL_DIR}.gan_train_optuna" optimize \
    --ticker "${TICKER}" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${STAGE1_OUTPUT}" \
    --n-trials 10 \
    --timeout 43200 \
    --train-best \
    2>&1 | tee "${LOGS_DIR}/stage1_optuna.log"

# Check if stage 1 completed successfully
if [ ! -f "${STAGE1_OUTPUT}/${TICKER}/best_model/best_params.json" ]; then
    echo "Error: Stage 1 did not produce best_params.json"
    exit 1
fi

echo "Stage 1 completed at: $(date)"
echo "Model saved to: ${STAGE1_OUTPUT}/${TICKER}/best_model"

###############################################################################
# STAGE 2: Trading Signal Hyperparameter Optimization
###############################################################################
echo ""
echo "========== STAGE 2: Trading Signal Hyperparameter Tuning =========="
echo "Starting at: $(date)"

STAGE2_OUTPUT="${OUTPUT_BASE}/stage2_signal_optimization"
mkdir -p "${STAGE2_OUTPUT}"

INPUT_FILE="${DATA_DIR}/${TICKER}_backtest.csv"

python -m "anomaly_backtest_vectorbt" \
    --input "${INPUT_FILE}" \
    --output "${STAGE2_OUTPUT}" \
    --model-path "${STAGE1_OUTPUT}/${TICKER}/best_model" \
    --model-type "${MODEL_TYPE}" \
    --optimize \
    --n-trials 10 \
    --short-ma-min 2 \
    --short-ma-max 60 \
    --long-ma-min 20 \
    --long-ma-max 300 \
    --start-date "2000-01-01" \
    --end-date "2003-12-31" \
    2>&1 | tee "${LOGS_DIR}/stage2_signal_optuna.log"

# Check if stage 2 completed successfully
if [ ! -f "${STAGE2_OUTPUT}/optimization_results.json" ]; then
    echo "Error: Stage 2 did not produce optimization_results.json"
    exit 1
fi

echo "Stage 2 completed at: $(date)"

# Extract best parameters
BEST_SHORT_MA=$(python -c "import json; print(json.load(open('${STAGE2_OUTPUT}/optimization_results.json'))['best_params']['short_ma_period'])")
BEST_LONG_MA=$(python -c "import json; print(json.load(open('${STAGE2_OUTPUT}/optimization_results.json'))['best_params']['long_ma_period'])")

echo "Best short MA: ${BEST_SHORT_MA}"
echo "Best long MA: ${BEST_LONG_MA}"

###############################################################################
# STAGE 3: Final Backtesting with Optimized Parameters
###############################################################################
echo ""
echo "========== STAGE 3: Final Backtesting =========="
echo "Starting at: $(date)"

STAGE3_OUTPUT="${OUTPUT_BASE}/stage3_final_backtest"
mkdir -p "${STAGE3_OUTPUT}"

python -m "anomaly_backtest_vectorbt" \
    --input "${INPUT_FILE}" \
    --output "${STAGE3_OUTPUT}" \
    --model-path "${STAGE1_OUTPUT}/best_model" \
    --short-ma "${BEST_SHORT_MA}" \
    --long-ma "${BEST_LONG_MA}" \
    --start-date "2018-01-01" \
    --end-date "2023-12-31" \
    2>&1 | tee "${LOGS_DIR}/stage3_final_backtest.log"

echo "Stage 3 completed at: $(date)"

###############################################################################
# Generate Summary Report
###############################################################################
echo ""
echo "========== Generating Summary Report =========="

SUMMARY_FILE="${OUTPUT_BASE}/workflow_summary.txt"

cat > "${SUMMARY_FILE}" << EOF
======================================
GAN Anomaly Detection Workflow Summary
======================================
Ticker: ${TICKER}
Model Type: ${MODEL_TYPE}
Completed: $(date)

Stage 1: Model Hyperparameter Optimization
- Output: ${STAGE1_OUTPUT}
- Model: ${STAGE1_OUTPUT}/best_model

Stage 2: Trading Signal Optimization
- Output: ${STAGE2_OUTPUT}
- Best Short MA: ${BEST_SHORT_MA}
- Best Long MA: ${BEST_LONG_MA}

Stage 3: Final Backtesting
- Output: ${STAGE3_OUTPUT}
- Test Period: 2018-01-01 to 2023-12-31

Results Files:
- Model parameters: ${STAGE1_OUTPUT}/best_model/best_params.json
- Signal optimization: ${STAGE2_OUTPUT}/optimization_results.json
- Final backtest: ${STAGE3_OUTPUT}/backtest_results.json
- Performance plots: ${STAGE3_OUTPUT}/backtest_plots.png

======================================
EOF

cat "${SUMMARY_FILE}"

echo ""
echo "========================================"
echo "Workflow completed successfully!"
echo "Summary saved to: ${SUMMARY_FILE}"
echo "========================================"