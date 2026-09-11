#!/bin/bash
#SBATCH --job-name=lc-downstream
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Frozen-encoder downstream eval of a pretrained checkpoint: variable-star
# classification + log-period regression, full cross-survey transfer matrix
# (probe fit on one instrument's train-split embeddings, tested on every
# instrument's val split incl. the ATLAS holdout). Eval views use the same
# 500-1500 day capped windows as pretraining.
# Submit:  sbatch jobs/downstream.slurm [--ckpt ... extra flags]

set -euo pipefail
mkdir -p logs
source ../../.venv/bin/activate
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export LOGURU_LEVEL=WARNING

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python downstream.py \
    --ckpt runs/Cross-Survey-LC/cs3079146-lejepa/checkpoints/epoch=699-step=59500.ckpt \
    --dataset hibb/tess-ztf-atlas-asassn-isect \
    --exclude-instrument TESS --holdout-instrument ATLAS \
    --width 256 --depth 4 \
    --n-eval-views 4 --batch-size 256 --num-workers 16 \
    "$@"
