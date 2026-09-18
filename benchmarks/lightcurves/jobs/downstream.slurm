#!/bin/bash
#SBATCH --job-name=lc-downstream
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Frozen-encoder downstream eval of a pretrained checkpoint: variable-star
# classification + log-period regression, full cross-survey transfer matrix
# (probe fit on one instrument's train-split embeddings, tested on every
# instrument's val and test splits incl. the holdout survey). Eval views use the same
# 500-1500 day capped windows as pretraining.
# Submit:  sbatch jobs/downstream.slurm --ckpt <ckpt> --holdout-instrument ATLAS --exclude-instrument TESS [flags]

set -euo pipefail
mkdir -p logs
source ../../.venv/bin/activate
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export LOGURU_LEVEL=WARNING

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# All run-specific flags (--ckpt, --holdout-instrument, wave flags, --wandb-id)
# come from the caller; cross-survey.slurm submits this automatically.
python downstream.py \
    --dataset pc/ZTF-ATLAS-ASASSN-isect \
    --width 360 --depth 6 \
    --n-eval-views 4 --batch-size 256 --num-workers 16 \
    "$@"
