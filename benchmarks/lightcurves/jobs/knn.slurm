#!/bin/bash
#SBATCH --job-name=knn-inspect
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Joint-embedding-space k-NN inspection of a pretrained checkpoint: embed 4
# segments per star per instrument, retrieve 5 nearest neighbours in the
# pooled ZTF+ASASSN+ATLAS space (same star+instrument excluded), report
# per-query neighbour tables + alignment/purity/period metrics.
# Submit:  sbatch jobs/knn.slurm [extra flags, e.g. --ckpt ...]

set -euo pipefail
mkdir -p logs
source ../../.venv/bin/activate
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export LOGURU_LEVEL=WARNING

python analysis/knn_inspect.py \
    --ckpt runs/Cross-Survey-LC/cs3079146-lejepa/checkpoints/epoch=699-step=59500.ckpt \
    --k 5 --views-per-star 4 --n-report 4 \
    --batch-size 256 --num-workers 16 \
    --wandb Cross-Survey-LC --wandb-id cs3079146-lejepa \
    "$@"
