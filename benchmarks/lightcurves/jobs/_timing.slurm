#!/bin/bash
#SBATCH --job-name=lc-timing
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80g
#SBATCH --time=00:40:00
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
BS=${BS:-64}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ../../.venv/bin/activate
export HF_DATASETS_DISABLE_PROGRESS_BARS=1 LOGURU_LEVEL=WARNING
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python pretrain.py \
    --dataset pc/ZTF-ATLAS-ASASSN-isect \
    --exclude-instrument TESS --holdout-instrument ATLAS \
    --mode lejepa --local-mode instrument \
    --width 360 --depth 6 --projector identity \
    --lamb-ref 0.02 --base-lr 4e-4 \
    --norm band-global --p-period-shift 0.5 \
    --batch-size ${BS} --epochs 4 --no-probe --no-wandb \
    --num-workers 16 --precision 16-mixed --wave-pos pct-nd
nvidia-smi --query-gpu=memory.used --format=csv,noheader
