#!/bin/bash
#SBATCH --job-name=lc-timing
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=sxm   # 80 GB A100 SXM or H100; the 40 GB PCIe A100s OOM (contrastive)
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/b1094/ywl7940/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Seconds per epoch of the jobs/quest/pair-ztf-asassn.slurm LeJEPA config on a
# gengpu GPU, to size --time for the 400-epoch run (400 x epoch time + ~10%).
# No wandb, no checkpoints kept beyond runs/lightning_logs.
# Submit:  sbatch jobs/quest/timing.slurm            (any gengpu GPU)
#          sbatch --gres=gpu:a100:1 jobs/quest/timing.slurm

set -euo pipefail
mkdir -p logs
source jobs/quest/env.sh

EPOCHS=${EPOCHS:-3}
start=$(date +%s)
python pretrain.py \
    --dataset pc/ZTF-ATLAS-ASASSN-isect \
    --exclude-instrument TESS --exclude-band i_ZTF --holdout-instrument ATLAS \
    --local-mode instrument \
    --width 360 --depth 6 --projector mlp --proj-dim 128 \
    --base-lr 4e-4 \
    --norm band-global --p-period-shift 0.5 \
    --batch-size 128 --epochs "$EPOCHS" --no-probe --no-wandb \
    --num-workers 12 --precision 16-mixed \
    --mode lejepa --lamb-ref 0.02 \
    --predictor-instrument ASASSN --predictor-hidden 1024 \
    --ckpt-every 0
end=$(date +%s)
echo "wall time for $EPOCHS epochs incl. data loading: $((end - start)) s"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
