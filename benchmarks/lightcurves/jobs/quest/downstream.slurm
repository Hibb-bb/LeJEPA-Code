#!/bin/bash
#SBATCH --job-name=lc-downstream
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --chdir=/projects/b1094/ywl7940/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Quest version of jobs/downstream.slurm: frozen-encoder eval (classification +
# log-period regression probes, cross-survey transfer matrix on the test split).
# All run-specific flags (--ckpt, --holdout-instrument, wave flags, --wandb-id)
# come from the caller; jobs/quest/pair-ztf-asassn.slurm submits this itself.
# Submit:  sbatch jobs/quest/downstream.slurm --ckpt <ckpt> --holdout-instrument ATLAS --exclude-instrument TESS [flags]

set -euo pipefail
mkdir -p logs
source jobs/quest/env.sh

python downstream.py \
    --dataset pc/ZTF-ATLAS-ASASSN-isect \
    --width 360 --depth 6 \
    --n-eval-views 4 --batch-size 256 --num-workers 12 \
    "$@"
