#!/bin/bash
#SBATCH --job-name=handcrafted
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200g
#SBATCH --time=06:00:00
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Handcrafted-feature baseline (StarEmbed FATS + light_curve feature set)
# for the ZTF + ASAS-SN / ATLAS-holdout protocol: extracts per-instrument
# features on the same capped eval windows downstream.py embeds, then runs
# the identical MLP-probe downstream eval (16-class classification +
# log-period regression, cross-survey transfer matrix on the test split).
# FATS is ~4 s per band, so the extraction is the long part (~130k band
# light curves over 63 workers).
# Submit from anywhere:  sbatch jobs/handcrafted.slurm [extra downstream flags]

set -euo pipefail
mkdir -p logs
source ../../.venv/bin/activate
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export LOGURU_LEVEL=WARNING
export OMP_NUM_THREADS=1
WANDB_PROJECT=${WANDB_PROJECT:-Pair-ZTF-ASASSN}
HOLDOUT=ATLAS
OUT=runs/$WANDB_PROJECT/handcrafted${SLURM_JOB_ID}/features

DATA_FLAGS=(
    --dataset pc/ZTF-ATLAS-ASASSN-isect
    --exclude-instrument TESS --exclude-band i_ZTF --holdout-instrument "$HOLDOUT"
)

python handcrafted_features.py "${DATA_FLAGS[@]}" \
    --n-eval-views 1 --num-workers $((SLURM_CPUS_PER_TASK - 1)) \
    --out "$OUT"

python downstream.py --features "$OUT.npz" "${DATA_FLAGS[@]}" \
    --probe mlp --norm band-global \
    --wandb "$WANDB_PROJECT" \
    "$@"
