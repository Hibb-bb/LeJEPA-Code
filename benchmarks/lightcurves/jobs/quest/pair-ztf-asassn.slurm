#!/bin/bash
#SBATCH --job-name=pair-ztf-asassn
#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=sxm   # 80 GB A100 SXM or H100; the 40 GB PCIe A100s OOM (contrastive)
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --chdir=/projects/b1094/ywl7940/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Quest version of jobs/pair-ztf-asassn.slurm (same science config):
# ZTF + ASAS-SN training pair, ATLAS held out, TESS and i_ZTF excluded.
#   array 0  lejepa       lamb_ref 0.02, MLP projector, MLP predictor on ASAS-SN
#   array 1  contrastive  multi-positive NT-Xent, temperature 0.2
# Each task submits jobs/quest/downstream.slurm on its final checkpoint.
#
# Checkpoints: runs/<project>/<run id>/checkpoints/last.ckpt every epoch plus a
# kept epoch=<n-1>-*.ckpt every --ckpt-every (100) epochs. Runs always start
# with --resume auto, so a job that hits the 48 h wall time is continued by
# resubmitting with the same RUN_TAG (the original array job id):
#   RUN_TAG=<first job id> sbatch --array=<task> jobs/quest/pair-ztf-asassn.slurm --epochs 400
# Keep --epochs (and all other flags) identical when resuming.
#
# Submit from anywhere:
#   sbatch jobs/quest/pair-ztf-asassn.slurm --epochs 400
#   sbatch --gres=gpu:h100:1 jobs/quest/pair-ztf-asassn.slurm --epochs 400   # pin the GPU type

set -euo pipefail
mkdir -p logs
source jobs/quest/env.sh
WANDB_PROJECT=${WANDB_PROJECT:-Pair-ZTF-ASASSN}
WAVE_FLAGS=${WAVE_FLAGS:-}
HOLDOUT=ATLAS

MODES=(lejepa contrastive)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
export WANDB_RUN_ID=pair${RUN_TAG:-$SLURM_ARRAY_JOB_ID}-${MODE}
echo "mode=$MODE holdout=$HOLDOUT wandb_run_id=$WANDB_RUN_ID"

COMMON=(
    --dataset pc/ZTF-ATLAS-ASASSN-isect
    --exclude-instrument TESS --exclude-band i_ZTF --holdout-instrument "$HOLDOUT"
    --local-mode instrument
    --width 360 --depth 6 --projector mlp --proj-dim 128
    --base-lr 4e-4
    --norm band-global --p-period-shift 0.5
    --batch-size 128 --epochs 100 --no-probe
    --num-workers 12 --precision 16-mixed
    --wandb "$WANDB_PROJECT"
    --resume auto
)

if [[ "$MODE" == "lejepa" ]]; then
    python pretrain.py "${COMMON[@]}" $WAVE_FLAGS \
        --mode lejepa --lamb-ref 0.02 \
        --predictor-instrument ASASSN --predictor-hidden 1024 \
        "$@"
else
    python pretrain_contrastive.py "${COMMON[@]}" $WAVE_FLAGS \
        --temperature 0.2 \
        "$@"
fi

CKPT=$(ls -t "runs/$WANDB_PROJECT/$WANDB_RUN_ID"/checkpoints/*.ckpt | head -1)
echo "pretraining done: $CKPT"
sbatch --job-name="lc-downstream-${WANDB_RUN_ID}" jobs/quest/downstream.slurm \
    --ckpt "$CKPT" \
    --exclude-instrument TESS --exclude-band i_ZTF --holdout-instrument "$HOLDOUT" \
    --projector mlp --probe mlp \
    --norm band-global $WAVE_FLAGS \
    --wandb "$WANDB_PROJECT" --wandb-id "$WANDB_RUN_ID"
