#!/bin/bash
#SBATCH --job-name=pair-ztf-asassn
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80g
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# ZTF + ASAS-SN training pair, ATLAS held out; TESS excluded. Two
# pretraining runs on the PC_matches unified split, each followed by the
# frozen-encoder downstream eval (MLP probes for 16-class classification and
# log-period regression, full cross-survey transfer matrix on the test
# split of ZTF / ASASSN / ATLAS separately):
#   array 0  lejepa       MLP projector (losses in projection space, probes
#                         on the encoder), MLP predictor on the ASAS-SN views
#   array 1  contrastive  multi-positive NT-Xent (globals + locals), MLP
#                         projector, no predictor
# The handcrafted-feature baseline is jobs/handcrafted.slurm (same protocol).
# Submit from anywhere:  sbatch jobs/pair-ztf-asassn.slurm [extra flags]

set -euo pipefail
mkdir -p logs
source ../../.venv/bin/activate
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export LOGURU_LEVEL=WARNING
WANDB_PROJECT=${WANDB_PROJECT:-Pair-ZTF-ASASSN}
WAVE_FLAGS=${WAVE_FLAGS:-}
HOLDOUT=ATLAS

MODES=(lejepa contrastive)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
export WANDB_RUN_ID=pair${SLURM_ARRAY_JOB_ID}-${MODE}
echo "mode=$MODE holdout=$HOLDOUT wandb_run_id=$WANDB_RUN_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

COMMON=(
    --dataset pc/ZTF-ATLAS-ASASSN-isect
    --exclude-instrument TESS --exclude-band i_ZTF --holdout-instrument "$HOLDOUT"
    --local-mode instrument
    --width 360 --depth 6 --projector mlp --proj-dim 128
    --base-lr 4e-4
    --norm band-global --p-period-shift 0.5
    --batch-size 128 --epochs 100 --no-probe
    --num-workers 16 --precision 16-mixed
    --wandb "$WANDB_PROJECT"
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
sbatch --job-name="lc-downstream-${WANDB_RUN_ID}" jobs/downstream.slurm \
    --ckpt "$CKPT" \
    --exclude-instrument TESS --exclude-band i_ZTF --holdout-instrument "$HOLDOUT" \
    --projector mlp --probe mlp \
    --norm band-global $WAVE_FLAGS \
    --wandb "$WANDB_PROJECT" --wandb-id "$WANDB_RUN_ID"
