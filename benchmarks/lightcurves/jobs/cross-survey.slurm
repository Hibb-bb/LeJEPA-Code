#!/bin/bash
#SBATCH --job-name=cross-survey
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80g
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Cross-Survey-LC: pretraining (this job, 1 GPU, 100 epochs ~1.7 h at
# width 360 / depth 6) which then submits jobs/downstream.slurm for the
# frozen-encoder eval; both log to the same wandb run.
#   lejepa: identity projector; 100 epochs (budget for the ablation sweep;
#            the old 256x4 runs used 700)
#   supervised: baseline (same encoder + linear CE head, no LeJEPA
#            pretraining, same budget)
# band-global norm keeps the star's variability amplitude in the inputs;
# p-period-shift adds same-phase different-cycle windows (|k| <= 2).
# After training, downstream.py fits classification + period-regression
# probes on the run's checkpoint and logs the transfer tables and the
# pred-vs-true period scatter into the same run.
# Submit from anywhere:  sbatch jobs/cross-survey.slurm [extra flags]

set -euo pipefail
mkdir -p logs
source ../../.venv/bin/activate
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export LOGURU_LEVEL=WARNING
WANDB_PROJECT=Cross-Survey-LC
WAVE_FLAGS=${WAVE_FLAGS:-}

# Train on two surveys, evaluate on the third (leave-one-survey-out over
# ZTF / ATLAS / ASASSN; TESS is excluded). Default: LeJEPA only, array
# 0/1/2 = hold out ATLAS / ASASSN / ZTF. To add the supervised baseline:
#   MODES="lejepa supervised" sbatch --array=0-5 jobs/cross-survey.slurm
# (array id = n_modes * holdout + mode).
# Width 360 x 6 heads (head_dim 60) x depth 6 (RoMAE-base was OOM at batch 128).
# Wavelength-encoding ablation: WAVE_FLAGS="--wave-pos pct" sbatch jobs/cross-survey.slurm
# (same flags go to pretrain.py AND downstream.py; see pretrain.py --help).
read -ra MODES <<< "${MODES:-lejepa}"
HOLDOUTS=(ATLAS ASASSN ZTF)
NM=${#MODES[@]}
MODE=${MODES[$((SLURM_ARRAY_TASK_ID % NM))]}
HOLDOUT=${HOLDOUTS[$((SLURM_ARRAY_TASK_ID / NM))]}
# Fixed run id -> predictable checkpoint dir + downstream logs into the
# same wandb run (WandbLogger picks the id up from the environment).
export WANDB_RUN_ID=cs${SLURM_ARRAY_JOB_ID}-${MODE}-ho${HOLDOUT}
echo "mode=$MODE holdout=$HOLDOUT wandb_run_id=$WANDB_RUN_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python pretrain.py \
    --dataset pc/ZTF-ATLAS-ASASSN-isect \
    --exclude-instrument TESS --holdout-instrument "$HOLDOUT" \
    --mode "$MODE" --local-mode instrument \
    --width 360 --depth 6 --projector identity \
    $WAVE_FLAGS \
    --lamb-ref 0.02 --base-lr 4e-4 \
    --norm band-global --p-period-shift 0.5 \
    --batch-size 128 --epochs 100 --no-probe \
    --num-workers 16 --precision 16-mixed \
    --wandb "$WANDB_PROJECT" \
    "$@"

CKPT=$(ls -t "runs/$WANDB_PROJECT/$WANDB_RUN_ID"/checkpoints/*.ckpt | head -1)
echo "pretraining done: $CKPT"
# Downstream runs as its own short job (queues faster than one long job).
sbatch --job-name="lc-downstream-${WANDB_RUN_ID}" jobs/downstream.slurm \
    --ckpt "$CKPT" \
    --exclude-instrument TESS --holdout-instrument "$HOLDOUT" \
    --norm band-global $WAVE_FLAGS \
    --wandb "$WANDB_PROJECT" --wandb-id "$WANDB_RUN_ID"
