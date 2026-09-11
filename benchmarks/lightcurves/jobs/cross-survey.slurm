#!/bin/bash
#SBATCH --job-name=cross-survey
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80g
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Cross-Survey-LC: long pretraining + frozen-encoder downstream eval, both
# logged to the same wandb run.
#   array 0: lejepa (identity projector; 700 epochs — the 1500-epoch run
#            overshot: sep_acc bottomed at ~0.60 around epoch 711 and the
#            downstream metrics were worse than at 500)
#   array 1: supervised baseline (same encoder + linear CE head, no LeJEPA
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

MODES=(lejepa supervised)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
# Fixed run id -> predictable checkpoint dir + downstream logs into the
# same wandb run (WandbLogger picks the id up from the environment).
export WANDB_RUN_ID=cs${SLURM_ARRAY_JOB_ID}-${MODE}
echo "mode=$MODE wandb_run_id=$WANDB_RUN_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python pretrain.py \
    --dataset hibb/tess-ztf-atlas-asassn-isect \
    --exclude-instrument TESS --holdout-instrument ATLAS \
    --mode "$MODE" --local-mode instrument \
    --width 256 --depth 4 --projector identity \
    --lamb-ref 0.02 --base-lr 4e-4 \
    --norm band-global --p-period-shift 0.5 \
    --batch-size 128 --epochs 700 --no-probe \
    --num-workers 16 --precision 16-mixed \
    --wandb "$WANDB_PROJECT" \
    "$@"

CKPT=$(ls -t "runs/$WANDB_PROJECT/$WANDB_RUN_ID"/checkpoints/*.ckpt | head -1)
echo "downstream eval on $CKPT"
unset WANDB_RUN_ID  # downstream resumes the run explicitly via --wandb-id
python downstream.py \
    --ckpt "$CKPT" \
    --dataset hibb/tess-ztf-atlas-asassn-isect \
    --exclude-instrument TESS --holdout-instrument ATLAS \
    --width 256 --depth 4 --norm band-global \
    --n-eval-views 4 --batch-size 256 --num-workers 16 \
    --wandb "$WANDB_PROJECT" --wandb-id "cs${SLURM_ARRAY_JOB_ID}-${MODE}"
