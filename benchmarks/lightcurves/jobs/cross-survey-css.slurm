#!/bin/bash
#SBATCH --job-name=cross-survey-css
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=440g
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# CSS-expanded pretraining (4x GH200 DDP, same scaling recipe as
# cross-survey-4gpu.slurm):
#   data     = isect + hibb/CSSxPC merged by gaia id (~82k stars) MINUS the
#              StarEmbed val/test/anom crossmatch (starembed_exclude_gaia.txt,
#              13,363 stars) -> ~69k records; CSS-only stars train as
#              single-instrument records (--min-train-instruments 1) whose
#              positives are different-epoch windows of the same survey.
#   pairing  = multi-instrument stars oversampled 3x so cross-survey pairs
#              stay ~1/3 of the epoch despite the single-survey majority.
#   epochs   = 200 (~5.6x more stars/epoch than the 12k runs).
#   holdout  = ATLAS (unchanged); TESS excluded.
# Downstream gets the same data flags so its record list + split match.
# Submit from anywhere:  sbatch jobs/cross-survey-css.slurm [extra flags]

set -euo pipefail
mkdir -p logs
source ../../.venv/bin/activate
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export LOGURU_LEVEL=WARNING
WANDB_PROJECT=Cross-Survey-LC

MODES=(lejepa supervised)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
export WANDB_RUN_ID=cs${SLURM_ARRAY_JOB_ID}-${MODE}
echo "mode=$MODE wandb_run_id=$WANDB_RUN_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python pretrain.py \
    --dataset hibb/tess-ztf-atlas-asassn-isect+css \
    --exclude-instrument TESS --holdout-instrument ATLAS \
    --min-train-instruments 1 \
    --exclude-stars-file data/starembed_exclude_gaia.txt \
    --oversample-paired 3 \
    --mode "$MODE" --local-mode instrument \
    --width 256 --depth 4 --projector identity \
    --lamb-ref 0.02 --base-lr 8e-4 \
    --norm band-global --p-period-shift 0.5 \
    --batch-size 128 --epochs 200 --no-probe \
    --devices 4 --num-workers 32 --precision 16-mixed \
    --wandb "$WANDB_PROJECT" \
    "$@"

CKPT=$(ls -t "runs/$WANDB_PROJECT/$WANDB_RUN_ID"/checkpoints/*.ckpt | head -1)
echo "downstream eval on $CKPT"
unset WANDB_RUN_ID
python downstream.py \
    --ckpt "$CKPT" \
    --dataset hibb/tess-ztf-atlas-asassn-isect+css \
    --exclude-instrument TESS --holdout-instrument ATLAS \
    --min-train-instruments 1 \
    --exclude-stars-file data/starembed_exclude_gaia.txt \
    --width 256 --depth 4 --norm band-global \
    --n-eval-views 4 --batch-size 256 --num-workers 16 \
    --wandb "$WANDB_PROJECT" --wandb-id "cs${SLURM_ARRAY_JOB_ID}-${MODE}"
