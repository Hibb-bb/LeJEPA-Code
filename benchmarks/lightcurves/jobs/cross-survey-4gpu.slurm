#!/bin/bash
#SBATCH --job-name=cross-survey-4gpu
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=440g
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# 4-GPU DDP version of cross-survey.slurm (full node: 4x GH200).
#   array = 3 holdout surveys x (lejepa, supervised); see below.
# Same science config as the single-GPU run except for the DDP scaling:
#   - --batch-size 128 stays PER RANK, so SIGReg still sees batches of 128
#     per rank (no cross-rank gather in the loss) and lambda_ref 0.02
#     transfers unchanged via the master rule. Global optimizer batch = 512.
#   - base-lr is sqrt-scaled with the 4x global batch: 4e-4 -> 8e-4.
#   - InstrumentEmbeddingEval all-gathers the val shards, so inst/* metrics
#     and the UMAP are computed on the full val set as in single-GPU runs.
# 32 workers/rank + prefetch/pin_memory to keep GPU utilization up.
# Submit from anywhere:  sbatch jobs/cross-survey-4gpu.slurm [extra flags]

set -euo pipefail
mkdir -p logs
source ../../.venv/bin/activate
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export LOGURU_LEVEL=WARNING
WANDB_PROJECT=Cross-Survey-LC
WAVE_FLAGS=${WAVE_FLAGS:-}

# Train on two surveys, evaluate on the third (leave-one-survey-out over
# ZTF / ATLAS / ASASSN; TESS is excluded). array id = 2*holdout + mode:
#   0/1 = hold out ATLAS  (lejepa / supervised)
#   2/3 = hold out ASASSN
#   4/5 = hold out ZTF
# Width 360 x 6 heads (head_dim 60) x depth 6 (RoMAE-base was OOM at batch 128). Star split: data/splits/<dataset>_seed0.json.
# Wavelength-encoding ablation: export WAVE_FLAGS="--wave-pos pct --wave-ctx none"
# (same flags go to pretrain.py AND downstream.py; see pretrain.py --help).
MODES=(lejepa supervised)
HOLDOUTS=(ATLAS ASASSN ZTF)
MODE=${MODES[$((SLURM_ARRAY_TASK_ID % 2))]}
HOLDOUT=${HOLDOUTS[$((SLURM_ARRAY_TASK_ID / 2))]}
# Fixed run id -> predictable checkpoint dir + downstream logs into the
# same wandb run (WandbLogger picks the id up from the environment).
export WANDB_RUN_ID=cs${SLURM_ARRAY_JOB_ID}-${MODE}-ho${HOLDOUT}
echo "mode=$MODE holdout=$HOLDOUT wandb_run_id=$WANDB_RUN_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# srun launches one Lightning rank per GPU (SLURM env is auto-detected).
srun python pretrain.py \
    --dataset pc/ZTF-ATLAS-ASASSN-isect \
    --exclude-instrument TESS --holdout-instrument "$HOLDOUT" \
    --mode "$MODE" --local-mode instrument \
    --width 360 --depth 6 --projector identity \
    $WAVE_FLAGS \
    --lamb-ref 0.02 --base-lr 8e-4 \
    --norm band-global --p-period-shift 0.5 \
    --batch-size 128 --epochs 300 --no-probe \
    --devices 4 --num-workers 32 --precision 16-mixed \
    --wandb "$WANDB_PROJECT" \
    "$@"

CKPT=$(ls -t "runs/$WANDB_PROJECT/$WANDB_RUN_ID"/checkpoints/*.ckpt | head -1)
echo "downstream eval on $CKPT"
unset WANDB_RUN_ID  # downstream resumes the run explicitly via --wandb-id
python downstream.py \
    --ckpt "$CKPT" \
    --dataset pc/ZTF-ATLAS-ASASSN-isect \
    --exclude-instrument TESS --holdout-instrument "$HOLDOUT" \
    --width 360 --depth 6 --norm band-global \
    $WAVE_FLAGS \
    --n-eval-views 4 --batch-size 256 --num-workers 16 \
    --wandb "$WANDB_PROJECT" --wandb-id "cs${SLURM_ARRAY_JOB_ID}-${MODE}-ho${HOLDOUT}"
