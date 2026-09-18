#!/bin/bash
#SBATCH --job-name=predictor-eval
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# LeJEPA lambda=0.02 (100 ep): represent ASAS-SN by the *predictor output*
# (predictor applied per eval view to the encoder output, then averaged)
# instead of the raw encoder output; ZTF / ATLAS keep the encoder output.
#   1. downstream.py re-embeds with --predictor-instrument ASASSN and runs the
#      default (unweighted) MLP probes: classification + period regression,
#      full 3x3 transfer matrix -> checkpoints/downstream_pred.*
#   2. probe_sweep.py sweeps the classification probe on both embedding sets
#      (encoder-only and predictor) with val-F1 selection, for a like-for-like
#      comparison -> probe_sweep_100ep_pred.json + bar plot.
# Submit from anywhere:  sbatch jobs/predictor-eval.slurm

set -euo pipefail
mkdir -p logs
source ../../.venv/bin/activate
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export LOGURU_LEVEL=WARNING
R=runs/Pair-ZTF-ASASSN
RUN=$R/pair3163526-lejepa/checkpoints
CKPT=$(ls -t $RUN/*.ckpt | head -1)

python downstream.py --ckpt "$CKPT" \
    --dataset pc/ZTF-ATLAS-ASASSN-isect \
    --exclude-instrument TESS --exclude-band i_ZTF --holdout-instrument ATLAS \
    --width 360 --depth 6 --projector mlp \
    --predictor-instrument ASASSN --predictor-hidden 1024 \
    --probe mlp --class-weight none --norm band-global \
    --n-eval-views 4 --batch-size 256 --num-workers 16 \
    --out $RUN/downstream_pred

python analysis/probe_sweep.py \
    "LeJEPA λ=0.02=$RUN/downstream_embeddings.npz" \
    "LeJEPA λ=0.02 + ASAS-SN predictor=$RUN/downstream_pred_embeddings.npz" \
    --out $R/probe_sweep_100ep_pred.json

python analysis/plot_sweep_selected.py $R/probe_sweep_100ep_pred.json \
    --handcrafted $R/handcrafted3163304/features_rf.json \
    --out $R/selected_f1_100ep_pred.png
