#!/bin/bash
#SBATCH --job-name=probe-sweep
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --chdir=/projects/bfrf/hibb/LeJEPA-Code/benchmarks/lightcurves
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Hyper-parameter sweep of the class-weighted MLP classification probe on the
# saved per-survey embeddings of the 100-epoch encoders (no re-encoding):
# class-weighting x lr x batch = 27 configs per training survey, selected on that
# survey's val macro-F1, test macro-F1 reported on every survey. Then the
# report figures are regenerated from the selected probes' predictions.
# ~30 min on one GH200.
# Submit from anywhere:  sbatch jobs/probe-sweep.slurm [TAG]   (default 100ep)

set -euo pipefail
mkdir -p logs
source ../../.venv/bin/activate
export LOGURU_LEVEL=WARNING
R=runs/Pair-ZTF-ASASSN
TAG=${1:-100ep}
OUT=$R/probe_sweep_${TAG}_cw

python analysis/probe_sweep.py \
    "LeJEPA λ=0.02=$R/pair3163526-lejepa/checkpoints/downstream_embeddings.npz" \
    "Contrastive=$R/pair3163303-contrastive/checkpoints/downstream_embeddings.npz" \
    "LeJEPA λ=0.2=$R/pair3163588-lejepa/checkpoints/downstream_embeddings.npz" \
    "LeJEPA λ=2.0=$R/pair3163589-lejepa/checkpoints/downstream_embeddings.npz" \
    --out $OUT.json

python analysis/plot_sweep_selected.py $OUT.json \
    --handcrafted $R/handcrafted3163304/features_rf.json \
    --out $R/selected_f1_${TAG}_cw.png
