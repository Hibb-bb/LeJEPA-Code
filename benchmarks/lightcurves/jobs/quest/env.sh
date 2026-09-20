# Shared setup for the Quest (Northwestern) jobs; sourced by jobs/quest/*.slurm.
# The Delta scripts in jobs/ are unchanged.
source ../../.venv/bin/activate
# Local PC_matches DatasetDicts (pc/<name> ids in pretrain.py / downstream.py).
export PC_MATCHES_ROOT=${PC_MATCHES_ROOT:-/projects/b1094/rehemtulla/SkAI/skai_universal_forecaster/data/PC_matches}
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export LOGURU_LEVEL=WARNING
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
