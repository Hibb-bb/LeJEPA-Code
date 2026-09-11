# Light-curve experiments (LeJEPA cross-survey)

```
lightcurves/
├── lejepa-mup-ladder.py   main experiment: mu-P width ladder + LeJEPA / supervised pretraining
├── downstream.py          frozen-encoder eval (classification + period regression, transfer matrix)
├── jobs/                  SLURM scripts; submit from anywhere:  sbatch jobs/<name>.slurm
│   ├── cross-survey.slurm         1 GPU, array 0 = lejepa, 1 = supervised, then downstream
│   ├── cross-survey-4gpu.slurm    same on a full GH200 node (DDP)
│   ├── cross-survey-css.slurm     CSS-expanded pretraining (~69k stars)
│   ├── downstream.slurm           downstream eval of one checkpoint
│   └── knn.slurm                  k-NN inspection of one checkpoint
├── analysis/              post-hoc analysis of checkpoints (run from this folder)
│   ├── knn_inspect.py     nearest-neighbour tables + alignment / purity metrics
│   ├── umap_compare.py    LeJEPA vs supervised vs random-init UMAP
│   └── umap_css.py        UMAP on the CSS cross-match
├── data/
│   ├── build_starembed_exclusion.py   builds the StarEmbed val/test/anom exclusion list
│   └── starembed_exclude_gaia.txt     -> pass as --exclude-stars-file data/starembed_exclude_gaia.txt
├── notes/                 write-ups (embedding_diagnostics.md)
├── runs/                  gitignored. <wandb project>/<run id>/checkpoints/*.ckpt + downstream_*.npz/png
└── logs/                  gitignored. SLURM stdout/stderr
```

Run ids are `cs<slurm array job id>-<mode>`, so a run's checkpoint and its
downstream outputs live in `runs/Cross-Survey-LC/cs<job>-<mode>/checkpoints/`.
All scripts resolve paths relative to this folder (`RUNS_DIR` in the main
script), so the layout works regardless of the current working directory;
the SLURM jobs `--chdir` here.

## Hugging Face token

The datasets are private. No token is stored in the repo; scripts resolve it
as `--hf-token` flag > `HF_TOKEN` env var > the login cached by
`hf auth login` (`~/.cache/huggingface/token`). The cluster account already
has the cached login, so the SLURM jobs need no extra setup.
