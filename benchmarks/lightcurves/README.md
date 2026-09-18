# Light-curve experiments (LeJEPA cross-survey)

```
lightcurves/
├── pretrain.py            main experiment: mu-P width ladder + LeJEPA / supervised pretraining
├── pretrain_contrastive.py  same pipeline with the multi-positive NT-Xent (nt_xent_multi) objective
├── downstream.py          frozen-encoder eval (MLP probes: 16-class classification + period regression, transfer matrix)
├── handcrafted_features.py  StarEmbed FATS + light_curve feature baseline on the same eval windows (-> downstream.py --features)
├── jobs/                  SLURM scripts; submit from anywhere:  sbatch jobs/<name>.slurm
│   ├── pair-ztf-asassn.slurm      ZTF+ASASSN pair, ATLAS holdout: array 0 = lejepa (+ASAS-SN predictor), 1 = contrastive, then downstream
│   ├── handcrafted.slurm          handcrafted-feature baseline under the same protocol
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
│   ├── starembed_exclude_gaia.txt     -> pass as --exclude-stars-file data/starembed_exclude_gaia.txt
│   ├── make_splits.py                 (hibb/ HF datasets only) star-level split by gaia id, stratified by class
│   ├── splits/<dataset>_seed0.json    -> picked up automatically for hibb/ datasets (--split-file)
│   └── filter_percentiles.json        10..90% transmission wavelengths per filter (SVO), loaded into BAND_PERCENTILES
├── notes/                 write-ups (embedding_diagnostics.md)
├── runs/                  gitignored. <wandb project>/<run id>/checkpoints/*.ckpt + downstream_*.npz/png
└── logs/                  gitignored. SLURM stdout/stderr
```

Run ids are `cs<slurm array job id>-<mode>`, so a run's checkpoint and its
downstream outputs live in `runs/Cross-Survey-LC/cs<job>-<mode>/checkpoints/`.
All scripts resolve paths relative to this folder (`RUNS_DIR` in the main
script), so the layout works regardless of the current working directory;
the SLURM jobs `--chdir` here.

## Data

The default `--dataset pc/ZTF-ATLAS-ASASSN-isect` is the local PC_matches
`DatasetDict` at `$PC_MATCHES_ROOT` (default `/projects/bfrf/data/PC_matches`,
override with `--data-root`). It ships a unified, gaia-id-disjoint
train/validation/test split (the `split` column, used as-is by both scripts;
`--split-file` is ignored) and the corrected ASAS-SN photometry.
`pc/ZTF-ATLAS-ASASSN-CSS-LINEAR-PTF-isect` adds CSS, LINEAR and PTF. Per
star the rows carry the light curves (per band: `mjd`, `mag`, `mag_unc`,
`clean` flag, mag system), the catalog `period`, `class_str` and the coarser
`superclass_str`, and Gaia DR3 astrometry (ra/dec, parallax, proper motion,
radial velocity, each with errors). The legacy `hibb/...` HF Hub datasets
still work (need `HF_TOKEN`; split from `data/splits/`).

## Contrastive baseline

`pretrain_contrastive.py` reuses every flag of `pretrain.py` (it imports it
and registers `--mode contrastive`) and swaps the objective for the
multi-positive NT-Xent: every view of a star (globals + locals) is an anchor
whose positives are all other views of the same star and whose negatives are
the other stars' views (across GPUs under DDP; oversampled duplicates of a
star are positives, not false negatives, via the batch `star_id`). Extra
flags: `--temperature` (0.2), `--no-anchor-locals`; `--projector` defaults
to `mlp`. Logs `train/pos_sim`, `train/neg_sim` instead of the SIGReg terms.
Checkpoints are consumed by `downstream.py` unchanged.

## Downstream protocol (pair-ztf-asassn)

Encoder pretrained on ZTF + ASAS-SN (ATLAS held out, TESS excluded) on the
PC_matches unified split; `downstream.py` embeds every star once per survey
on the capped eval windows and fits one probe per *source* survey on train
stars, selected on that survey's val stars, then tests on the **test** stars
of ZTF, ATLAS and ASASSN separately (3x3 transfer matrix). Probe: 2x512 GELU
MLP with early stopping (`--probe mlp`, default; `linear` = logistic /
ridge). Tasks: 16-class `class_str` classification (macro F1, balanced acc,
top-1) and `log10(period)` regression (R², median |Δ| dex). The handcrafted
baseline (`handcrafted_features.py`, `jobs/handcrafted.slurm`) computes the
StarEmbed FATS + `light_curve` features on the identical windows, band-
averaged to a 67-d survey-agnostic vector, and runs through the same probes.

## Wavelength-encoding ablation

`pretrain.py` / `downstream.py` share the flags below (pass the same ones to
both, e.g. through `WAVE_FLAGS` in the SLURM jobs). Defaults reproduce the
original setup (effective wavelength as one rotary axis, CLS pooling).

| flag | choices | meaning |
|---|---|---|
| `--wave-pos` | `eff` `pct` `pct-nd` `width` `index` `none` | wavelength as rotary **position**: effective wavelength (1 axis); 10/50/90% transmission percentiles as 3 axial axes; the same 3 coords encoded jointly with nD-RoPE (simplex block); centre + log-bandwidth (2 axes); raw band index; time only |
| `--wave-ctx` | `none` `pct` `onehot` | wavelength as token **content** (extra input channels): log percentile triple, or a one-hot band id (non-transferable ceiling) |
| `--pool` | `cls` `mean` | CLS sits at position 0 so absolute wavelength is visible to it; mean pooling sees relative wavelength only |
| `--shuffle-wave` | | sanity control: permute the band -> descriptor mapping (a model that matches the unshuffled run is not using the filter shape) |
| `--time-frac`, `--p-rope`, `--head-dim`, `--wavelength-scale` | | rotary layout knobs (time keeps `time-frac` of each head; the wavelength axes share the rest) |

Percentiles come from `data/filter_percentiles.json`; the rotary layout for
each mode is logged at startup (`wave encoding: ...`).

## Running on another cluster

Nothing in the Python code is tied to this machine; three things are:

- **Data path.** `pc/<name>` datasets are read from `$PC_MATCHES_ROOT`
  (default `/projects/bfrf/data/PC_matches`). Export it, or pass
  `--data-root <dir>` to `pretrain.py`, `pretrain_contrastive.py`,
  `downstream.py` and `handcrafted_features.py`. The directory must hold the
  saved `DatasetDict`s (`ZTF-ATLAS-ASASSN-isect/`, ...). Their `split` column
  is the train/validation/test split, so every cluster evaluates on identical
  stars with no split file to copy.
- **SLURM headers.** `jobs/*.slurm` hard-code `--account`, `--partition`,
  `--chdir` and the `../../.venv` activation for DeltaAI (GH200). Override on
  the command line (`sbatch --account=... --partition=... --chdir=$PWD
  jobs/pair-ztf-asassn.slurm`) or edit the headers. Everything after the
  header is plain `python ...` with relative paths.
- **Extra packages** for the handcrafted baseline only:
  `pip install -e ".[lightcurves]"` from the repo root (`light-curve` has
  wheels for x86_64 and aarch64; FATS is installed from its Python-3 fork).

`runs/` and `logs/` are gitignored, so checkpoints and results stay local to
each cluster; copy `runs/<project>/<run id>/checkpoints/` to evaluate a
checkpoint elsewhere (`downstream.py --ckpt`). wandb logging needs
`wandb login` once per cluster, or `--no-wandb`.

## Hugging Face token

The datasets are private. No token is stored in the repo; scripts resolve it
as `--hf-token` flag > `HF_TOKEN` env var > the login cached by
`hf auth login` (`~/.cache/huggingface/token`). The cluster account already
has the cached login, so the SLURM jobs need no extra setup.
