# Roadmap: from the cross-survey comparison to light-curve dynamics

Written 2026-09-18. Ordered; each step lists what it is, why, what it needs,
and where the code or reference lives. Paths are relative to
`benchmarks/lightcurves/` unless noted.

## Where we are

- Setup: pretrain on ZTF (g, r) + ASAS-SN (g, V), ATLAS held out, TESS and
  `i_ZTF` excluded, dataset `pc/ZTF-ATLAS-ASASSN-isect` with its own
  train / validation / test split. Encoder width 360, depth 6.
- 100-epoch results (`runs/Pair-ZTF-ASASSN/`):
  - LeJEPA at `--lamb-ref 0.02` is the best encoder: best in every in-survey
    cell and on period transfer (cross-survey period R² about 0.50, no negative
    cells). Larger lambda (0.2, 2.0) is monotonically worse.
  - Contrastive (multi-positive NT-Xent) is level with LeJEPA on classification
    transfer but unstable on period transfer (negative R² into ASAS-SN).
  - Handcrafted FATS + `light_curve` features with a random forest lead
    in-survey on all three surveys; they are not evaluated cross-survey.
  - Rare classes are the weak spot of the encoders: ROT and RRD are absorbed
    into EW/EB and RRC. Class-weighted probe losses do not fix it (they raise
    balanced accuracy by about 0.05 and lower macro-F1 by up to 0.07), so the
    limitation is in the embeddings.
  - The probe alone moves macro-F1 by 0.08 to 0.16 across hyper-parameters, the
    same size as the gaps between methods.
- Reporting convention: test macro-F1 of the MLP probe selected on the
  training survey's val macro-F1, one number per train -> test setting, bar
  plots, no averaging over out-of-survey cells.

## 1. Close the open items

Effort: hours, mostly queue time.

- **Predictor-output evaluation for ASAS-SN.** LeJEPA `lamb_ref 0.02` only.
  ASAS-SN is embedded as predictor(encoder output), applied per eval window and
  then averaged; ZTF and ATLAS keep the encoder output. Job
  `jobs/predictor-eval.slurm` (3171563). Report the ASAS-SN-trained row and the
  ASAS-SN-tested column. Outputs: `checkpoints/downstream_pred.*`,
  `probe_sweep_100ep_pred.json`, `selected_f1_100ep_pred.png`.
- **Decide the probe's `--class-weight` default** in `downstream.py`. It is
  currently `balanced`; the sweep says `none` or `sqrt`
  (`probe_sweep_100ep_cw.json`).

## 2. 400-epoch pretraining

Effort: about 11 hours per run on one GH200, plus the automatic downstream job.

The earlier 400-epoch submissions (3166110 to 3166112) were cancelled while
pending. Priority is LeJEPA 0.02 and contrastive; the two larger lambdas are
optional.

```
sbatch --time=14:00:00 --array=0-1 jobs/pair-ztf-asassn.slurm --epochs 400
```

Afterwards rerun the probe sweep and the figures with the 400-epoch tag
(`jobs/probe-sweep.slurm` takes the tag; the embeddings paths inside it need
pointing at the new run directories), then `analysis/plot_sweep_selected.py`,
`analysis/compare_pair.py` and `analysis/report_figs.py`.

## 3. Make the evaluation trustworthy

Effort: about a day. Runs on saved embeddings, no re-encoding.

- **Probe seeds or bootstrap confidence intervals** on every reported cell, so
  that a 0.03 difference between methods can be interpreted.
- **Superclass label as the headline metric** (`superclass_str` ships with the
  dataset; ACEP + DCEP + T2CEP become CEP), fine classes as secondary. Several
  fine classes have 0 to 2 test stars.
- **Anomaly detection** on the three excluded classes (RRab-Blazhko, EW/EB-OC,
  RRc-Blazhko): class-conditional Mahalanobis distance in the frozen embedding,
  fit per survey on train stars of the 13 normal classes, scored on val + test
  normals plus all anomaly-class stars; AUROC, average precision, recall at k.
  The saved `downstream_embeddings.npz` files still contain those stars.
  - Lee et al. 2018, *A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks* (class-conditional
    Mahalanobis).
  - Ren et al. 2021, *A Simple Fix to Mahalanobis Distance for Improving
    Near-OOD Detection* (relative Mahalanobis; suits anomalies that live inside
    a parent class).

## 4. Dynamics, stage 0: future-window latent prediction

Effort: half a day, inside the current code.

Predict the pooled embedding of a *later* window from an earlier one,
conditioned on the time gap, in projector space. Reuses the existing window
sampler (`make_views`, `--p-period-shift`), the predictor module and SIGReg in
`stable_pretraining/methods/lejepa_lightcurve.py`.

Purpose: a cheap signal on whether a dynamics objective improves period R² and
the periodic rare classes (RRD, ROT) before building stage 1.

## 5. Dynamics, stage 1: next-latent prediction as fine-tuning

Effort: 2 to 3 days including debugging.

Design agreed so far:

- **Start from the pretrained LeJEPA checkpoint** rather than training from
  scratch.
- **Query tokens instead of a causal mask.** Split a light curve at time t. The
  context before t stays fully bidirectional, exactly as pretrained. Append
  query tokens with a learned content vector and the rotary position of the
  target (its time and wavelength), so conditioning on the time gap and the band
  comes from the continuous RoPE, with no separate embedding.
- **K queries per sequence**, horizons spread log-uniformly from under one
  period to hundreds of days. Queries attend to the context but not to each
  other, so one target's position cannot leak information about another.
- **Uncertainty**: feed `mag_unc` in as an input channel (today it is only used
  for the resampling augmentation), and weight each target's loss by inverse
  variance with an intrinsic-scatter floor, normalised within each survey so
  ZTF does not dominate ASAS-SN. The tokenizer is linear in magnitude, which
  makes inverse-variance weighting the correct Gaussian likelihood at the
  input-embedding level. Soft weights, not a hard cut.
- **Views stay on.** Keep the multi-view invariance loss and batch SIGReg during
  fine-tuning, with a lower learning rate, so instrument robustness is not
  forgotten.
- **Token anti-collapse.** The view loss protects only the pooled embedding.
  Add temporal SIGReg on projected tokens (per star, along the time axis) or a
  stop-gradient on targets. The existing `EppsPulley` module accepts
  `[stars, tokens, dims]` directly; it needs masked means for padding and the
  cross-GPU averaging turned off for per-sample statistics.
- **Loss and readout** as in the paper: MSE in a projector space that is
  discarded afterwards; evaluate mean-pooled tokens from several layers, since
  the paper finds a middle layer best for classification.

Experiment arms:

1. Current LeJEPA checkpoint.
2. The same checkpoint continued for the same number of steps *without* the new
   loss (equal-compute control; 100 epochs is known to be under-trained).
3. The same checkpoint fine-tuned *with* next-latent prediction.
4. Joint training from scratch (later; the expensive arm).

Metrics: latent prediction error versus horizon; period R²; per-class F1 on the
periodic rare classes; decoded-magnitude forecasting against a per-star
harmonic (Lomb-Scargle) fit; plus the standard 3x3 transfer matrices, to check
nothing regressed.

References:

- LeNEPA: Chemeris, Jin, Balestriero, *No-Augmentation Next-Latent Prediction
  for Time-Series Representation Learning*, arXiv:2607.00958. Next-latent
  prediction with a causal backbone, no EMA or stop-gradient, temporal SIGReg,
  projector-space loss, mid-layer readout.
- LeJEPA: Balestriero and LeCun, arXiv:2511.08544 (SIGReg).
- In the repo: `stable_pretraining/backbone/romae.py`, class
  `RoMAEForPreTraining`, already does position-queried masked prediction with a
  light decoder; `stable_pretraining/methods/nepa.py` is an image NEPA reference;
  `pretrain.py` exposes `VIEW_MODES` / `FORWARDS` / `MODEL_BUILDERS` hooks so a
  new objective plugs in without touching data, probes or figures
  (`pretrain_contrastive.py` is the worked example).

## 6. Only if stage 1 helps

- **Cross-survey queries**: ASAS-SN context answering ZTF targets, which asks
  for dynamics and instrument mapping at once.
- **Larger pretraining set** from the single-survey PC_matches catalogs
  (`--min-train-instruments 1` is already supported). This is also the main
  lever for the rare classes: RRD 794 versus 151 stars, DSCT 301 versus 79,
  T2CEP 213 versus 31, ACEP 155 versus 20.
- **Cluster-balanced sampling** of pretraining batches (label-free), so neither
  objective spends most of its capacity on contact binaries.
