"""Offline downstream evaluation of a pretrained LeJEPA light-curve encoder.

Loads a Lightning checkpoint produced by ``pretrain.py``, embeds
every object once per instrument (mean of ``--n-eval-views`` un-augmented
capped-span eval views, i.e. the same 500-1500-day windows the encoder was
pretrained on) and runs two frozen-encoder downstream tasks:

1. **Variable-star classification** — probe on the embeddings (``--probe
   mlp``: 2-hidden-layer MLP selected on the source survey's val split;
   ``linear``: multinomial logistic regression), target ``class_str``
   (the fine 16-class label). The label is a property of the *star*
   (this is an intersection dataset: the same objects appear in every
   survey), so the class taxonomy is identical across surveys by
   construction; only per-survey coverage differs.
2. **Period regression** — the same probe family (MLP / ridge) on
   ``log10(period)`` (objects with a finite catalog period only).

The same probes run on a **handcrafted-feature baseline** instead of a
checkpoint with ``--features <npz>`` (built by ``handcrafted_features.py``
on identical eval windows), so learned embeddings and FATS/light_curve
features are compared under one protocol.

Both tasks are evaluated as a full cross-survey transfer matrix: for each
source instrument A the probe is fit on the embeddings of the *pretraining
train-split* stars seen through A, then evaluated on the *val* and *test*
stars seen through every instrument B (the star-level gaia-id split from
``data/splits/``, shared with pretraining, so the encoder never saw a
val/test star and the probe never saw one through any survey). Val is for
model selection; test is the number to report. ``A == B`` rows are the
in-survey reference; ``B == <--holdout-instrument>`` is the zero-shot
instrument transfer number.

Usage (matches the debug run's geometry)::

    python downstream.py \
        --ckpt runs/Cross-Survey-LC/cs3079146-lejepa/checkpoints/epoch=699-step=59500.ckpt \
        --dataset hibb/tess-ztf-atlas-asassn-isect \
        --exclude-instrument TESS --holdout-instrument ATLAS \
        --width 360 --depth 6
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# ``pretrain.py`` lives next to this script (sys.path[0]); import it so the
# view pipeline, record loader and model builder stay single-sourced.
import pretrain as ladder  # noqa: E402


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
class EvalViewDataset(torch.utils.data.Dataset):
    """One item = one seeded, un-augmented eval view of one object.

    ``indices`` are positions into ``records`` (only objects that actually
    have the instrument); each object contributes ``n_views`` items whose
    windows differ only through the seeded rng, so the per-object embedding
    (the mean over its views) is deterministic.
    """

    def __init__(self, records, indices, bands, cfg, n_views, seed):
        self.records = records
        self.indices = indices
        self.bands = tuple(bands)
        self.cfg = cfg
        self.n_views = n_views
        self.seed = seed

    def __len__(self):
        return len(self.indices) * self.n_views

    def __getitem__(self, k):
        obj, rep = divmod(k, self.n_views)
        rec = self.records[self.indices[obj]]
        rng = np.random.default_rng((self.seed, self.indices[obj], rep))
        view = ladder.make_eval_view(rec, self.cfg, rng, bands=self.bands,
                                     augment=False)
        return obj, view


def _collate_views(batch):
    objs = torch.tensor([b[0] for b in batch], dtype=torch.long)
    return objs, ladder._tokenize_batch([b[1] for b in batch])


@torch.no_grad()
def embed_instrument(backbone, records, indices, inst, cfg, args, device,
                     head=None):
    """[len(indices), width] mean-of-views embedding for one instrument.

    ``head`` (the pretrained LeJEPA predictor, eval mode) is applied to every
    view's encoder output *before* averaging over views, exactly as in
    pretraining, when this instrument is the ``--predictor-instrument``.
    """
    ds = EvalViewDataset(records, indices, ladder.INST_BANDS[inst], cfg,
                         args.n_eval_views, args.seed)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=_collate_views,
    )
    out = torch.zeros(len(indices), backbone.embed_dim)
    for objs, (v, p, m) in dl:
        z = backbone(v.to(device), p.to(device), m.to(device))
        if head is not None:
            z = head(z)
        out.index_add_(0, objs, z.float().cpu())
    return (out / args.n_eval_views).numpy()


def load_backbone(args, device):
    """Rebuild the pretrained rung and load the checkpoint's encoder.

    With ``--predictor-instrument`` the LeJEPA predictor is rebuilt and loaded
    too and stored as ``backbone.eval_predictor`` (eval mode: BatchNorm uses
    its running statistics), see :func:`embed_instrument`.
    """
    pred_inst = getattr(args, "predictor_instrument", None)
    if pred_inst:
        ladder.PREDICTOR_INST = pred_inst
        ladder.PREDICTOR_HIDDEN = args.predictor_hidden
    model = ladder.build_model(
        args.width, args.width, lamb=0.0, n_slices=args.n_slices,
        depth=args.depth, mode="lejepa", projector=args.projector,
    )
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Only the encoder is evaluated: load backbone.* exactly and ignore the
    # projector / predictor / SIGReg / contrastive heads (their shapes depend
    # on pretraining flags this script does not need to know).
    sd = {k[len("model.backbone."):]: v for k, v in ck["state_dict"].items()
          if k.startswith("model.backbone.")}
    missing, unexpected = model.backbone.load_state_dict(sd, strict=True)
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch')}): "
          f"{len(sd)} backbone tensors")
    backbone = model.backbone.to(device).eval()
    backbone.eval_predictor = None
    if pred_inst:
        psd = {k[len("model.predictor."):]: v for k, v in ck["state_dict"].items()
               if k.startswith("model.predictor.")}
        if not psd:
            raise RuntimeError(f"{args.ckpt} has no predictor weights")
        model.predictor.load_state_dict(psd, strict=True)
        backbone.eval_predictor = model.predictor.to(device).eval()
        print(f"loaded predictor ({len(psd)} tensors): {pred_inst} embeddings "
              f"= predictor(encoder output)")
    return backbone


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
EVAL_SPLITS = ("val", "test")
PROBE = "mlp"          # --probe
MLP_CFG = dict(hidden=512, layers=2, dropout=0.1, lr=1e-3, weight_decay=1e-4,
               epochs=300, batch_size=256, patience=30,
               # "balanced": cross-entropy weighted by n / (K * n_c), the same
               # rule as sklearn's class_weight="balanced" used by the random
               # forest probe, so both probes treat the 79% EW/EB imbalance
               # alike; "none": plain cross-entropy. (--class-weight)
               class_weight="balanced")


def balanced_class_weights(y, n_classes, power=1.0):
    """``(n / (K_present * n_c)) ** power`` per class, rescaled so the mean
    sample weight is 1; classes absent from ``y`` get 0. ``power=1`` is
    sklearn's "balanced"; ``power=0.5`` ("sqrt") is the tempered version that
    trades less precision on the rare classes for their recall."""
    cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = cnt > 0
    w = np.zeros(n_classes, dtype=np.float64)
    w[present] = (cnt.sum() / (present.sum() * cnt[present])) ** power
    w *= cnt.sum() / (w * cnt).sum()
    return w.astype(np.float32)


class _Imputer:
    """Non-finite entries -> train-column median (handcrafted features have
    NaN for bands/features that could not be computed), then standardise."""

    def __init__(self, Xtr):
        Xtr = np.asarray(Xtr, dtype=np.float64)
        finite = np.isfinite(Xtr)
        self.med = np.array([np.median(c[f]) if f.any() else 0.0
                             for c, f in zip(Xtr.T, finite.T)])
        Z = self._fill(Xtr)
        self.mu = Z.mean(0)
        self.sd = Z.std(0) + 1e-8

    def _fill(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        bad = ~np.isfinite(X)
        X[bad] = np.broadcast_to(self.med, X.shape)[bad]
        return X

    def transform(self, X):
        # Clip: handcrafted features (CAR_*) have 1e12-scale outliers.
        Z = (self._fill(X) - self.mu) / self.sd
        return np.clip(Z, -20.0, 20.0).astype(np.float32)


class MLPProbe:
    """Small MLP probe (classification or regression) with early stopping.

    Trained with AdamW on the standardised inputs; the epoch with the best
    val metric (macro F1 / negative MSE on the *source* instrument's val
    stars) is kept, so model selection never touches test stars or other
    instruments.
    """

    def __init__(self, task, n_out, device, seed=0, **cfg):
        import torch.nn as nn
        self.task, self.device = task, device
        c = {**MLP_CFG, **cfg}
        self.c = c
        torch.manual_seed(seed)
        dims = [None] + [c["hidden"]] * c["layers"]
        self.n_out = n_out
        self._dims = dims
        self.net = None

    def _build(self, n_in):
        import torch.nn as nn
        layers, d = [], n_in
        for _ in range(self.c["layers"]):
            layers += [nn.Linear(d, self.c["hidden"]), nn.GELU(),
                       nn.Dropout(self.c["dropout"])]
            d = self.c["hidden"]
        layers.append(nn.Linear(d, self.n_out))
        return nn.Sequential(*layers).to(self.device)

    def _score(self, X, y):
        p = self.predict(X)
        if self.task == "cls":
            from sklearn.metrics import f1_score
            return f1_score(y, p, average="macro")
        return -float(np.mean((p - y) ** 2))

    def fit(self, Xtr, ytr, Xva, yva):
        import copy
        import torch.nn.functional as F
        self.net = self._build(Xtr.shape[1])
        Xt = torch.from_numpy(Xtr).to(self.device)
        yt = (torch.from_numpy(ytr).long() if self.task == "cls"
              else torch.from_numpy(ytr.astype(np.float32))).to(self.device)
        if self.task == "reg":
            self.y_mu, self.y_sd = float(ytr.mean()), float(ytr.std() + 1e-8)
            yt = (yt - self.y_mu) / self.y_sd
        cw = None
        mode = self.c.get("class_weight", "none")
        if self.task == "cls" and mode in ("balanced", "sqrt"):
            cw = torch.from_numpy(balanced_class_weights(
                ytr, self.n_out, 1.0 if mode == "balanced" else 0.5)).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.c["lr"],
                                weight_decay=self.c["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.c["epochs"])
        best, best_state, bad = -np.inf, None, 0
        n, bs = len(Xt), self.c["batch_size"]
        g = torch.Generator(device="cpu").manual_seed(0)
        for ep in range(self.c["epochs"]):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            for k in range(0, n, bs):
                idx = perm[k:k + bs].to(self.device)
                out = self.net(Xt[idx])
                loss = (F.cross_entropy(out, yt[idx], weight=cw) if self.task == "cls"
                        else F.mse_loss(out.squeeze(-1), yt[idx]))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            if len(Xva):
                score = self._score(Xva, yva)
                if score > best:
                    best, bad = score, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    bad += 1
                    if bad >= self.c["patience"]:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val, self.epochs_run = best, ep + 1
        return self

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        outs = []
        for k in range(0, len(X), 4096):
            xb = torch.from_numpy(X[k:k + 4096]).to(self.device)
            outs.append(self.net(xb).float().cpu())
        out = torch.cat(outs) if outs else torch.zeros(0, self.n_out)
        if self.task == "cls":
            return out.argmax(1).numpy()
        return (out.squeeze(-1).numpy() * self.y_sd + self.y_mu)


def _fit_probe(task, Xtr, ytr, Xva, yva, n_classes, device, seed):
    """Fit the configured probe; returns ``predict(X) -> np.ndarray``."""
    if PROBE == "mlp":
        m = MLPProbe(task, n_classes if task == "cls" else 1, device, seed)
        m.fit(Xtr, ytr, Xva, yva)
        return m.predict
    if PROBE == "rf":
        # StarEmbed-style tabular baseline for handcrafted features: 500
        # trees, balanced class weights, no val-based selection needed.
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        kw = dict(n_estimators=500, min_samples_leaf=2, n_jobs=-1,
                  random_state=seed)
        if task == "cls":
            return RandomForestClassifier(class_weight="balanced", **kw).fit(Xtr, ytr).predict
        return RandomForestRegressor(**kw).fit(Xtr, ytr).predict
    if task == "cls":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(Xtr, ytr)
        return clf.predict
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0).fit(Xtr, ytr).predict


# Classes treated as anomalies: never seen by the classification probe and
# not scored (they stay in pretraining, which is label-free, and in the
# period regression). --anomaly-class overrides; "none" keeps every class.
ANOMALY_CLASSES = ("RRab-Blazhko", "EW/EB-OC", "RRc-Blazhko")


def fit_eval_classification(emb, results, label_map, device="cpu", seed=0):
    """One probe per source instrument, evaluated on every target.

    ``label_map`` maps the record label index -> compact probe class index,
    or -1 for anomaly classes, whose stars are dropped from the probe's
    train / val / test rows. Keys are ``cls/{src}_to_{tgt}`` for the test
    split and ``cls/val/{src}_to_{tgt}`` for val.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    n_classes = int(label_map.max()) + 1

    def rows(inst, sp):
        y = label_map[emb[inst][f"{sp}_label"]]
        keep = y >= 0
        return emb[inst][sp][keep], y[keep]

    for src in ladder.INSTRUMENTS:
        Xtr, ytr = rows(src, "train")
        Xva, yva = rows(src, "val")
        scaler = _Imputer(Xtr)
        predict = _fit_probe(
            "cls", scaler.transform(Xtr), ytr, scaler.transform(Xva), yva,
            n_classes, device, seed,
        )
        for tgt, sp in _targets(src):
            Xte, yte = rows(tgt, sp)
            if len(yte) == 0 or _skip(src, tgt):
                results[_key("cls", sp, src, tgt)] = _empty_cell(len(ytr), "cls")
                continue
            pred = predict(scaler.transform(Xte))
            results[_key("cls", sp, src, tgt)] = {
                "n_train": len(ytr), "n_test": len(yte),
                "top1": float(accuracy_score(yte, pred)),
                "balanced": float(balanced_accuracy_score(yte, pred)),
                "macro_f1": float(f1_score(yte, pred, average="macro")),
            }


def fit_eval_period(emb, results, device="cpu", seed=0):
    """Probe on log10(period) per source instrument, all targets.

    Returns ``{f"{src}_to_{tgt}": (y_true, y_pred)}`` on the test split for
    the scatter figure.
    """
    from sklearn.metrics import r2_score

    preds = {}
    for src in ladder.INSTRUMENTS:
        ok = np.isfinite(emb[src]["train_logp"])
        Xtr, ytr = emb[src]["train"][ok], emb[src]["train_logp"][ok]
        okv = np.isfinite(emb[src]["val_logp"])
        scaler = _Imputer(Xtr)
        predict = _fit_probe(
            "reg", scaler.transform(Xtr), ytr,
            scaler.transform(emb[src]["val"][okv]), emb[src]["val_logp"][okv],
            None, device, seed,
        )
        for tgt, sp in _targets(src):
            ok = np.isfinite(emb[tgt][f"{sp}_logp"])
            Xte, yte = emb[tgt][sp][ok], emb[tgt][f"{sp}_logp"][ok]
            if len(yte) == 0 or _skip(src, tgt):
                yte = yte[:0]
                results[_key("per", sp, src, tgt)] = _empty_cell(len(ytr), "per")
                if sp == "test":
                    preds[f"{src}_to_{tgt}"] = (yte, yte)
                continue
            pred = predict(scaler.transform(Xte))
            err = np.abs(pred - yte)
            if sp == "test":
                preds[f"{src}_to_{tgt}"] = (yte, pred)
            results[_key("per", sp, src, tgt)] = {
                "n_train": int(len(ytr)), "n_test": int(len(yte)),
                "r2": float(r2_score(yte, pred)),
                "rmse_dex": float(np.sqrt(np.mean((pred - yte) ** 2))),
                "med_abs_dex": float(np.median(err)),
                "frac_within_0.1dex": float((err < 0.1).mean()),
            }
    return preds


def _empty_cell(n_train, task):
    """Metrics for a (src, tgt, split) cell with no test objects."""
    keys = (["top1", "balanced", "macro_f1"] if task == "cls"
            else ["r2", "rmse_dex", "med_abs_dex", "frac_within_0.1dex"])
    return {"n_train": int(n_train), "n_test": 0, **{k: float("nan") for k in keys}}


IN_SURVEY_ONLY = False  # --in-survey-only


def _targets(src=None):
    return [(t, sp) for sp in EVAL_SPLITS for t in ladder.INSTRUMENTS]


def _skip(src, tgt):
    """--in-survey-only: cross-survey cells are not evaluated (NaN)."""
    return IN_SURVEY_ONLY and tgt != src


def _key(task, split, src, tgt):
    return (f"{task}/{src}_to_{tgt}" if split == "test"
            else f"{task}/{split}/{src}_to_{tgt}")


def period_scatter_figure(preds, results):
    """Pred-vs-true log10(P) grid: one panel per (fit-on, tested-on) pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    insts = ladder.INSTRUMENTS
    n = len(insts)
    fig, axes = plt.subplots(n, n, figsize=(3.6 * n, 3.6 * n),
                             sharex=True, sharey=True)
    ys = [y for y, _ in preds.values() if len(y)]
    lo = min(y.min() for y in ys) if ys else 0.0
    hi = max(y.max() for y in ys) if ys else 1.0
    pad = 0.05 * (hi - lo)
    for i, src in enumerate(insts):
        for j, tgt in enumerate(insts):
            ax = axes[i, j]
            y, p = preds[f"{src}_to_{tgt}"]
            r = results[f"per/{src}_to_{tgt}"]
            ax.scatter(y, p, s=4, alpha=0.3, linewidths=0)
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_title(f"fit {src} → test {tgt}\n"
                         f"R²={r['r2']:.2f}  med|Δ|={r['med_abs_dex']:.2f} dex",
                         fontsize=9)
            if i == n - 1:
                ax.set_xlabel("true log10 P [d]")
            if j == 0:
                ax.set_ylabel("pred log10 P [d]")
    for ax in axes.ravel():
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("Period regression on frozen embeddings (test split)")
    fig.tight_layout()
    return fig


# StarEmbed (arXiv:2510.06200) benchmarks seven CSPVS classes on ZTF. Map our
# 16-class VSX-style taxonomy onto them for the class-highlight UMAP figure
# (classes with no analogue — Cepheids, DSCT, ELL, PCEB — stay gray).
# "RS CVn" has no exact match; ROT (spotted rotational variables) is the
# closest VSX analogue.
STAREMBED_CLASSES = {
    "EW": ("EW/EB", "EW/EB-OC"),
    "EA": ("EA",),
    "RRab": ("RRAB", "RRab-Blazhko"),
    "RRc": ("RRC", "RRc-Blazhko"),
    "RRd": ("RRD",),
    "RS CVn~ROT": ("ROT",),
    "LPV": ("LPV",),
}


def starembed_umap_figure(emb, idx_to_label, seed=0):
    """Per-instrument UMAP with one panel per StarEmbed class.

    Rows = instruments (each with its own UMAP fit on that instrument's
    train+val embeddings, cosine metric); columns = the seven StarEmbed
    classes. Every panel shows all of the instrument's objects in gray and
    colours only the one class, so per-class structure is readable even for
    the rare classes.

    Args:
        emb: the per-instrument dict built in ``main`` (``train``/``val``
            embeddings and labels).
        idx_to_label: class index -> ``class_str`` name.
        seed: UMAP/PCA random state.

    Returns:
        The matplotlib figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    insts = list(ladder.INSTRUMENTS)
    classes = list(STAREMBED_CLASSES)
    fig, axes = plt.subplots(
        len(insts), len(classes),
        figsize=(3.1 * len(classes), 3.3 * len(insts)), squeeze=False,
    )
    cmap = plt.get_cmap("tab10")
    method = "UMAP"
    for r, inst in enumerate(insts):
        X = np.concatenate([emb[inst]["train"], emb[inst]["val"]])
        X = _Imputer(emb[inst]["train"]).transform(X)  # finite, standardised
        y = np.concatenate([emb[inst]["train_label"], emb[inst]["val_label"]])
        names = np.array([idx_to_label[int(c)] for c in y])
        try:
            import umap
            z = umap.UMAP(n_components=2, random_state=seed,
                          metric="cosine").fit_transform(X)
        except ImportError:
            from sklearn.decomposition import PCA
            z = PCA(2, random_state=seed).fit_transform(X)
            method = "PCA"
        for c, (se_name, ours) in enumerate(STAREMBED_CLASSES.items()):
            ax = axes[r, c]
            ax.scatter(*z.T, c="lightgray", s=4, alpha=0.4, linewidths=0)
            m = np.isin(names, ours)
            ax.scatter(*z[m].T, c=[cmap(c)], s=6, alpha=0.8, linewidths=0)
            ax.set_title(f"{inst}: {se_name} (n={int(m.sum())})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{method} per instrument, frozen embeddings "
                 "(gray = all objects, colour = one StarEmbed class)")
    fig.tight_layout()
    return fig


def print_matrix(results, task, metric, title, split="test"):
    insts = ladder.INSTRUMENTS
    print(f"\n{title} ({metric}; rows = probe fit on, cols = tested on "
          f"{split} split)")
    print(f"{'':>8}" + "".join(f"{t:>10}" for t in insts))
    for src in insts:
        row = "".join(
            f"{results[_key(task, split, src, t)][metric]:>10.3f}" for t in insts
        )
        print(f"{src:>8}" + row)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Lightning .ckpt from pretrain.py / pretrain_contrastive.py")
    ap.add_argument("--features", default=None,
                    help="instead of a checkpoint: handcrafted-feature .npz "
                         "from handcrafted_features.py (same protocol)")
    ap.add_argument("--anomaly-class", action="append", default=None,
                    help="class_str excluded from the classification probe "
                         f"and its metrics (repeatable; default "
                         f"{list(ANOMALY_CLASSES)}; 'none' = keep all)")
    ap.add_argument("--probe", choices=["mlp", "linear", "rf"], default="mlp",
                    help="downstream probe: 2x512 GELU MLP with early stopping "
                         "on the source survey's val split (default), "
                         "logistic / ridge regression, or a 500-tree random "
                         "forest (tabular baseline for handcrafted features)")
    ap.add_argument("--class-weight", choices=["balanced", "sqrt", "none"],
                    default="balanced",
                    help="MLP probe loss: 'balanced' = cross-entropy weighted by "
                         "n/(K*n_c), matching the random forest's "
                         "class_weight='balanced' (default); 'sqrt' = square "
                         "root of those weights (tempered); 'none' = unweighted")
    ap.add_argument("--in-survey-only", action="store_true",
                    help="skip cross-survey cells: each probe is evaluated "
                         "only on its own survey (off-diagonal cells are NaN)")
    # --- model geometry (must match the checkpoint) ---
    ap.add_argument("--width", type=int, default=ladder.DEFAULT_WIDTH)
    ap.add_argument("--depth", type=int, default=6)
    ladder.add_wave_args(ap)  # must match the checkpoint's pretraining flags
    # (--seed below must also equal the pretraining seed for --wave-pos pct-nd:
    # it fixes the per-head simplex rotations.)
    ap.add_argument("--projector", choices=["identity", "mlp"], default="mlp",
                    help="must match pretraining (only the backbone is loaded)")
    ap.add_argument("--n-slices", type=int, default=128)
    ap.add_argument("--predictor-instrument", default=None,
                    help="embed this survey with the checkpoint's LeJEPA "
                         "predictor applied to the encoder output (the space "
                         "its views were trained in), e.g. ASASSN; other "
                         "surveys keep the raw encoder output")
    ap.add_argument("--predictor-hidden", type=int, default=1024)
    # --- data (defaults = the debug run) ---
    ap.add_argument("--dataset", default="pc/ZTF-ATLAS-ASASSN-isect",
                    choices=sorted(ladder.INSTRUMENT_REGISTRY))
    ap.add_argument("--data-root", type=str, default=None,
                    help="PC_matches directory for pc/ datasets")
    ap.add_argument("--exclude-instrument", action="append", default=[])
    ap.add_argument("--exclude-band", action="append", default=[],
                    help="must match pretraining, e.g. i_ZTF")
    ap.add_argument("--holdout-instrument", default=None)
    ap.add_argument("--max-objects", type=int, default=0)
    ap.add_argument("--min-obs", type=int, default=8)
    # Must match pretraining so load_records yields the identical record
    # list (and therefore the identical stratified split).
    ap.add_argument("--min-train-instruments", type=int, default=2)
    ap.add_argument("--exclude-stars-file", type=str, default=None)
    ap.add_argument("--split-file", type=str, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--hf-token", type=str, default=None,
                    help="overrides HF_TOKEN env var / cached `hf auth login`")
    # --- eval views (same 500-1500 day windows as pretraining) ---
    ap.add_argument("--window-days", type=float, default=500.0)
    ap.add_argument("--window-days-max", type=float, default=1500.0)
    ap.add_argument("--min-window-obs", type=int, default=200)
    ap.add_argument("--over-budget", choices=["tail", "random"], default="tail")
    ap.add_argument("--eval-tokens", type=int, default=512)
    ap.add_argument("--norm", choices=["band", "band-global", "object", "view"],
                    default="band")
    ap.add_argument("--n-eval-views", type=int, default=4,
                    help="windows averaged into each object's embedding")
    # --- runtime ---
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output prefix (default: <ckpt dir>/downstream)")
    ap.add_argument("--wandb", default=None,
                    help="wandb project to log tables/plots to")
    ap.add_argument("--wandb-id", default=None,
                    help="resume this wandb run id (log the downstream "
                         "results into the pretraining run)")
    args = ap.parse_args()
    if (args.ckpt is None) == (args.features is None):
        ap.error("pass exactly one of --ckpt or --features")
    global PROBE, IN_SURVEY_ONLY
    PROBE = args.probe
    IN_SURVEY_ONLY = args.in_survey_only
    MLP_CFG["class_weight"] = args.class_weight

    args.hf_token = ladder.resolve_hf_token(args.hf_token)

    ladder.configure_instruments(args.dataset, args.holdout_instrument,
                                 tuple(args.exclude_instrument),
                                 tuple(args.exclude_band))
    ladder.configure_wave(args)
    cfg = ladder.ViewConfig(
        window_days=args.window_days, window_days_max=args.window_days_max,
        min_window_obs=args.min_window_obs, over_budget=args.over_budget,
        eval_tokens=args.eval_tokens, min_tokens=args.min_obs, norm=args.norm,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_backbone(args, device) if args.ckpt else None

    records, label_to_idx = ladder.load_records(args)
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    train_idx, val_idx, test_idx = ladder.split_records(records, args)
    labels = np.array([r["label"] for r in records])
    anom = list(ANOMALY_CLASSES) if args.anomaly_class is None else \
        [c for c in args.anomaly_class if c.lower() != "none"]
    unknown = [c for c in anom if c not in label_to_idx]
    if unknown:
        print(f"warning: anomaly classes not in this dataset: {unknown}")
    kept = [c for c in sorted(label_to_idx) if c not in anom]
    cls_classes = {i: c for i, c in enumerate(kept)}  # compact probe index
    label_map = np.full(len(label_to_idx), -1, dtype=np.int64)
    for i, c in cls_classes.items():
        label_map[label_to_idx[c]] = i
    n_anom = int(np.isin(labels, [label_to_idx[c] for c in anom if c in label_to_idx]).sum())
    print(f"classification: {len(kept)} classes; {n_anom} anomaly-class stars "
          f"({anom}) excluded from the probe")
    logp = np.array([np.log10(r["period"]) if r["period"] else np.nan
                     for r in records], dtype=np.float64)
    print(f"period coverage: {np.isfinite(logp).sum()}/{len(records)} objects")

    # Per-instrument object availability + class histogram (star split is
    # shared with pretraining: probes are fit on train stars, tested on
    # val/test stars the encoder never saw).
    emb = {}
    hc = np.load(args.features) if args.features else None
    for inst in ladder.INSTRUMENTS:
        if hc is not None:
            # Handcrafted features computed by handcrafted_features.py on the
            # same records / split / eval windows; verify the object lists.
            emb[inst] = {k[len(inst) + 1:]: hc[k] for k in hc.files
                         if k.startswith(f"{inst}_")}
            n = {k: len(emb[inst][f"{k}_idx"]) for k in ("train", "val", "test")}
            exp = {"train": train_idx, "val": val_idx, "test": test_idx}
            for k, idx in exp.items():
                want = [i for i in idx if ladder._has_bands(records[i], ladder.INST_BANDS[inst])]
                if list(emb[inst][f"{k}_idx"]) != want:
                    raise RuntimeError(f"{args.features}: {inst} {k} objects differ "
                                       f"from this run's records; regenerate the "
                                       f"features with the same data flags")
            print(f"{inst}: {emb[inst]['train'].shape[1]} handcrafted features "
                  f"({n['train']} train / {n['val']} val / {n['test']} test)")
            continue
        has = np.array([ladder._has_bands(r, ladder.INST_BANDS[inst])
                        for r in records])
        tr = [i for i in train_idx if has[i]]
        va = [i for i in val_idx if has[i]]
        te = [i for i in test_idx if has[i]]
        hist = {idx_to_label[c]: int(n) for c, n in
                zip(*np.unique(labels[has], return_counts=True))}
        print(f"{inst}: {has.sum()} objects ({len(tr)} train / {len(va)} val "
              f"/ {len(te)} test); classes {hist}")
        emb[inst] = {}
        for name, idx in (("train", tr), ("val", va), ("test", te)):
            head = (backbone.eval_predictor
                    if inst == args.predictor_instrument else None)
            emb[inst][name] = embed_instrument(backbone, records, idx, inst,
                                               cfg, args, device, head=head)
            emb[inst][f"{name}_label"] = labels[idx]
            emb[inst][f"{name}_logp"] = logp[idx]
            emb[inst][f"{name}_idx"] = np.array(idx)
        print(f"{inst}: embedded")

    results = {}
    fit_eval_classification(emb, results, label_map, device, args.seed)
    preds = fit_eval_period(emb, results, device, args.seed)

    for sp in EVAL_SPLITS:
        print_matrix(results, "cls", "macro_f1", "Variable-star classification", sp)
        print_matrix(results, "cls", "top1", "Variable-star classification", sp)
        print_matrix(results, "per", "r2", "Period regression (log10 days)", sp)
        print_matrix(results, "per", "med_abs_dex", "Period regression (log10 days)", sp)

    out = Path(args.out or (Path(args.ckpt).parent / "downstream" if args.ckpt
                            else Path(args.features).with_suffix("")
                            .as_posix() + "_downstream"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.json", "w") as f:
        json.dump({"args": vars(args), "classes": idx_to_label,
                   "cls_classes": cls_classes, "anomaly_classes": anom,
                   "results": results}, f, indent=2)
    np.savez_compressed(
        f"{out}_embeddings.npz",
        **{f"{inst}_{k}": v for inst, d in emb.items() for k, v in d.items()},
    )
    fig = period_scatter_figure(preds, results)
    fig.savefig(f"{out}_period_scatter.png", dpi=120)
    se_fig = starembed_umap_figure(emb, idx_to_label, seed=args.seed)
    se_fig.savefig(f"{out}_starembed_umap.png", dpi=120)
    print(f"\nwrote {out}.json, {out}_embeddings.npz, "
          f"{out}_period_scatter.png and {out}_starembed_umap.png")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb, id=args.wandb_id, dir=str(ladder.RUNS_DIR),
            resume="allow" if args.wandb_id else None,
            name=None if args.wandb_id else
            f"downstream-{Path(args.ckpt or args.features).stem}",
            config={f"downstream/{k}": v for k, v in vars(args).items()},
        )
        insts = list(ladder.INSTRUMENTS)
        for task, metric in [("cls", "macro_f1"), ("cls", "top1"),
                             ("per", "r2"), ("per", "med_abs_dex")]:
          for sp in EVAL_SPLITS:
            rows = [[src] + [results[_key(task, sp, src, t)][metric]
                             for t in insts] for src in insts]
            tag = f"{task}_{metric}" + ("" if sp == "test" else f"_{sp}")
            run.log({f"downstream/{tag}":
                     wandb.Table(columns=["fit_on \\ test_on"] + insts,
                                 data=rows)})
        run.summary.update({
            f"downstream/{cell}/{k}": v
            for cell, m in results.items() for k, v in m.items()
        })
        run.log({"downstream/period_scatter": wandb.Image(fig),
                 "downstream/starembed_umap": wandb.Image(se_fig)})
        run.finish()


if __name__ == "__main__":
    main()
