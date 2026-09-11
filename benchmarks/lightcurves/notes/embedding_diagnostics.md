# Embedding-space diagnostics — cs3079146 (LeJEPA vs supervised)

Mathematically grounded measurements of the frozen embedding spaces, computed
from `downstream_embeddings.npz` of both checkpoints (12,191 stars, train+val,
mean-of-4-eval-view embeddings per star per survey; ZTF / ASASSN / ATLAS,
ATLAS = pretraining holdout). Date: 2026-09-10.

---

## 1. Alignment & Uniformity (Wang & Isola 2020)

*Theory*: for unit-norm embeddings, contrastive-family objectives provably
optimize two quantities — **alignment** `E‖z_a − z_b‖²` over positive pairs
(same star through two surveys; 0 = identical, 2 = orthogonal) and
**uniformity** `log E exp(−2‖z_i − z_j‖²)` over random pairs (more negative =
better spread on the hypersphere). Good linear separability needs both.

| pair | LeJEPA align ↓ | LeJEPA unif ↓ | Superv. align ↓ | Superv. unif ↓ |
|---|---|---|---|---|
| ZTF–ASASSN | **0.653** | −3.168 | 0.974 | −2.013 |
| ZTF–ATLAS | **0.929** | −3.036 | 0.630* | −1.849 |
| ASASSN–ATLAS | **0.959** | −3.142 | 1.056 | −1.955 |

LeJEPA has tighter positive pairs on the trained pair (0.65 ≈ cosine 0.67) at
much better uniformity. (*The supervised ZTF–ATLAS alignment looks good only
because its CE training view included ATLAS data — ATLAS is not zero-shot for
the supervised model.)

## 2. Shared linear subspace across surveys (SVCCA)

*Theory*: mean canonical correlation of the top-64 whitened dimensions
between survey A's and survey B's embeddings of the same stars — a
rotation-invariant measure of how much linear structure two representations
share (Raghu et al. 2017).

| pair | LeJEPA meanCCA@64 | Supervised meanCCA@64 |
|---|---|---|
| ZTF–ASASSN | **0.463** | 0.258 |
| ZTF–ATLAS | 0.254 | 0.271 |
| ASASSN–ATLAS | 0.311 | 0.248 |

The trained-pair shared subspace is ~2× larger for LeJEPA; zero-shot ATLAS
pairs sit at ~0.25–0.31 for both.

## 3. Procrustes decomposition — the mechanism behind the transfer gap

*Method*: residual `‖A·R − B‖²/‖B‖²` after the best orthogonal map R
(fit on same-star correspondences, unit-norm embeddings). Distinguishes
"clouds differ by a rigid rotation" from "clouds share no structure".

| pair | LeJEPA resid | Supervised resid |
|---|---|---|
| ZTF–ASASSN | 0.643 | **0.580** |
| ZTF–ATLAS | 0.894 | **0.478** |
| ASASSN–ATLAS | 0.880 | **0.514** |

Supervised drops from raw misalignment ~0.97 to ~0.5 after rotation — its
survey clouds are near-copies living in *survey-specific coordinate frames*.
LeJEPA barely improves under rotation because it is already in one shared
frame (raw 0.65 → 0.64).

**Causal test** (probe fit on ZTF train stars, evaluated on ASASSN val stars,
with/without rotating ASASSN embeddings into the ZTF frame via an orthogonal
map fit on *training* stars):

| model | period R² raw | period R² rotated | macro-F1 raw | rotated |
|---|---|---|---|---|
| Supervised | −0.072 | **+0.163** | 0.289 | 0.269 |
| LeJEPA | +0.524 | +0.527 | 0.300 | 0.299 |

Interpretation: a third of the supervised transfer failure is a coordinate-
frame mismatch (fixable with paired-star supervision — exactly what LeJEPA
consumes during pretraining); the remaining gap (0.16 vs 0.52) is genuinely
missing shared information. LeJEPA is invariant to the rotation because it
needs none.

*(Note: raw numbers here differ slightly from the downstream tables because
this probe uses unit-normalised embeddings rather than StandardScaler.)*

## 4. Spectral diagnostics (RankMe, α-ReQ) and Gaussianity

*Theory*: **RankMe** (Garrido et al. 2023) = exp(entropy of normalized
covariance eigenvalues) — a label-free effective-rank predictor of probe
performance. **α** = power-law decay index of the eigenspectrum (α ≈ 1 is
the empirically good regime; Agrawal et al. 2022). **Gaussianity** = median
|excess kurtosis| over 512 random 1-D projections (0 = Gaussian) — the same
family of statistic SIGReg penalises during training.

| model | RankMe (of 256) | α decay | top-1 eigval share | med \|kurtosis\| |
|---|---|---|---|---|
| LeJEPA | 17.5 | 3.50 | 13.8 % | 1.303 |
| Supervised | 25.3 | 1.95 | 16.3 % | 0.504 |

**Honest finding**: LeJEPA's *eval-view* embeddings are far from the
isotropic Gaussian that SIGReg targets (and that the LeJEPA paper argues is
the optimal embedding prior for worst-case downstream risk) — only ~18
effective dimensions, steep spectrum, heavy-tailed projections. Likely
causes: λ_ref = 0.02 is a gentle regularisation, and SIGReg acts on
augmented *training* views, not the capped eval views measured here. This is
concrete, theory-linked headroom: ablating λ_ref (or checking the CSS run's
spectrum) is the principled next experiment.

## 5. Related numbers from other analyses (context)

- Pairwise instrument separability, same embeddings (3-fold logistic;
  0.5 = indistinguishable): ZTF–ASASSN 0.786, ZTF–ATLAS 0.807,
  ATLAS–ASASSN 0.882; with CSS: ZTF–CSS 0.890, ASASSN–CSS 0.882,
  ATLAS–CSS 0.923 (12k stars, mean-of-view embeddings; not comparable to
  the wandb `inst/sep_acc` ≈ 0.60 which uses 1.2k single-view val stars).
- Joint-space 5-NN (146k segments): class purity@5 = 0.785, kNN majority
  top-1 = 0.842 (vs 0.715 marginal), neighbour period agreement
  med |Δlog₁₀P| = 0.112 dex; kNN period prediction med |err| = 0.087 dex.
- Cross-survey same-star retrieval hit@5 = 2.0 % (~67× chance), inversely
  tied to class crowding (LPV 10.6 %, EW 1.3 %).

## Reproduce

All Section 1–4 numbers: inline scripts over
`Cross-Survey-LC/cs3079146-{lejepa,supervised}/checkpoints/downstream_embeddings.npz`
(spectral: SVD of centred embeddings; CCA: SVD of whitened-basis overlap;
Procrustes: SVD of A᙭B; alignment/uniformity: unit-norm distances).
Section 5: `knn_metrics.json`, `umap_css.py` output logs.

## References

- Wang & Isola 2020, *Understanding Contrastive Representation Learning
  through Alignment and Uniformity on the Hypersphere* (ICML).
- Raghu et al. 2017, *SVCCA* (NeurIPS).
- Garrido et al. 2023, *RankMe: Assessing the Downstream Performance of
  Pretrained SSL Representations by Their Rank* (ICML).
- Agrawal et al. 2022, *α-ReQ: Assessing Representation Quality by
  Eigenspectrum Decay*.
- Balestriero & LeCun 2025, *LeJEPA* (SIGReg / isotropic-Gaussian optimality).
