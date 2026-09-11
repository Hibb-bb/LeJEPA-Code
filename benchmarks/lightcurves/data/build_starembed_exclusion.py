"""Build the StarEmbed evaluation-star exclusion list.

Crossmatches the StarEmbed ZTF_40k *validation*, *test* and *anom* splits
against our training datasets and writes ``starembed_exclude_gaia.txt`` (one
``gaia_dr3_source_id`` per line). Pretraining with
``--exclude-stars-file data/starembed_exclude_gaia.txt`` then guarantees the
encoder never sees those stars through ANY survey, so frozen-encoder results
on the StarEmbed benchmark carry no transductive contamination (their
*train* split stays usable for pretraining, as is standard in SSL).

Matching: exact ``sourceid`` string match into hibb/CSSxPC (both derive
from the Drake+14 Catalina catalog), plus a 2-arcsec positional match into
every dataset's gaia coordinates as a safety net.
"""

import os
from pathlib import Path

import numpy as np
from huggingface_hub import get_token
from datasets import load_dataset
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent  # data/
ROOT = HERE.parent  # benchmarks/lightcurves
TOL_ARCSEC = 2.0
DATASETS = ["hibb/tess-ztf-atlas-asassn-isect", "hibb/CSSxPC"]
SPLITS = ["validation", "test", "anom"]


def unit(ra, dec):
    ra, dec = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def main():
    tok = os.environ.get("HF_TOKEN") or get_token()  # flag > env > `hf auth login`

    se_ids, se_ra, se_dec = [], [], []
    for split in SPLITS:
        d = load_dataset("StarEmbed/ZTF_40k", split=split)
        se_ids += list(d["sourceid"])
        se_ra += list(d["ra"])
        se_dec += list(d["dec"])
    se_id_set = set(se_ids)
    se_tree = cKDTree(unit(se_ra, se_dec))
    tol = 2.0 * np.sin(np.radians(TOL_ARCSEC / 3600) / 2)  # chord distance
    print(f"StarEmbed {'+'.join(SPLITS)}: {len(se_ids)} stars")

    excl = set()
    for repo in DATASETS:
        ds = load_dataset(repo, split="train", token=tok)
        gaia = np.array(ds["gaia_dr3_source_id"])
        n0 = len(excl)
        if "sourceid" in ds.column_names:
            m = np.isin(np.array(ds["sourceid"]), list(se_id_set))
            excl.update(int(g) for g in gaia[m] if g)
        pts = unit(ds["gaia_dr3_ra"], ds["gaia_dr3_dec"])
        d, _ = se_tree.query(pts, k=1, distance_upper_bound=tol)
        excl.update(int(g) for g in gaia[np.isfinite(d)] if g)
        print(f"{repo}: +{len(excl) - n0} newly matched (total {len(excl)})")

    out = HERE / "starembed_exclude_gaia.txt"
    out.write_text("\n".join(str(g) for g in sorted(excl)) + "\n")
    print(f"wrote {out}: {len(excl)} gaia ids")


if __name__ == "__main__":
    main()
