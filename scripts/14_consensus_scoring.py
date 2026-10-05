"""
Step 14 — Consensus scoring of DrugBank compounds per pathogen, anchored on the reference library.

A sub-model's rank is a position against the 50,000 drug-like molecules of the LazyQSAR reference
library: rank 0.65 means "beats 99% of that library". The consensus is a weighted mean of several
such positions, which is not itself a position against anything, so it is placed on the same scale
the same way:

  1. consensus of the 50,000 REFERENCE molecules   (output/12_reference/rank/{pathogen}.csv)
  2. consensus of the DrugBank compounds           (output/12_drugbank/rank/{pathogen}.csv)
     Both use the same function and the same weights: quality weights w1..w6 and w_screen from
     output/10_reports/10_reports.csv, plus the per-compound ramp w7 (0 at or below the model's
     decision cutoff, rising to 1 at rank 1). weight = mean of the eight terms;
     consensus = sum_m(rank * weight) / sum_m(weight).
  3. read where the reference consensus sits at its p50 / p90 / p99 / p99.9, and map those four
     values to 0.25 / 0.50 / 0.65 / 0.75 (piecewise linear, LazyQSAR's own rule), with a straight
     line from the last anchor to (raw 1.0 -> rank 1.0) above it
  4. push each DrugBank compound's consensus through that map; the reference molecules are pushed
     through it too, so their consensus is written next to DrugBank's

The result reads like a sub-model's rank: consensus_rank >= 0.65 means "beats 99% of drug-like
chemistry", so the cutoff keeps its meaning and about 1% of generic chemistry passes it by
construction. The order of compounds is untouched (the map is monotone).

Every column has its own anchors, because each averages a different set of models: the full
consensus and each leave-one-out consensus (all models except one), weighted and unweighted.

Outputs, one folder per pathogen: output/14_consensus/{pathogen}/
  drugbank_raw.csv                smiles | consensus_raw | consensus_raw_without_{model} ...
  drugbank_rank.csv               smiles | consensus_rank | consensus_rank_without_{model} ...
  drugbank_unweighted_raw.csv     the same from the plain mean of the ranks (no quality weights)
  drugbank_unweighted_rank.csv
  reference_raw.csv, reference_rank.csv, reference_unweighted_raw.csv, reference_unweighted_rank.csv
                                  the same four files for the 50,000 reference molecules. These are
                                  the molecules the anchors were built from, so about 1% of
                                  reference_rank is at or above 0.65 by construction: the files show
                                  the distribution the scale is defined on, not an independent test.
  anchors.json                    the anchor table of every column, the bootstrap standard error of
                                  its four anchors (diagnostic only), and a fingerprint of the
                                  models, weights and cutoffs it was built with

Step 18b ships the anchor table to the Hub models and refuses to if the fingerprint no longer
matches the reports it is shipping.

Fails (exit code 1) if a pathogen with two or more retained models is skipped for missing inputs,
so a partial run cannot pass unnoticed.

Usage:
    python scripts/14_consensus_scoring.py
    python scripts/14_consensus_scoring.py --pathogen ecoli
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.append(os.path.join(ROOT, "..", "src"))

# LazyQSAR reads these when it is imported (where the reference bundle is cached); the same
# redirection as steps 08, 09 and 12. Step 14 never downloads anything.
os.environ["HOME"] = os.path.join(REPO_ROOT, "output", "08_weights")
os.environ["LAZYQSAR_HOME"] = os.path.join(REPO_ROOT, "output", "08_weights", ".lazyqsar")
os.environ.setdefault("LAZYQSAR_REFERENCE_OFFLINE", "1")

import lazyqsar  # noqa: E402
from lazyqsar.reference import identity  # noqa: E402
from lazyqsar.reference.manifest import manifest_sha256  # noqa: E402
from lazyqsar.utils.ranking import DECISION_RANK  # noqa: E402

import consensus as cs  # noqa: E402  (src/consensus.py)
from default import QUALITY_WEIGHT_COLS, RANDOM_SEED  # noqa: E402

DEFAULT_DRUGBANK_DIR  = os.path.join(REPO_ROOT, "output", "12_drugbank")
DEFAULT_REFERENCE_DIR = os.path.join(REPO_ROOT, "output", "12_reference")
DEFAULT_OUT_DIR       = os.path.join(REPO_ROOT, "output", "14_consensus")
REPORTS_PATH          = os.path.join(REPO_ROOT, "output", "10_reports", "10_reports.csv")

# How the quality weights and the per-compound ramp w7 combine: equal terms. Change here to reweight.
W_WEIGHTS   = np.ones(len(QUALITY_WEIGHT_COLS) + 1)
N_BOOTSTRAP = 200      # resamples of the reference for the anchor standard errors (diagnostic only)
DECIMALS    = 6        # decimals written to the CSVs; the hit counts use the same rounded values
SHARE_TOL   = 0.001    # the reference share at rank >= cutoff must be 1% +/- this (a self-check)


def at_cutoff(rank: np.ndarray) -> np.ndarray:
    """True where a rank reaches the decision cutoff, judged on the value as written to the CSV.

    A rank of 0.6499996 is written as 0.650000, so whoever reads the CSV sees a hit; counting on
    the unrounded value would make the console and anchors.json disagree with the files by one.
    """
    return np.round(rank, DECIMALS) >= DECISION_RANK


def load_inputs(pathogen: str, drugbank_dir: str, reference_dir: str, reports: pd.DataFrame):
    """Read and validate everything one pathogen needs.

    Returns (inputs, None), or (None, reason) when an input file is missing. Anything wrong with
    files that DO exist raises: a consensus built on mismatched inputs is silently wrong.
    """
    src_db  = os.path.join(drugbank_dir,  "rank", f"{pathogen}.csv")
    src_ref = os.path.join(reference_dir, "rank", f"{pathogen}.csv")
    missing = [p for p in (src_db, src_ref) if not os.path.isfile(p)]
    if missing:
        return None, "missing input: " + ", ".join(missing)

    df_db, df_ref = pd.read_csv(src_db), pd.read_csv(src_ref)
    models     = [c for c in df_db.columns  if c != "smiles"]
    ref_models = [c for c in df_ref.columns if c != "smiles"]
    if models != ref_models:
        raise ValueError(f"[{pathogen}] the DrugBank and reference files score different models "
                         f"(or in a different order): {models} vs {ref_models}")

    in_reports = reports[reports["pathogen"] == pathogen].set_index("model_name")
    if set(models) != set(in_reports.index):
        raise ValueError(
            f"[{pathogen}] the rank files and 10_reports.csv list different models. Only in the files: "
            f"{sorted(set(models) - set(in_reports.index))}; only in the reports: "
            f"{sorted(set(in_reports.index) - set(models))}. Re-run 10a, then steps 12a and 12b.")

    for label, df in (("DrugBank", df_db), ("reference", df_ref)):
        nan = df[models].isna().sum()
        if nan.any():
            raise ValueError(f"[{pathogen}] NaN predictions in the {label} file: {nan[nan > 0].to_dict()}")
        lo, hi = df[models].to_numpy().min(), df[models].to_numpy().max()
        if lo < -1e-6 or hi > 1.0 + 1e-6:
            raise ValueError(f"[{pathogen}] the {label} file is not on the [0,1] rank scale "
                             f"(min={lo:.3f}, max={hi:.3f}); step 14 needs the 'rank' predict type.")

    n_ref = identity.default_n()
    ref_smiles = pd.read_csv(identity.reference_dir() / identity.smiles_filename())["smiles"]
    if len(df_ref) != n_ref or df_ref["smiles"].tolist() != ref_smiles.tolist():
        raise ValueError(f"[{pathogen}] the reference file is not the {n_ref:,} molecules of the cached "
                         f"reference library ({identity.REFERENCE_ID}), in the same order: the anchors "
                         "would describe the wrong molecules.")

    w_quality = in_reports.loc[models, QUALITY_WEIGHT_COLS].to_numpy(dtype=float)
    cutoffs   = in_reports.loc[models, "decision_cutoff_rank"].to_numpy(dtype=float)
    if np.isnan(w_quality).any() or np.isnan(cutoffs).any():
        raise ValueError(f"[{pathogen}] missing quality weights or cutoffs in 10_reports.csv")

    return {
        "models": models, "smiles": df_db["smiles"], "reference_smiles": df_ref["smiles"],
        "drugbank": df_db[models].to_numpy(dtype=float),
        "reference": df_ref[models].to_numpy(dtype=float),
        "w_quality": w_quality, "cutoffs": cutoffs, "reference_file": src_ref,
    }, None


def score_variant(inp: dict, weighted: bool, boot_idx: np.ndarray):
    """The full consensus and every leave-one-out consensus, for one variant (weighted or not).

    Returns (columns, tables). `columns[dataset]["raw" | "rank"]` holds the consensus columns of that
    dataset ("drugbank" or "reference") on the raw and on the rank scale; `tables` holds the anchor
    table of each column. Reference and DrugBank go through the same function, so the anchors
    describe exactly the scores they are applied to.
    """
    models, m = inp["models"], len(inp["models"])
    subsets = [("full", list(range(m)))] + [
        (model, [j for j in range(m) if j != i]) for i, model in enumerate(models)
    ]

    def consensus(ranks: np.ndarray, idx: list) -> np.ndarray:
        if weighted:
            return cs.weighted_consensus(ranks[:, idx], inp["w_quality"][idx], inp["cutoffs"][idx], W_WEIGHTS)
        return cs.plain_consensus(ranks[:, idx])

    columns = {d: {"raw": {}, "rank": {}} for d in ("drugbank", "reference")}
    tables = {"full": None, "without": {}}
    for name, idx in subsets:
        ref_raw = consensus(inp["reference"], idx)
        db_raw  = consensus(inp["drugbank"], idx)
        x, y    = cs.build_anchor_table(ref_raw)

        ref_rank = cs.apply_anchor_table(ref_raw, x, y)
        share = float(at_cutoff(ref_rank).mean())
        if abs(share - 0.01) > SHARE_TOL:
            raise AssertionError(f"anchor self-check failed for '{name}': {share:.4%} of the reference "
                                 f"is at or above rank {DECISION_RANK}, expected 1%")
        rank = cs.apply_anchor_table(db_raw, x, y)

        suffix = "" if name == "full" else f"_without_{name}"
        for dataset, raw_values, rank_values in (("drugbank", db_raw, rank), ("reference", ref_raw, ref_rank)):
            columns[dataset]["raw"][f"consensus_raw{suffix}"]   = raw_values
            columns[dataset]["rank"][f"consensus_rank{suffix}"] = rank_values
        table = {
            "x": x.tolist(), "y": y.tolist(),
            "anchors_raw": cs.anchor_values(ref_raw).tolist(),
            "anchors_se":  cs.anchor_standard_errors(ref_raw, boot_idx).tolist(),
            "reference_share_at_cutoff": share,
            "drugbank_hits_at_cutoff": int(at_cutoff(rank).sum()),
        }
        if name == "full":
            tables["full"] = table
        else:
            tables["without"][name] = table
    return columns, tables


def run(pathogen: str, drugbank_dir: str, reference_dir: str, reports: pd.DataFrame, out_dir: str):
    """One pathogen end to end. Returns (status, message); status is 'ok', 'skip' or 'missing'."""
    n_models = int((reports["pathogen"] == pathogen).sum())
    if n_models < 2:
        return "skip", f"{n_models} retained model(s): a consensus needs at least 2"

    inp, reason = load_inputs(pathogen, drugbank_dir, reference_dir, reports)
    if inp is None:
        return "missing", reason

    models = inp["models"]
    boot_idx = cs.bootstrap_indices(len(inp["reference"]), N_BOOTSTRAP, RANDOM_SEED)
    cols_w,  tables_w  = score_variant(inp, True,  boot_idx)
    cols_uw, tables_uw = score_variant(inp, False, boot_idx)

    folder = os.path.join(out_dir, pathogen)
    os.makedirs(folder, exist_ok=True)
    smiles = {"drugbank": inp["smiles"], "reference": inp["reference_smiles"]}
    for dataset in ("drugbank", "reference"):
        for prefix, cols in ((dataset, cols_w[dataset]), (f"{dataset}_unweighted", cols_uw[dataset])):
            for scale in ("raw", "rank"):
                pd.DataFrame({"smiles": smiles[dataset], **cols[scale]}).round(DECIMALS).to_csv(
                    os.path.join(folder, f"{prefix}_{scale}.csv"), index=False)

    anchors = {
        "pathogen": pathogen,
        "created_by": "scripts/14_consensus_scoring.py",
        "lazyqsar_version": lazyqsar.__version__,
        "reference": {
            "id": identity.REFERENCE_ID, "n": identity.default_n(), "manifest_sha256": manifest_sha256(),
            "rank_file_sha256": cs.file_sha256(inp["reference_file"]),
        },
        "scale": {
            "percentiles": cs.ANCHOR_PERCENTILES, "ranks": cs.ANCHOR_RANKS, "decision_rank": DECISION_RANK,
            "top_of_scale": "straight line from (reference p99.9 -> 0.75) to (raw 1.0 -> rank 1.0); no actives anchor",
        },
        "models": models,
        "weights": {
            m: {**{c: float(v) for c, v in zip(QUALITY_WEIGHT_COLS, inp["w_quality"][i])},
                "decision_cutoff_rank": float(inp["cutoffs"][i])}
            for i, m in enumerate(models)
        },
        "w_weights": [float(v) for v in W_WEIGHTS],
        "weights_fingerprint": cs.weights_fingerprint(models, QUALITY_WEIGHT_COLS, inp["w_quality"], inp["cutoffs"], W_WEIGHTS),
        "weighted": tables_w, "unweighted": tables_uw,
    }
    with open(os.path.join(folder, "anchors.json"), "w") as f:
        json.dump(anchors, f, indent=1)

    full, a_raw, a_se = tables_w["full"], tables_w["full"]["anchors_raw"], tables_w["full"]["anchors_se"]
    n_db = len(inp["smiles"])
    return "ok", (
        f"{len(models)} models | {n_db:,} DrugBank compounds\n"
        f"      reference anchors (raw -> rank): "
        + "  ".join(f"p{q:g}={v:.3f}±{se:.3f}" for q, v, se in zip(cs.ANCHOR_PERCENTILES, a_raw, a_se)) + "\n"
        f"      DrugBank at consensus_rank >= {DECISION_RANK}: {full['drugbank_hits_at_cutoff']:,} "
        f"({100 * full['drugbank_hits_at_cutoff'] / n_db:.2f}%)  "
        f"[reference: {100 * full['reference_share_at_cutoff']:.2f}%]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--pathogen",              default=None)
    parser.add_argument("--input_dir_drugbank",    default=DEFAULT_DRUGBANK_DIR)
    parser.add_argument("--input_dir_reference",   default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output_dir",            default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    reports   = pd.read_csv(REPORTS_PATH)
    pathogens = [args.pathogen] if args.pathogen else list(dict.fromkeys(reports["pathogen"]))

    done, skipped, missing = [], [], []
    for pathogen in pathogens:
        status, message = run(pathogen, args.input_dir_drugbank, args.input_dir_reference, reports, args.output_dir)
        print(f"[{pathogen}] {status.upper() if status != 'ok' else 'ok'}: {message}")
        {"ok": done, "skip": skipped, "missing": missing}[status].append(pathogen)

    print(f"\n{len(done)} pathogen(s) scored; {len(skipped)} skipped (fewer than 2 models); "
          f"{len(missing)} missing inputs -> {args.output_dir}")
    if missing:
        print("Missing inputs for: " + ", ".join(missing) + "\n(steps 12a and 12b must finish for them first)")
        sys.exit(1)


if __name__ == "__main__":
    main()
