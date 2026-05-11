"""
Step 13 — Consensus scoring of DrugBank compounds per pathogen.

Reads per-pathogen rank matrices from output/results/12_drugbank/{pathogen}.csv
and computes a weighted consensus score per compound using 8 weights:
  W1–W7: model quality weights from 10_reports.csv
  W8:    piecewise linear function of rank vs decision_cutoff_rank (0→0, cutoff→0.5, 1→1)

weight[i,m] = mean(W1..W7, W8[i,m])
score[i]    = mean_m(rank[i,m] * weight[i,m])

Output: output/results/13_consensus/{pathogen}.csv — smiles, consensus_score, model columns.

Usage:
    python scripts/13_consensus_scoring.py
    python scripts/13_consensus_scoring.py --pathogen ecoli
"""

import argparse
import os

import numpy as np
import pandas as pd

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

DEFAULT_IN_DIR  = os.path.join(REPO_ROOT, "output", "results", "12_drugbank")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "output", "results", "13_consensus")
REPORTS_PATH    = os.path.join(REPO_ROOT, "output", "results", "10_reports.csv")
W_COLS          = ["w1", "w2", "w3", "w4", "w5", "w6", "w7"]


def _score(ranks: np.ndarray, w1_7: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    c = cutoffs[np.newaxis, :]
    nan = np.isnan(c)
    c   = np.where(nan, 0.5, c)
    w8  = np.where(ranks <= c,
                   0.5 * ranks / np.clip(c, 1e-9, 1),
                   0.5 + 0.5 * (ranks - c) / np.clip(1 - c, 1e-9, 1))
    w8  = np.where(nan, ranks, w8)
    w   = (w1_7[np.newaxis, :] + w8) / 8.0
    return (ranks * w).mean(axis=1)


def run(pathogen: str, in_dir: str, reports_df: pd.DataFrame, out_path: str) -> None:
    src = os.path.join(in_dir, f"{pathogen}.csv")
    if not os.path.isfile(src):
        print(f"  [SKIP] {pathogen}: {src} not found")
        return

    df   = pd.read_csv(src)
    rows = reports_df[reports_df["pathogen"] == pathogen].set_index("model_name")
    cols = [c for c in df.columns if c != "smiles" and c in rows.index]

    if not cols:
        print(f"  [SKIP] {pathogen}: no models overlap with 10_reports.csv")
        return

    ranks   = df[cols].fillna(0.0).values
    w1_7    = np.array([rows.loc[m, W_COLS].sum() for m in cols])
    cutoffs = np.array([rows.loc[m, "decision_cutoff_rank"] for m in cols], dtype=float)
    scores  = _score(ranks, w1_7, cutoffs)

    out = pd.DataFrame({"smiles": df["smiles"], "consensus_score": scores.round(4)})
    for c in cols:
        out[c] = df[c].values
    out = out.sort_values("consensus_score", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"  [{pathogen}] {len(cols)} models -> {len(out)} compounds -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathogen",   default=None)
    parser.add_argument("--input_dir",  default=DEFAULT_IN_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    reports_df = pd.read_csv(REPORTS_PATH)
    pathogens  = [args.pathogen] if args.pathogen else list(dict.fromkeys(reports_df["pathogen"]))

    for pathogen in pathogens:
        out_path = args.output or os.path.join(args.output_dir, f"{pathogen}.csv")
        run(pathogen, args.input_dir, reports_df, out_path)


if __name__ == "__main__":
    main()
