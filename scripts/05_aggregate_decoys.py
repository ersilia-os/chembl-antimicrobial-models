"""
Step 05 — Aggregate eos3e6s decoy results and optionally clean up intermediates.

Streams all per-split CSVs from output/results/04_decoys/ into a single
output/results/05_eos3e6s.csv without loading everything into memory.

With --cleanup, removes directories that are redundant after aggregation.
The expected split count is read from output/results/02_selected_positives.csv.
Cleanup is skipped with a warning if fewer files than expected were aggregated.

Removed with --cleanup:
  - output/results/03_positives_splits/   (step-04 inputs, regenerable)
  - output/results/04_decoys/             (raw per-split outputs, now aggregated)
  - output/results/04_logs/              (SLURM job logs)

Kept even with --cleanup (expensive to recreate):
  - output/results/03_eos3e6s_v1.sif     (Apptainer SIF image)

Usage:
    python scripts/05_aggregate_decoys.py
    python scripts/05_aggregate_decoys.py --cleanup
    python scripts/05_aggregate_decoys.py --decoys_dir path/to/dir --output path/to/out.csv
"""

import argparse
import glob
import os
import shutil

import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
MODEL = "eos3e6s"

POSITIVES_PATH = os.path.join(REPO_ROOT, "output", "results", "02_selected_positives.csv")
CLEANUP_DIRS = [
    os.path.join(REPO_ROOT, "output", "results", "03_positives_splits"),
    os.path.join(REPO_ROOT, "output", "results", "04_decoys"),
    os.path.join(REPO_ROOT, "output", "results", "04_logs"),
]


def aggregate(decoys_dir: str, output_path: str) -> int:
    pattern = os.path.join(decoys_dir, f"{MODEL}_*.csv")
    split_files = sorted(glob.glob(pattern))

    if not split_files:
        print(f"No files matching {pattern}")
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_rows = 0
    with open(output_path, "w", newline="") as out:
        for i, path in enumerate(tqdm(split_files, desc="Aggregating splits", unit="split")):
            with open(path, newline="") as f:
                header = next(f)
                if i == 0:
                    out.write(header)
                for line in f:
                    out.write(line)
                    n_rows += 1

    print(f"Aggregated {len(split_files)} splits ({n_rows:,} rows) → {output_path}")
    return len(split_files)


def resolve_expected() -> int | None:
    if os.path.isfile(POSITIVES_PATH):
        return int(pd.read_csv(POSITIVES_PATH)["split"].max()) + 1
    return None


def cleanup() -> None:
    for path in CLEANUP_DIRS:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Removed {path}")
        else:
            print(f"Already absent: {path}")


def main(decoys_dir: str, output_path: str, do_cleanup: bool) -> None:
    n_found = aggregate(decoys_dir, output_path)
    n_expected = resolve_expected()

    if n_expected is None:
        print("[WARN] Could not read expected split count from 02_selected_positives.csv.")
        return

    if n_found < n_expected:
        print(f"[WARN] Cleanup not possible: only {n_found} of {n_expected} expected splits were aggregated.")
    elif not do_cleanup:
        print(f"All {n_found} splits aggregated successfully. Run with --cleanup to remove intermediate files.")
    else:
        cleanup()
        print("All clean!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate eos3e6s per-split outputs into a single csv."
    )
    parser.add_argument(
        "--decoys_dir",
        default=os.path.join(REPO_ROOT, "output", "results", "04_decoys"),
        help="Directory containing per-split CSVs (default: output/results/04_decoys).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "output", "results", f"05_{MODEL}_v1.csv"),
        help=f"Output path (default: output/results/05_{MODEL}_v1.csv).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove intermediate directories no longer needed after aggregation.",
    )
    args = parser.parse_args()
    main(args.decoys_dir, args.output, args.cleanup)
