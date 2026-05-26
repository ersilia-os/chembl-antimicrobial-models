"""
Step 06 — Aggregate eos3e6s decoy results and optionally clean up intermediates.

Streams all per-split CSVs from output/05_decoys/ into a single
output/06_decoys/06_eos3e6s_v1.csv without loading everything into memory.

Also handles the local-run case where 05_run_decoys_ersilia.sh produced a single
eos3e6s_all.csv instead of per-split files.

With --cleanup, removes directories that are redundant after aggregation.
For HPC runs, cleanup is skipped with a warning if fewer files than expected were aggregated
(expected count is read from output/03_select_positives/03_selected_positives.csv).

Removed with --cleanup:
  - output/04_positives_splits/   (step-04 inputs, regenerable)
  - output/05_decoys/             (raw per-split outputs, now aggregated)
  - output/05_logs/              (SLURM job logs)

Kept even with --cleanup (expensive to recreate):
  - output/04_decoys_sif_image/   (Apptainer SIF image)

Usage:
    python scripts/06_aggregate_decoys.py
    python scripts/06_aggregate_decoys.py --cleanup
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

POSITIVES_PATH = os.path.join(REPO_ROOT, "output", "03_select_positives", "03_selected_positives.csv")
DECOYS_DIR = os.path.join(REPO_ROOT, "output", "05_decoys")
OUTPUT_PATH = os.path.join(REPO_ROOT, "output", "06_decoys", f"06_{MODEL}_v1.csv")
CLEANUP_DIRS = [
    os.path.join(REPO_ROOT, "output", "04_positives_splits"),
    os.path.join(REPO_ROOT, "output", "05_decoys"),
    os.path.join(REPO_ROOT, "output", "05_logs"),
]


def aggregate(decoys_dir: str, output_path: str) -> tuple[int, bool]:
    local_file = os.path.join(decoys_dir, f"{MODEL}_all.csv")
    if os.path.isfile(local_file):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(local_file, output_path)
        print(f"Copied local run result → {output_path}")
        return 1, True

    pattern = os.path.join(decoys_dir, f"{MODEL}_*.csv")
    split_files = sorted(glob.glob(pattern))

    if not split_files:
        print(f"No files matching {pattern}")
        return 0, False

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
    return len(split_files), False


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


def main(do_cleanup: bool) -> None:
    n_found, local_run = aggregate(DECOYS_DIR, OUTPUT_PATH)
    if n_found == 0:
        return

    if local_run:
        if do_cleanup:
            cleanup()
            print("All clean!")
        return

    n_expected = resolve_expected()
    if n_expected is None:
        print("[WARN] Could not read expected split count from 03_selected_positives.csv.")
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
        "--cleanup",
        action="store_true",
        help="Remove intermediate directories no longer needed after aggregation.",
    )
    args = parser.parse_args()
    main(do_cleanup=args.cleanup)
