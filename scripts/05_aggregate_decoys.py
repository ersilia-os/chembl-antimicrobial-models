"""
Step 05 — Aggregate eos3e6s decoy results and optionally clean up intermediates.

Streams all per-split CSVs from output/results/04_decoys/ into a single
output/results/05_eos3e6s.csv.gz without loading everything into memory.
Missing split files (failed SLURM jobs) are warned about and skipped.

With --cleanup, removes directories that are redundant after aggregation:
  - output/results/03_positives_splits/   (step-04 inputs, regenerable)
  - output/results/04_decoys/             (raw per-split outputs, now aggregated)
  - output/results/03_ersilia_apptainer/  (git clone, already installed)

Kept even with --cleanup (expensive to recreate):
  - output/results/03_eos3e6s/            (446 MB SIF image)
  - output/results/03_conda_camm/         (conda env, needed for step-04 re-runs)
  - output/results/04_logs/              (SLURM logs, useful for debugging)

Usage:
    python scripts/05_aggregate_decoys.py
    python scripts/05_aggregate_decoys.py --cleanup
    python scripts/05_aggregate_decoys.py --decoys_dir path/to/dir --output path/to/out.csv.gz
"""

import argparse
import gzip
import glob
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
MODEL = "eos3e6s"

CLEANUP_DIRS = [
    os.path.join(REPO_ROOT, "output", "results", "03_positives_splits"),
    os.path.join(REPO_ROOT, "output", "results", "04_decoys"),
    os.path.join(REPO_ROOT, "output", "results", "03_ersilia_apptainer"),
]


def aggregate(decoys_dir: str, output_path: str) -> None:
    pattern = os.path.join(decoys_dir, f"{MODEL}_*.csv")
    split_files = sorted(glob.glob(pattern))

    if not split_files:
        print(f"No files matching {pattern}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_rows = 0
    with gzip.open(output_path, "wt", newline="") as gz:
        for i, path in enumerate(split_files):
            if not os.path.exists(path):
                print(f"[WARN] Missing split: {path}")
                continue
            with open(path, newline="") as f:
                header = next(f)
                if i == 0:
                    gz.write(header)
                for line in f:
                    gz.write(line)
                    n_rows += 1

    print(f"Aggregated {len(split_files)} splits ({n_rows:,} rows) → {output_path}")


def cleanup() -> None:
    for path in CLEANUP_DIRS:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Removed {path}")
        else:
            print(f"Already absent: {path}")


def main(decoys_dir: str, output_path: str, do_cleanup: bool) -> None:
    aggregate(decoys_dir, output_path)
    if do_cleanup:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate eos3e6s per-split outputs into a single csv.gz."
    )
    parser.add_argument(
        "--decoys_dir",
        default=os.path.join(REPO_ROOT, "output", "results", "04_decoys"),
        help="Directory containing per-split CSVs (default: output/results/04_decoys).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "output", "results", f"05_{MODEL}.csv.gz"),
        help=f"Output path (default: output/results/05_{MODEL}.csv.gz).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove intermediate directories no longer needed after aggregation.",
    )
    args = parser.parse_args()
    main(args.decoys_dir, args.output, args.cleanup)
