"""
Step 07 — Download LazyQSAR descriptor weights.

Run this ONCE from the login node before submitting the SLURM array job (step 08).
Re-running is safe: each file is skipped if it already exists.

Usage:
    python scripts/07_download_weights.py
    python scripts/07_download_weights.py --path /custom/weights/dir
"""

import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

DEFAULT_PATH = os.path.join(REPO_ROOT, "output", "results", "07_weights")

WEIGHTS = {
    "chemeleon_mp.pt":         "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
    "cddd_encoder.onnx":       "https://zenodo.org/records/14811055/files/encoder.onnx?download=1",
    "cddd_encoder_fpsim.h5":   "https://ersilia-models.s3.eu-central-1.amazonaws.com/eos4rw4/model/checkpoints/fpsim2_database_chembl.h5",
    "cddd_encoder_smiles.csv": "https://ersilia-models.s3.eu-central-1.amazonaws.com/eos4rw4/model/checkpoints/fpsim2_database_chembl_smiles.csv",
    "clamp_encoder.onnx":      "https://ersilia-models.s3.eu-central-1.amazonaws.com/eos3l5f/model/checkpoints/clamp_clip/compound_encoder.onnx",
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already exists: {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urlretrieve(url, tmp)
    tmp.replace(dest)


def _to_array_spec(indices: list[int]) -> str:
    if not indices:
        return ""
    indices = sorted(indices)
    parts = []
    start = end = indices[0]
    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            parts.append(str(start) if start == end else f"{start}-{end}")
            start = end = i
    parts.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"Directory for weight caching; weights land in <path>/.lazyqsar/ (default: {DEFAULT_PATH})",
    )
    args = parser.parse_args()

    cache_dir = Path(args.path).expanduser().resolve() / ".lazyqsar"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Weights directory: {cache_dir}")

    for filename, url in WEIGHTS.items():
        _download(url, cache_dir / filename)

    print("All weights ready.")

    script_path = os.path.join(ROOT, "08_run_models.sh")
    metadata_path = os.path.join(REPO_ROOT, "output", "results", "06_datasets_metadata.csv")

    if not os.path.exists(metadata_path):
        print(f"\nMetadata not found at {metadata_path} — run step 06 first.")
        return

    import pandas as pd
    df = pd.read_csv(metadata_path)
    large = df.index[df["final_compounds"] > 30_000].tolist()
    small = df.index[df["final_compounds"] <= 30_000].tolist()

    print(f"\nSmall jobs (≤30k compounds, {len(small)} datasets):")
    print(f"    sbatch --chdir={REPO_ROOT} --job-name=camm-lq-sm --array={_to_array_spec(small)}%20 --mem=16G {script_path}")
    print(f"\nLarge jobs (>30k compounds, {len(large)} datasets):")
    print(f"    sbatch --chdir={REPO_ROOT} --job-name=camm-lq-lg --array={_to_array_spec(large)}%5 --mem=64G {script_path}")


if __name__ == "__main__":
    main()
