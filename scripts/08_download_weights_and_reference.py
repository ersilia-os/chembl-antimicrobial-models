"""
Step 08 — Download the LazyQSAR descriptor weights and the reference library.

Run this ONCE from the login node before submitting the SLURM array job (step 09), with
the project environment:

    envs/camm/bin/python scripts/08_download_weights_and_reference.py
    envs/camm/bin/python scripts/08_download_weights_and_reference.py --path /custom/dir

Two halves, in this order because the second depends on the first:

  1. Descriptor weights (~610 MB): the chemeleon, cddd (encoder + FPSim index + SMILES) and
     CLAMP checkpoints step 09 featurizes with. Fetched by lazyqsar itself rather than from
     a copy of the URLs kept here, so the two cannot drift apart, and so every file is
     checked against a published sha256 on arrival — a truncated or substituted download is
     refused instead of surfacing later as a parse error on a compute node. Already-cached
     files are re-verified, not assumed good.

  2. Reference library (~260 MB): `predict_rank` reports a molecule's position against a
     fixed 50,000-molecule drug-like library, and fitting a model with lazyqsar >= 3.6
     needs that library's descriptor matrices cached locally. Fetching them once here lets
     the step-09 tasks run with LAZYQSAR_REFERENCE_OFFLINE=1, so none of them reaches for
     the network or races the others into the same cache. Four checks, any of which exits
     non-zero:
       a. fetch the manifest, the reference SMILES and one matrix per descriptor
       b. structural check of every cached file (opens, right shape, finite values)
       c. sha256 of every cached file against the manifest
       d. descriptor-drift canary: recompute the manifest's 16 canary molecules with THIS
          install and compare against the published values. lazyqsar does not run this at
          fit time, so an install that cannot reproduce the matrices would otherwise fit
          and rank silently. Neural descriptors whose checkpoint hash is unchanged are
          reported as "skip" by lazyqsar itself (the values are not recomputed).
     The canary in (d) recomputes descriptors with the local install, which is why the
     weights in part 1 have to be in place first.

Re-running is safe: files already cached are not downloaded again, but every check in
part 2 still runs. Both halves land under <path>/.lazyqsar/ — the weights at the top
level, the library in reference/<reference_id>/.

Finally prints the `sbatch` command for step 09, split into small and large datasets.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

DEFAULT_PATH = os.path.join(REPO_ROOT, "output", "08_weights")


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


def download_weights() -> None:
    """Part 1 — the descriptor checkpoints step 09 featurizes with.

    lazyqsar owns the URLs and their checksums (`lazyqsar/utils/checkpoints.py`), and these
    are the same three calls `lazyqsar setup --descriptors` makes. Keeping our own copy of
    the URLs here would mean a checkpoint changing upstream goes unnoticed: the old file
    would keep being fetched, and step 09 would featurize with weights that no longer match
    what lazyqsar expects. Morgan and rdkit need no checkpoints, so nothing covers them.

    Like `download_reference`, call only after the environment redirection in `main`.
    """
    from lazyqsar.utils.checkpoints import checkpoint_dir
    from lazyqsar.utils.setup import download_cddd, download_chemeleon, download_clamp

    print(f"Weights directory: {checkpoint_dir()}")
    download_chemeleon()
    download_cddd()
    download_clamp()
    print("All weights ready.")


def download_reference() -> None:
    """Part 2 — fetch and verify the reference library.

    Call only after the environment redirection in `main`. lazyqsar captures LAZYQSAR_HOME
    at import time in at least one place (`checkpoints.CHECKPOINT_DIR`), so importing it
    here rather than at module level is what keeps the cache where the fit jobs read it.
    """
    from lazyqsar.reference import identity, manifest, store
    from lazyqsar.reference.download import ReferenceDownloadError, download
    from lazyqsar.registry import DESCRIPTOR_TYPES

    descriptors = sorted(DESCRIPTOR_TYPES)
    n = identity.default_n()
    target = identity.reference_dir()
    print(f"\nReference library: {identity.REFERENCE_ID} (tier {n:,})")
    print(f"Cache directory:   {target}")

    # a. Fetch
    files = [identity.smiles_filename(n)] + [
        identity.descriptor_filename(name, n) for name in descriptors
    ]
    try:
        download(files, n=n)
    except ReferenceDownloadError as exc:
        sys.exit(f"\nFetch failed: {exc}")

    # b. Structural check
    problems = store.verify(n)
    if problems:
        sys.exit("\nStructural check failed:\n  " + "\n  ".join(problems))
    print("\n[ok] every cached file is present and well formed")

    # c. Hashes against the manifest
    published = manifest.load(n)
    if published is None:
        sys.exit("\nNo manifest published for this bundle; its files cannot be verified.")
    try:
        for name in files:
            manifest.verify_file(target / name, published)
    except manifest.ReferenceDriftError as exc:
        sys.exit(f"\nHash check failed: {exc}")
    print(f"[ok] {len(files)} files match the manifest sha256")
    print(f"     manifest_sha256 (recorded in every checkpoint): {manifest.manifest_sha256()}")

    # d. Descriptor drift
    try:
        results = manifest.check_environment(descriptors, published)
    except manifest.ReferenceDriftError as exc:
        sys.exit(f"\nDescriptor drift: {exc}")
    print("\nDescriptor-drift canary:")
    for name in descriptors:
        r = results.get(name, {"status": "not checked"})
        extra = f"  ({r['reason']})" if r.get("reason") else ""
        print(f"  {name:<10} {r['status']}{extra}")

    print(
        "\nReference library ready. Run the step-09 fits with "
        "LAZYQSAR_REFERENCE_OFFLINE=1 so no task fetches anything."
    )


def print_sbatch_hint(subset: bool) -> None:
    """Part 3 — the step-09 submission commands, split by dataset size."""
    script_path = os.path.join(ROOT, "09_run_models.sh")
    metadata_path = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")

    if not os.path.exists(metadata_path):
        print(f"\nMetadata not found at {metadata_path} — run step 07 first.")
        return

    import pandas as pd
    df = pd.read_csv(metadata_path)

    if subset:
        keep = {"abaumannii", "saureus", "ecoli", "calbicans"}
        df   = df[df["pathogen"].isin(keep)]

    large = df.index[df["final_compounds"] > 30_000].tolist()
    small = df.index[df["final_compounds"] <= 30_000].tolist()

    print(f"\nSmall jobs (≤30k compounds, {len(small)} datasets):")
    print(f"    sbatch --chdir={REPO_ROOT} --job-name=camm-lq-sm --array={_to_array_spec(small)}%20 --mem=16G {script_path}")
    print(f"\nLarge jobs (>30k compounds, {len(large)} datasets):")
    print(f"    sbatch --chdir={REPO_ROOT} --job-name=camm-lq-lg --array={_to_array_spec(large)}%5 --mem=64G {script_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"Cache root; weights land in <path>/.lazyqsar/ and the reference library in "
             f"<path>/.lazyqsar/reference/ (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Restrict SLURM array indices to abaumannii, saureus, ecoli and calbicans only",
    )
    args = parser.parse_args()

    # Resolved before HOME is reassigned below, or a `~` in --path would expand against
    # the new value rather than the real home directory.
    weights_root = Path(args.path).expanduser().resolve()
    cache_dir    = weights_root / ".lazyqsar"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Both halves are lazyqsar downloads, so the redirection has to come before either of
    # them. Same redirection as 09_run_models.sh / 12a_predict_drugbank.py, so everything
    # lands in the cache the fit jobs read. LAZYQSAR_HOME is set as well because it is the
    # variable lazyqsar actually consults; HOME alone only works through its ~/.lazyqsar
    # default.
    os.environ["HOME"] = str(weights_root)
    os.environ["LAZYQSAR_HOME"] = str(cache_dir)

    # The reference library is downloaded by running the `eosvc` executable found on PATH.
    # Calling the env's python by path (envs/camm/bin/python) does not activate the env, so
    # put its bin/ on PATH here or the fetch reports eosvc as not installed.
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]

    download_weights()
    download_reference()
    print_sbatch_hint(args.subset)


if __name__ == "__main__":
    main()
