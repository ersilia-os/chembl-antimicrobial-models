"""
Step 19 — Apply the refresh package produced by step 18b to a local clone,
then run `ersilia fetch --from_dir` on it as a structural verification.

Prereq: `python scripts/18a_clone_hub_repos.py` has cloned the target eos-id's
        repo (or it already exists at --repo-dir), and
        `python scripts/18b_update_ersilia_model.py --pathogen <p> --repo-dir <r>`
        has been run for the same (pathogen, repo-dir), populating
        output/18_emh_files/{pathogen}/ with the full refresh package.

What this script does:
  1. rsyncs the new checkpoints from output/09_models/{pathogen}/ into
     {repo-dir}/model/checkpoints/models/ (--delete drops removed sub-models).
  2. Copies the 6 generated artifacts from output/18_emh_files/{pathogen}/
     into the right paths under {repo-dir}/.
  3. `ersilia delete {eosXXXX}` (silent — clears any cached install).
  4. `ersilia fetch {eosXXXX} --from_dir {repo-dir}`. Fails loud on non-zero
     exit. This validates metadata.yml schema, runs install.yml end-to-end,
     and executes run.sh once.
Steps 3-4 (and the CONDA_SH requirement) are skipped with --no-fetch, which
stops after syncing checkpoints and copying artifacts.

The user (with Claude) then reviews `cd {repo-dir} && git diff`, commits,
and pushes direct to main on ersilia-os/{eosXXXX}.

Usage:
    python scripts/19_apply_and_fetch.py \\
        --pathogen abaumannii \\
        --repo-dir /path/to/clone/eos21dr

    python scripts/19_apply_and_fetch.py \\
        --pathogen abaumannii \\
        --repo-dir /path/to/clone/eos21dr \\
        --no-fetch
"""

import argparse
import os
import shutil
import subprocess
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import ERSILIA_MODEL_IDS

REPO_ROOT  = os.path.abspath(os.path.join(root, ".."))
PKG_DIR    = os.path.join(REPO_ROOT, "output", "18_emh_files")
MODELS_DIR = os.path.join(REPO_ROOT, "output", "09_models")

CONDA_SH    = os.environ.get("CONDA_SH")
ERSILIA_ENV = "ersilia"

# (src under output/18_emh_files/{pathogen}/, dest under {repo_dir}/)
_FILE_COPIES = [
    ("reports.csv",    "model/checkpoints/reports.csv"),
    ("run_columns.csv", "model/framework/columns/run_columns.csv"),
    ("consensus.py",   "model/framework/code/consensus.py"),
    ("metadata.yml",   "metadata.yml"),
    ("install.yml",    "install.yml"),
    ("run_output.csv", "model/framework/examples/run_output.csv"),
]


def _run_in_ersilia_env(cmd):
    """bash -c 'source conda.sh && conda activate ersilia && <cmd>'."""
    full = f"source {CONDA_SH} && conda activate {ERSILIA_ENV} && {cmd}"
    return subprocess.run(["bash", "-c", full], text=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--pathogen", required=True)
    p.add_argument("--repo-dir", required=True,
                   help="Path to the user's clone of ersilia-os/{eosXXXX}.")
    p.add_argument("--no-fetch", action="store_true",
                   help="Skip `ersilia delete`/`ersilia fetch` validation; "
                        "only sync checkpoints and copy artifacts.")
    args = p.parse_args()

    pathogen = args.pathogen
    repo_dir = os.path.abspath(args.repo_dir)
    total_steps = 2 if args.no_fetch else 4

    if pathogen not in ERSILIA_MODEL_IDS:
        sys.exit(f"Unknown pathogen '{pathogen}'. Known: {sorted(ERSILIA_MODEL_IDS)}")
    eosXXXX = ERSILIA_MODEL_IDS[pathogen]

    if not args.no_fetch:
        if not CONDA_SH:
            sys.exit(
                "FAIL: CONDA_SH environment variable is not set. Export it to your "
                "machine's conda.sh, e.g. export CONDA_SH=~/miniconda3/etc/profile.d/conda.sh"
            )
        if not os.path.exists(CONDA_SH):
            sys.exit(f"FAIL: CONDA_SH does not exist: {CONDA_SH}")

    if not os.path.isdir(repo_dir):
        sys.exit(f"--repo-dir does not exist: {repo_dir}")
    pkg = os.path.join(PKG_DIR, pathogen)
    if not os.path.isdir(pkg):
        sys.exit(
            f"No refresh package at {pkg}.\n"
            f"Run scripts/18b_update_ersilia_model.py --pathogen {pathogen} "
            f"--repo-dir {repo_dir} first."
        )

    print(f"[1/{total_steps}] sync kept-only checkpoints -> {repo_dir}/model/checkpoints/models/")
    # Only sync the sub-models that survived AUROC filtering (those in
    # reports.csv). Without this filter, checkpoints for filtered-out
    # sub-models would bloat the published repo even though main.py never
    # loads them (it reads MODEL_NAMES from run_columns.csv, which has only
    # the kept set).
    import csv as _csv
    with open(os.path.join(pkg, "reports.csv")) as f:
        pkg_rows = list(_csv.DictReader(f))
    kept = sorted({r["model_name"] for r in pkg_rows})
    orig_of = {r["model_name"]: r["original_name"] for r in pkg_rows}
    src_root = os.path.join(MODELS_DIR, pathogen)
    dst_root = os.path.join(repo_dir, "model", "checkpoints", "models")
    os.makedirs(dst_root, exist_ok=True)
    # Drop any dst sub-dir not in `kept` (this covers both filtered-out
    # sub-models and ones removed by retraining).
    dropped = sorted(set(os.listdir(dst_root)) - set(kept))
    for sub in dropped:
        shutil.rmtree(os.path.join(dst_root, sub))
    if dropped:
        print(f"      dropped {len(dropped)}: {dropped}")
    # Mirror each kept sub-model from source. `rsync -a --delete` per-dir
    # gives byte-identical contents and prunes stale files within a sub-dir.
    # orig_of[sub] is reports.csv's original_name -- the on-disk checkpoint dir
    # name under output/09_models/, copied there verbatim by 18b.
    for sub in kept:
        src = os.path.join(src_root, orig_of[sub]) + "/"
        if not os.path.isdir(src_root) or not os.path.isdir(src.rstrip("/")):
            sys.exit(f"FAIL: no source checkpoint dir for sub-model '{sub}' at {src}.")
        dst = os.path.join(dst_root, sub) + "/"
        os.makedirs(dst, exist_ok=True)
        res = subprocess.run(["rsync", "-a", "--delete", src, dst])
        if res.returncode != 0:
            sys.exit(f"FAIL: rsync sub-model '{sub}' exit {res.returncode}.")
    print(f"      {len(kept)} sub-models synced")

    print(f"[2/{total_steps}] copy 6 artifacts from {pkg}/")
    for src_name, dst_rel in _FILE_COPIES:
        src = os.path.join(pkg, src_name)
        dst = os.path.join(repo_dir, dst_rel)
        if not os.path.exists(src):
            sys.exit(f"FAIL: missing {src}.")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"      {src_name} -> {dst_rel}")

    if not args.no_fetch:
        print(f"[3/{total_steps}] ersilia delete {eosXXXX}  (clear any cached install)")
        _run_in_ersilia_env(f"ersilia delete {eosXXXX}")  # exit code intentionally ignored

        print(f"[4/{total_steps}] ersilia fetch {eosXXXX} --from_dir {repo_dir}")
        res = _run_in_ersilia_env(f"ersilia fetch {eosXXXX} --from_dir {repo_dir}")
        if res.returncode != 0:
            sys.exit(f"FAIL: ersilia fetch exit {res.returncode}.")

    print()
    print("=" * 60)
    if args.no_fetch:
        print(f"DONE: checkpoints synced and artifacts copied to {repo_dir}.")
        print(f"ersilia fetch validation was skipped (--no-fetch).")
    else:
        print(f"PASS: {eosXXXX} fetched from {repo_dir}.")
    print(f"Next, review and ship:")
    print(f"  cd {repo_dir}")
    print(f"  git diff")
    print(f"  git add -A && git commit -m 'Refresh checkpoints for {pathogen}'")
    print(f"  git push origin main")


if __name__ == "__main__":
    main()
