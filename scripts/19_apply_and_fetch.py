"""
Step 19 — Apply the refresh package produced by step 18 to a local clone,
then run `ersilia fetch --from_dir` on it as a structural verification.

Prereq: `python scripts/18_update_ersilia_model.py --pathogen <p> --repo-dir <r>`
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

The user (with Claude) then reviews `cd {repo-dir} && git diff`, commits,
and pushes direct to main on ersilia-os/{eosXXXX}.

Usage:
    python scripts/19_apply_and_fetch.py \\
        --pathogen abaumannii \\
        --repo-dir /path/to/clone/eos21dr
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

CONDA_SH    = os.environ.get(
    "CONDA_SH",
    os.path.expanduser("~/programs/miniconda3/etc/profile.d/conda.sh"),
)
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
    args = p.parse_args()

    pathogen = args.pathogen
    repo_dir = os.path.abspath(args.repo_dir)

    if pathogen not in ERSILIA_MODEL_IDS:
        sys.exit(f"Unknown pathogen '{pathogen}'. Known: {sorted(ERSILIA_MODEL_IDS)}")
    eosXXXX = ERSILIA_MODEL_IDS[pathogen]

    if not os.path.isdir(repo_dir):
        sys.exit(f"--repo-dir does not exist: {repo_dir}")
    pkg = os.path.join(PKG_DIR, pathogen)
    if not os.path.isdir(pkg):
        sys.exit(
            f"No refresh package at {pkg}.\n"
            f"Run scripts/18_update_ersilia_model.py --pathogen {pathogen} "
            f"--repo-dir {repo_dir} first."
        )

    print(f"[1/4] rsync checkpoints -> {repo_dir}/model/checkpoints/models/")
    src_models = os.path.join(MODELS_DIR, pathogen) + "/"
    dst_models = os.path.join(repo_dir, "model", "checkpoints", "models") + "/"
    os.makedirs(dst_models, exist_ok=True)
    res = subprocess.run(["rsync", "-a", "--delete", src_models, dst_models])
    if res.returncode != 0:
        sys.exit(f"FAIL: rsync exit {res.returncode}.")

    print(f"[2/4] copy 6 artifacts from {pkg}/")
    for src_name, dst_rel in _FILE_COPIES:
        src = os.path.join(pkg, src_name)
        dst = os.path.join(repo_dir, dst_rel)
        if not os.path.exists(src):
            sys.exit(f"FAIL: missing {src}.")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"      {src_name} -> {dst_rel}")

    print(f"[3/4] ersilia delete {eosXXXX}  (clear any cached install)")
    _run_in_ersilia_env(f"ersilia delete {eosXXXX}")  # exit code intentionally ignored

    print(f"[4/4] ersilia fetch {eosXXXX} --from_dir {repo_dir}")
    res = _run_in_ersilia_env(f"ersilia fetch {eosXXXX} --from_dir {repo_dir}")
    if res.returncode != 0:
        sys.exit(f"FAIL: ersilia fetch exit {res.returncode}.")

    print()
    print("=" * 60)
    print(f"PASS: {eosXXXX} fetched from {repo_dir}.")
    print(f"Next, review and ship:")
    print(f"  cd {repo_dir}")
    print(f"  git diff")
    print(f"  git add -A && git commit -m 'Refresh checkpoints for {pathogen}'")
    print(f"  git push origin main")


if __name__ == "__main__":
    main()
