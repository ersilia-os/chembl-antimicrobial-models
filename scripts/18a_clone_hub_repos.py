"""
Step 18a — Clone every pathogen's Ersilia Hub repo for the refresh round.

Clones `ersilia-os/{eosXXXX}` (from ERSILIA_MODEL_IDS in src/default.py) into
chembl-models-tmp/{eosXXXX}/, one directory per pathogen — the --repo-dir both
18b_update_ersilia_model.py and 19_apply_and_fetch.py expect. Skips any clone
that already exists, so it's safe to re-run (e.g. after adding a new pathogen).

Requires the `gh` CLI authenticated against GitHub.

Usage:
    python scripts/18a_clone_hub_repos.py
    python scripts/18a_clone_hub_repos.py --pathogen abaumannii efaecium
"""

import argparse
import os
import subprocess
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import ERSILIA_MODEL_IDS, PATHOGENS

REPO_ROOT  = os.path.abspath(os.path.join(root, ".."))
CLONE_ROOT = os.path.join(REPO_ROOT, "..", "chembl-models-tmp")
os.makedirs(CLONE_ROOT, exist_ok=True)


def clone_one(pathogen: str, eos_id: str) -> bool:
    target = os.path.join(CLONE_ROOT, eos_id)
    if os.path.isdir(target):
        print(f"  [SKIP] {pathogen} ({eos_id}): already cloned at {target}")
        return True
    print(f"  Cloning {pathogen} ({eos_id})...")
    res = subprocess.run(
        ["gh", "repo", "clone", f"ersilia-os/{eos_id}", target],
        cwd=CLONE_ROOT,
    )
    if res.returncode != 0:
        print(f"  [WARN] {pathogen} ({eos_id}): clone failed (exit {res.returncode})")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone every pathogen's Ersilia Hub repo.")
    parser.add_argument(
        "--pathogen", nargs="+", choices=PATHOGENS, default=None, metavar="PATHOGEN",
        help="Restrict to specific pathogen(s). Defaults to all pathogens.",
    )
    args = parser.parse_args()

    targets = args.pathogen if args.pathogen else PATHOGENS
    ok = 0
    for pathogen in targets:
        eos_id = ERSILIA_MODEL_IDS[pathogen]
        if clone_one(pathogen, eos_id):
            ok += 1
    print(f"\n{ok}/{len(targets)} repos ready under {CLONE_ROOT}")


if __name__ == "__main__":
    main()
