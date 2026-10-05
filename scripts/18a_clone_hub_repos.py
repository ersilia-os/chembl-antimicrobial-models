"""
Step 18a — Clone or update the Ersilia Hub repo of each pathogen, ready for a refresh round.

The repo of a pathogen is `ersilia-os/{eosXXXX}` (ERSILIA_MODEL_IDS in src/default.py). Its local copy
lives at HUB_CLONES_DIR/{eosXXXX} (src/default.py: chembl-models-tmp, next to this repo), or at
--repo-dir, the same argument 18b and 19 take. For each one:

  missing                                    git clone
  exists, clean, on main, nothing unpushed   git fetch, then a fast-forward only (never a merge commit)
  exists but not a git repo, not on main,    left alone and reported: a refresh must not pull over
  with uncommitted changes, or ahead of      someone's work
  origin/main
  any git command fails                      reported with git's own message

The Hub repos are also updated by CI, so a clone from weeks ago is behind upstream; the refresh
(18b, then 19) must start from the current main. The exit code is 1 if any repo was left alone or
failed, so that cannot pass unnoticed. At the end the commit each repo is at is printed: that is
the Hub commit the refresh starts from.

Needs git and network access to github.com (public repos, no login for reading). Pushing the
refresh is a manual step after 19, with whatever credentials your git has.

Usage:
    python scripts/18a_clone_hub_repos.py
    python scripts/18a_clone_hub_repos.py --pathogen abaumannii kpneumoniae calbicans
    python scripts/18a_clone_hub_repos.py --pathogen abaumannii --repo-dir /path/to/clone/eos21dr
"""

import argparse
import os
import subprocess
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import ERSILIA_MODEL_IDS, HUB_BRANCH, HUB_CLONES_DIR, HUB_URL, PATHOGENS

os.makedirs(HUB_CLONES_DIR, exist_ok=True)

GIT_TIMEOUT = 300           # seconds per git command (a clone carries the model checkpoints)
OK_STATUSES = ("cloned", "updated", "up-to-date")


def git(args: list, cwd: str = None) -> subprocess.CompletedProcess:
    """Run git; a non-zero exit is returned, not raised. A timeout counts as a failure."""
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=GIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", f"timed out after {GIT_TIMEOUT} s")


def _why(res: subprocess.CompletedProcess) -> str:
    """git's own error: its first 'fatal:' / 'error:' line, else the last line it printed."""
    lines = [l.strip() for l in (res.stderr or res.stdout or "").splitlines() if l.strip()]
    errors = [l for l in lines if l.startswith(("fatal:", "error:"))]
    return errors[0] if errors else (lines[-1] if lines else f"exit {res.returncode}")


def head_of(repo_dir: str) -> str:
    res = git(["log", "-1", "--format=%h %ad %s", "--date=short"], repo_dir)
    return res.stdout.strip() if res.returncode == 0 else "(unknown)"


def update_one(repo_dir: str, url: str) -> tuple:
    """Bring one local copy up to date. Returns (status, message)."""
    if not os.path.exists(repo_dir):
        os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
        res = git(["clone", url, repo_dir])
        if res.returncode != 0:
            return "failed", f"clone failed: {_why(res)}"
        return "cloned", f"cloned {url}"

    # The folder must be the root of its own repository: a plain folder that happens to sit inside
    # another repo would otherwise be judged by that repo's state.
    top = git(["rev-parse", "--show-toplevel"], repo_dir)
    if top.returncode != 0 or os.path.realpath(top.stdout.strip()) != os.path.realpath(repo_dir):
        return "skipped", f"{repo_dir} exists but is not the root of a git repository"

    branch = git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir).stdout.strip()
    if branch != HUB_BRANCH:
        return "skipped", f"on branch '{branch}', not '{HUB_BRANCH}'"

    changes = git(["status", "--porcelain"], repo_dir).stdout.splitlines()
    if changes:
        return "skipped", f"{len(changes)} uncommitted change(s), e.g. '{changes[0].strip()}'"

    res = git(["fetch", "origin", HUB_BRANCH], repo_dir)
    if res.returncode != 0:
        return "failed", f"fetch failed: {_why(res)}"

    ahead = git(["rev-list", "--count", f"origin/{HUB_BRANCH}..HEAD"], repo_dir)
    behind = git(["rev-list", "--count", f"HEAD..origin/{HUB_BRANCH}"], repo_dir)
    if ahead.returncode != 0 or behind.returncode != 0:
        return "failed", f"cannot compare with origin/{HUB_BRANCH}: {_why(ahead if ahead.returncode else behind)}"
    n_ahead, n_behind = int(ahead.stdout), int(behind.stdout)
    if n_ahead:
        return "skipped", f"{n_ahead} unpushed commit(s) ahead of origin/{HUB_BRANCH}"
    if n_behind == 0:
        return "up-to-date", f"already at origin/{HUB_BRANCH}"

    res = git(["merge", "--ff-only", f"origin/{HUB_BRANCH}"], repo_dir)
    if res.returncode != 0:
        return "failed", f"fast-forward failed: {_why(res)}"
    return "updated", f"fast-forwarded {n_behind} commit(s)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--pathogen", nargs="+", choices=PATHOGENS, default=None, metavar="PATHOGEN",
        help="Restrict to specific pathogen(s). Defaults to all pathogens.",
    )
    parser.add_argument(
        "--repo-dir", default=None,
        help="Path of the local copy, instead of HUB_CLONES_DIR/{eosXXXX}. Same argument as 18b and 19; "
             "it is one repo, so it needs exactly one --pathogen.",
    )
    parser.add_argument(
        "--url-template", default=HUB_URL,
        help="Remote URL with an {eos_id} placeholder. Only changed to test against a local repo.",
    )
    args = parser.parse_args()

    targets = args.pathogen if args.pathogen else PATHOGENS
    if args.repo_dir and len(targets) != 1:
        parser.error("--repo-dir is the path of one repo: give exactly one --pathogen with it.")

    results = []
    for pathogen in targets:
        eos_id = ERSILIA_MODEL_IDS[pathogen]
        repo_dir = os.path.abspath(args.repo_dir) if args.repo_dir else os.path.join(HUB_CLONES_DIR, eos_id)
        status, message = update_one(repo_dir, args.url_template.format(eos_id=eos_id))
        print(f"  [{status.upper()}] {pathogen} ({eos_id}): {message}")
        results.append((pathogen, eos_id, repo_dir, status))

    print("\nHub commit each refresh starts from:")
    for pathogen, eos_id, repo_dir, status in results:
        if status in OK_STATUSES:
            print(f"  {pathogen:14s} {eos_id}  {head_of(repo_dir)}")

    problems = [f"{pathogen} ({status})" for pathogen, _, _, status in results if status not in OK_STATUSES]
    print(f"\n{len(results) - len(problems)}/{len(results)} repos ready")
    if problems:
        print("Needs attention: " + ", ".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
