"""
Step 20 — Cut the next GitHub release for a pathogen's Hub repo.

Deliberately separate from the push itself (scripts 18b/19): a release is the only
thing that triggers retag-release-docker.yml (republishes the model's Docker image) —
upload-model.yml runs on every push to main but never creates a release. Run this only
after confirming that pathogen's post-push CI has actually succeeded; it refuses to
cut a release otherwise rather than silently skipping the check.

Per pathogen:
  1. Reads the current HEAD of ersilia-os/{eosXXXX}'s main branch and the most recent
     workflow run on main. Exits (not just warns) unless that run's commit matches
     HEAD and it completed successfully.
  2. Reads the latest existing release tag (vN.0.0) and bumps to v{N+1}.0.0 — per
     pathogen, since pathogens sit at different versions (most are at v2, a couple
     already at v3 from an earlier individual refresh).
  3. Creates the release via `gh release create ... --generate-notes`.

Requires the `gh` CLI authenticated against GitHub.

Usage:
    python scripts/20_cut_releases.py --pathogen abaumannii
"""

import argparse
import json
import os
import re
import subprocess
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import ERSILIA_MODEL_IDS, PATHOGENS

_TAG_RE = re.compile(r"^v(\d+)\.0\.0$")


def _run(cmd: list) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"FAIL: {' '.join(cmd)}\n{res.stderr}")
    return res.stdout.strip()


def latest_main_run(eos_id: str) -> dict:
    out = _run([
        "gh", "run", "list", "--repo", f"ersilia-os/{eos_id}", "--branch", "main",
        "--limit", "1", "--json", "headSha,status,conclusion,url",
    ])
    runs = json.loads(out)
    if not runs:
        sys.exit(f"FAIL: no workflow runs found on main for ersilia-os/{eos_id}.")
    return runs[0]


def remote_head_sha(eos_id: str) -> str:
    return _run(["gh", "api", f"repos/ersilia-os/{eos_id}/commits/main", "--jq", ".sha"])


def latest_release_tag(eos_id: str) -> str | None:
    out = _run([
        "gh", "release", "list", "--repo", f"ersilia-os/{eos_id}",
        "--limit", "1", "--json", "tagName", "--jq", ".[0].tagName",
    ])
    return out or None


def next_tag(current: str | None) -> str:
    if current is None:
        return "v1.0.0"
    m = _TAG_RE.match(current)
    if not m:
        sys.exit(f"FAIL: latest release tag {current!r} doesn't match expected 'vN.0.0' pattern.")
    return f"v{int(m.group(1)) + 1}.0.0"


def cut_release(pathogen: str) -> None:
    eos_id = ERSILIA_MODEL_IDS[pathogen]
    print(f"[{pathogen}] ({eos_id})")

    head_sha = remote_head_sha(eos_id)
    run = latest_main_run(eos_id)
    print(f"  main HEAD:        {head_sha}")
    print(f"  latest run:       {run['headSha']}  status={run['status']}  conclusion={run['conclusion']}")
    print(f"  {run['url']}")

    if run["headSha"] != head_sha:
        sys.exit(
            f"FAIL: latest CI run is for commit {run['headSha']}, but main HEAD is "
            f"{head_sha} — CI for the latest push hasn't been picked up yet (or a newer "
            f"commit landed since). Not cutting a release."
        )
    if run["status"] != "completed" or run["conclusion"] != "success":
        sys.exit(
            f"FAIL: latest CI run on main is status={run['status']!r} "
            f"conclusion={run['conclusion']!r} — not a clean success. Not cutting a release."
        )

    current = latest_release_tag(eos_id)
    new_tag = next_tag(current)
    print(f"  release: {current or '(none)'} -> {new_tag}")

    _run([
        "gh", "release", "create", new_tag, "--repo", f"ersilia-os/{eos_id}",
        "--generate-notes",
    ])
    print(f"  Released {new_tag} for {pathogen} ({eos_id}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut the next GitHub release for a pathogen's Hub repo, only if its latest CI run succeeded."
    )
    parser.add_argument("--pathogen", required=True, choices=PATHOGENS, metavar="PATHOGEN")
    args = parser.parse_args()
    cut_release(args.pathogen)


if __name__ == "__main__":
    main()
