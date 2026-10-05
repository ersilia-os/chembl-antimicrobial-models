"""
Step 18b — Refresh an already-incorporated Ersilia Hub model with newly trained checkpoints.

Prereq: `python scripts/18a_clone_hub_repos.py` has cloned or updated the pathogen's Hub repo
        (default location HUB_CLONES_DIR/{eosXXXX}, or --repo-dir), and step 14 has been run for
        the pathogen (it holds the calibration of the consensus).

The published model's repo is `ersilia-os/{eosXXXX}`. This script produces a refresh package under
`output/18_emh_files/{pathogen}/`, runs the model once in a staging copy to produce and check its
example output, and writes the copy instructions (step 19 applies them). Nothing is written into the
clone and nothing is pushed.

What the package ships for the consensus: the weighted mean of the sub-models' ranks, placed on the
rank scale with the anchor table of step 14 (the reference-library calibration of that pathogen,
baked into consensus.py as 6 + 6 numbers). The consensus threshold is the decision rank 0.65, the
same as every sub-model. See src/hub_consensus.py (the shipped template) and src/consensus.py (the
same formulas, used by step 14).

Safeguards, each of which stops the script:
  - the anchors of step 14 must have been built from exactly the kept models and the weights and
    cutoffs now in 10_reports.csv (weights fingerprint), otherwise "re-run step 14";
  - 10_reports.csv must hold no model below MIN_AUROC (10a keeps only retained models);
  - the model run in staging must use the pinned lazyqsar version;
  - consensus_score of the staged output must equal src/consensus.py on the printed sub-model ranks;
    and on a DrugBank sample (the 15 top of the step-14 consensus + 10 seeded random molecules) the
    staged consensus must equal that of step 14 within 0.01. The staged sub-model ranks are compared
    with step 12b too: a cell beyond 0.01 is FLAGGED in DIFF_SUMMARY.txt for manual review but does not
    stop the script (the CPU family alone moves a rank by up to ~0.014 in rare cells).
DIFF_SUMMARY.txt lists any change of a public column name or position as BREAKING: they are the
model's API.

Inputs:
  output/10_reports/10_reports.csv            — per-sub-model metrics + weights
  output/07_datasets/07_datasets_metadata.csv — assay context for descriptions
  output/09_models/{pathogen}/                — new sub-model checkpoints
  output/14_consensus/{pathogen}/anchors.json — anchor table of the weighted consensus + fingerprint
  output/12_drugbank/rank/{pathogen}.csv, output/14_consensus/{pathogen}/drugbank_rank.csv
                                              — compared with the staged model on a DrugBank sample
  {repo-dir}/metadata.yml                     — current published metadata, patched in place
  {repo-dir}/model/framework/{code/main.py,run.sh,examples/run_input.csv} — seed the staging dir

Outputs (all under output/18_emh_files/{pathogen}/):
  reports.csv, run_columns.csv, consensus.py, metadata.yml, install.yml, run_output.csv,
  DIFF_SUMMARY.txt, COPY_INSTRUCTIONS.txt

The model runs in the conda env HUB_RUNTIME_ENV (src/default.py), which must have the pinned
lazyqsar; conda.sh is taken from $CONDA_SH, else from `conda` on the PATH.

Usage:
    python scripts/18b_update_ersilia_model.py --pathogen abaumannii
    python scripts/18b_update_ersilia_model.py --pathogen abaumannii --repo-dir /path/to/clone/eos21dr
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
os.environ.setdefault("LAZYQSAR_REFERENCE_OFFLINE", "1")   # src/consensus.py imports lazyqsar; nothing is downloaded

import consensus as cs  # noqa: E402  (src/consensus.py, the formulas of step 14)
from default import (  # noqa: E402
    ERSILIA_MODEL_IDS, HUB_CLONES_DIR, HUB_RUNTIME_ENV, MIN_AUROC, QUALITY_WEIGHT_COLS, RANDOM_SEED,
)

REPO_ROOT        = os.path.abspath(os.path.join(root, ".."))
REPORTS_PATH     = os.path.join(REPO_ROOT, "output", "10_reports", "10_reports.csv")
METADATA_PATH    = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")
MODELS_DIR       = os.path.join(REPO_ROOT, "output", "09_models")
CONSENSUS_DIR    = os.path.join(REPO_ROOT, "output", "14_consensus")
DRUGBANK_RANKS   = os.path.join(REPO_ROOT, "output", "12_drugbank", "rank")
OUTPUT_DIR       = os.path.join(REPO_ROOT, "output", "18_emh_files")
TEMPLATE_PATH    = os.path.join(REPO_ROOT, "src", "hub_consensus.py")
TMP_DIR          = os.path.join(REPO_ROOT, "tmp")

_DROP_COLS    = ["predict_rank_actives", "predict_rank_inactives", "member_assay_ids"]
_DESCRIPTORS  = ("chemeleon", "clamp", "cddd")

# Pinned versions for the refreshed install.yml. ersilia-pack-utils matches what was shipped at
# initial incorporation (02_init_pathogen.py); lazyqsar matches environment.yml and the version the
# checkpoints were trained and the rank calibrations built with (3.6.0 and the patched 3.6.1 code).
_ERSILIA_PACK_UTILS_VERSION = "0.1.5"
_LAZYQSAR_VERSION           = "3.6.1"

# The staged model's consensus must equal src/consensus.py applied to the sub-model ranks it printed
# (rounded to 4 decimals, hence 3e-4: the anchor table has slopes up to ~2). Against steps 12b/14 the
# ranks come from another run, on another CPU: the consensus must agree within _PARITY_TOL_PIPELINE
# or the script stops. A sub-model rank beyond it is only FLAGGED for manual review: the CPU family
# alone moves a rank by up to ~0.014 in rare cells (calbicans DR_0008: 0.8058 on Intel nodes, 0.7918 on
# AMD and on dante, consensus effect 5e-4), so a flag needs a person's look, not a stop.
_PARITY_TOL_SELF = 3e-4
_PARITY_TOL_PIPELINE = 1e-2
# DrugBank molecules the staged model is also run on, to compare it with steps 12b and 14: the top of
# the step-14 consensus (where the calibration matters) and a seeded random draw.
_PARITY_N_TOP = 15
_PARITY_N_RANDOM = 10


def render_install_yml(descriptors_needed):
    only = ",".join(descriptors_needed)
    return (
        f'python: "3.12"\n'
        f"commands:\n"
        f'    - ["pip", "ersilia-pack-utils", "{_ERSILIA_PACK_UTILS_VERSION}"]\n'
        f'    - ["pip", "lazyqsar[all]", "{_LAZYQSAR_VERSION}"]\n'
        f'    - "lazyqsar setup --descriptors --only {only}"\n'
    )


# ---------------------------------------------------------------------------
# Filter + sort
# ---------------------------------------------------------------------------

def _run_columns_rank(row) -> int:
    """Public-facing sub-model order for run_columns.csv / consensus.py's output
    columns: SP before DR (dominant — every SP entry, pool or catch-all, ranks
    ahead of every DR entry), pools before catch-alls within each, PubChem merged
    before single. Deliberately separate from 10a's `_type_rank` (DR before SP,
    used for internal training order) — the two audiences are allowed to order
    differently; nothing downstream of 10_reports.csv looks up sub-models by
    position, only by name, so this reorder is safe to scope to this script alone.
    """
    if row["source"] == "pubchem":
        return 4 if bool(row.get("is_merged", False)) else 5
    cat = 0 if row["label"] == "SP" else 1                      # SP before DR — dominant
    tier = 0 if row.get("assay_type", "") == "pool" else 1      # pools before catch-alls — secondary
    return cat * 2 + tier


def filter_and_sort(reports_df, meta_df, pathogen):
    """The pathogen's rows of 10_reports.csv, ordered per _run_columns_rank (compound count
    descending within each group). 10a writes only retained models to that file, so a model below
    MIN_AUROC means the file is not what 10a wrote: stop, do not drop it silently."""
    df = reports_df[reports_df["pathogen"] == pathogen].copy()
    if df.empty:
        sys.exit(f"No rows in 10_reports.csv for pathogen '{pathogen}'.")

    meta = meta_df[["pathogen", "name", "source", "label", "assay_type", "is_merged", "member_assay_ids"]]
    df = df.merge(meta, on=["pathogen", "name"], how="left", validate="one_to_one")
    missing = df[df["source"].isna()]
    if not missing.empty:
        sys.exit(
            f"Sub-models missing source in 07_datasets_metadata.csv: "
            f"{missing['model_name'].tolist()}"
        )

    below = df[df["auroc_mean"] < MIN_AUROC]
    if not below.empty:
        sys.exit(
            f"FAIL: 10_reports.csv holds {len(below)} model(s) of {pathogen} with auroc_mean < {MIN_AUROC}: "
            f"{below['model_name'].tolist()}. 10a keeps only retained models; re-run 10a."
        )

    df["_rank"] = df.apply(_run_columns_rank, axis=1)
    df = df.sort_values(["_rank", "n_compounds"], ascending=[True, False]).reset_index(drop=True)

    return df.drop(columns=["_rank", "source", "label", "assay_type", "is_merged"])


# ---------------------------------------------------------------------------
# Consensus: anchors of step 14 -> shipped consensus.py
# ---------------------------------------------------------------------------

def load_anchors(pathogen, df):
    """The anchor table of the weighted consensus of `pathogen`, or stop.

    The table describes the consensus of ONE set of models under ONE set of weights. The weights
    fingerprint that step 14 stored is recomputed here from 10_reports.csv, so anchors built before a
    retraining or a change of weights cannot be shipped.
    """
    path = os.path.join(CONSENSUS_DIR, pathogen, "anchors.json")
    if not os.path.isfile(path):
        sys.exit(f"FAIL: {path} not found. Run scripts/14_consensus_scoring.py --pathogen {pathogen} first.")
    with open(path) as f:
        a = json.load(f)

    kept = df["model_name"].tolist()
    if a["pathogen"] != pathogen or sorted(a["models"]) != sorted(kept):
        sys.exit(
            f"FAIL: the anchors of step 14 were built from other models than the ones to ship. "
            f"Only in anchors.json: {sorted(set(a['models']) - set(kept))}; "
            f"only to ship: {sorted(set(kept) - set(a['models']))}. Re-run step 14."
        )
    w_weights = np.asarray(a["w_weights"], dtype=float)
    if len(w_weights) != len(QUALITY_WEIGHT_COLS) + 1 or not np.all(w_weights == 1.0):
        sys.exit(f"FAIL: step 14 used weights {a['w_weights']} to combine the terms; the shipped consensus.py "
                 "assumes equal weights. Change src/hub_consensus.py and this script together.")

    rows = df.set_index("model_name").loc[a["models"]]
    fingerprint = cs.weights_fingerprint(
        a["models"], QUALITY_WEIGHT_COLS, rows[QUALITY_WEIGHT_COLS].to_numpy(dtype=float),
        rows["decision_cutoff_rank"].to_numpy(dtype=float), w_weights,
    )
    if fingerprint != a["weights_fingerprint"]:
        sys.exit(
            "FAIL: 10_reports.csv no longer holds the weights or cutoffs the anchors of step 14 were "
            f"built with ({pathogen}). Re-run step 14 before shipping."
        )

    table = a["weighted"]["full"]
    x, y = np.asarray(table["x"], dtype=float), np.asarray(table["y"], dtype=float)
    if not (len(x) == len(y) >= 3 and np.all(np.diff(x) > 0) and np.all(np.diff(y) >= 0)
            and (x[0], y[0], x[-1], y[-1]) == (0.0, 0.0, 1.0, 1.0)):
        sys.exit(f"FAIL: the anchor table of {pathogen} is not a monotone map from (0, 0) to (1, 1): x={x}, y={y}")
    return {
        "x": x, "y": y, "decision_rank": float(a["scale"]["decision_rank"]),
        "percentiles": a["scale"]["percentiles"], "reference": a["reference"],
        "lazyqsar_version": a["lazyqsar_version"],
    }


def render_consensus_py(anchors):
    """src/hub_consensus.py with the anchor table of the pathogen filled in."""
    with open(TEMPLATE_PATH) as f:
        src = f.read()
    for name, values in (("_ANCHOR_X", anchors["x"]), ("_ANCHOR_Y", anchors["y"])):
        src, n = re.subn(rf"^{name} = None.*$", f"{name} = {[float(v) for v in values]!r}", src, flags=re.MULTILINE)
        if n != 1:
            sys.exit(f"FAIL: could not place {name} in the consensus template ({TEMPLATE_PATH}).")
    w_cols = re.search(r"^_W_COLS = (\[.*\])$", src, flags=re.MULTILINE)
    if not w_cols or ast.literal_eval(w_cols.group(1)) != list(QUALITY_WEIGHT_COLS):
        sys.exit("FAIL: _W_COLS of the consensus template differs from QUALITY_WEIGHT_COLS in src/default.py.")
    compile(src, "consensus.py", "exec")
    return src


# ---------------------------------------------------------------------------
# Description builders
# ---------------------------------------------------------------------------

_CATEGORY = {"DR": "dose-response", "SP": "single-point"}


def _display_activity_type(activity_type: str) -> str:
    """Prose display form for a raw ChEMBL activity_type. Genuine acronyms (MIC, IC50,
    EC50, ...) are left as-is; "INHIBITION" reads as an ordinary word in a sentence, so
    it's shown title-cased instead of ChEMBL's all-caps constant. Used for metadata.yml's
    Description text (main()) -- run_columns.csv's own descriptions no longer need it."""
    return "Inhibition" if activity_type == "INHIBITION" else activity_type


def build_description(meta_row, dataset_name, dcr):
    """Human-readable sub-model description for run_columns.csv.

    The rebuilt datasets are signal-based pools (ChEMBL stage4) or transfer-pooled organism
    assays (PubChem step 08). `cutoff` is a Youden *score*, not a concentration threshold,
    so it isn't quoted here — only the model's own decision_cutoff_rank is.
    """
    source      = meta_row.get("source", "")
    assay_type  = meta_row.get("assay_type", "")
    label       = meta_row.get("label", "")
    n_assays    = int(meta_row["n_assays"]) if not pd.isna(meta_row.get("n_assays", np.nan)) else None
    n_compounds = int(meta_row["final_compounds"])
    _an = meta_row.get("added_negatives", 0)
    _ad = meta_row.get("added_decoys", 0)
    n_added     = (0 if pd.isna(_an) else int(_an)) + (0 if pd.isna(_ad) else int(_ad))

    added_str     = f"; incl. {n_added} added negatives" if n_added > 0 else ""
    threshold_str = f"Recommended threshold: {round(dcr, 3)}."
    category      = _CATEGORY.get(label, "")

    # ChEMBL pools: name the specific assay when the pool truly reduces to one
    # (member_assay_ids, from 25_pool_members.csv via 01_download_datasets_chembl.py);
    # otherwise state the count. Catch-alls always use the count (no per-assay file,
    # and by construction they merge every leftover in a deferred category).
    member_ids = meta_row.get("member_assay_ids")
    member_ids = [] if pd.isna(member_ids) else str(member_ids).split("|")
    if assay_type == "pool" and n_assays == 1 and len(member_ids) == 1:
        assays_str = f" (assay {member_ids[0]})"
    else:
        assays_str = f" of {n_assays} assay{'s' if n_assays != 1 else ''}" if n_assays else ""

    if source == "pubchem":
        if bool(meta_row.get("is_merged", False)) and not pd.isna(meta_row.get("n_members", np.nan)):
            body = (f"PubChem whole-cell organism screen merged from "
                    f"{int(meta_row['n_members'])} assays ({n_compounds} compounds{added_str})")
        else:
            body = f"PubChem whole-cell organism assay AID {dataset_name} ({n_compounds} compounds{added_str})"
    elif assay_type == "pool":
        body = f"ChEMBL {category} signal-based pool{assays_str} ({n_compounds} compounds{added_str})"
    elif assay_type == "catchall":
        body = f"ChEMBL {category} low-data catch-all pool{assays_str} ({n_compounds} compounds{added_str})"
    else:
        body = f"dataset {dataset_name} ({n_compounds} compounds{added_str})"

    return f"Probability from sub-model trained on {body}. {threshold_str}"


def consensus_description(n_models, decision_rank):
    """No commas: run_columns.csv descriptions must not contain any (see _check_description)."""
    threshold = round(decision_rank, 3)
    return (
        f"Quality-weighted consensus across the {n_models} sub-models on the same rank scale as the sub-models. "
        f"Calibrated against a reference library of 50K drug-like molecules so that a score of {threshold} "
        f"is better than 99% of them. Recommended threshold: {threshold}."
    )


def _check_description(name, description):
    """run_columns.csv is read without CSV quoting by the Hub tooling: a comma (or a quote or a line
    break) in a description would split or corrupt the row. Stop instead of writing it."""
    bad = [c for c in (",", '"', "\n", "\r") if c in description]
    if bad:
        sys.exit(f"FAIL: the description of '{name}' contains {bad!r}; run_columns.csv descriptions must not "
                 f"contain commas, quotes or line breaks: {description!r}")


# ---------------------------------------------------------------------------
# Public-facing sub-model names
# ---------------------------------------------------------------------------

_TYPE_BUCKET = {"DR": "dose_response", "SP": "single_point"}


def assign_public_names(df, path_meta):
    """Public-facing sub-model identifiers for run_columns.csv / reports.csv, replacing
    the internal training-pipeline model_name (e.g. "DR_0001", "SP_catchall").

    Scheme (all lowercase, in `df`'s existing sorted order from filter_and_sort):
      - PubChem, single AID (not merged): pubchem_aid{aid}.
      - PubChem, merged (multiple AIDs):  pubchem_{counter}.
      - ChEMBL pool reducing to one member assay: chembl_{dose_response|single_point}_chembl{id}.
      - ChEMBL, everything else (pools and catch-alls alike -- assay_type doesn't split
        the bucket, only label does): chembl_{dose_response|single_point}_{counter}.
    Counters are zero-padded to fit each bucket's actual size this run and are NOT
    persisted across refreshes: retraining can shift compound counts, which can reorder the
    sort and therefore renumber a sub-model. The public names are the model's API, so
    write_diff_summary compares them with the published run_columns.csv and flags any
    change as BREAKING.

    Returns {model_name: public_name}.
    """
    # Pass 1: classify every row, tallying counter-bucket sizes.
    plan = []  # (model_name, bucket_or_None, fixed_name_or_None)
    bucket_size = {}
    for _, r in df.iterrows():
        meta_row = path_meta.loc[r["name"]]
        if meta_row.get("source", "") == "pubchem":
            if bool(meta_row.get("is_merged", False)):
                bucket_size["pubchem"] = bucket_size.get("pubchem", 0) + 1
                plan.append((r["model_name"], "pubchem", None))
            else:
                plan.append((r["model_name"], None, f"pubchem_aid{r['name']}".lower()))
            continue

        type_bucket = _TYPE_BUCKET.get(meta_row.get("label", ""), "single_point")
        n_assays    = meta_row.get("n_assays")
        member_ids  = meta_row.get("member_assay_ids")
        member_ids  = [] if pd.isna(member_ids) else str(member_ids).split("|")
        if meta_row.get("assay_type", "") == "pool" and n_assays == 1 and len(member_ids) == 1:
            plan.append((r["model_name"], None, f"chembl_{type_bucket}_{member_ids[0]}".lower()))
        else:
            bucket = f"chembl_{type_bucket}"
            bucket_size[bucket] = bucket_size.get(bucket, 0) + 1
            plan.append((r["model_name"], bucket, None))

    widths = {b: max(1, len(str(n - 1))) for b, n in bucket_size.items()}

    # Pass 2: assign counters in the same order.
    counters = {b: 0 for b in bucket_size}
    public_name_map = {}
    for model_name, bucket, fixed_name in plan:
        if fixed_name is not None:
            public_name_map[model_name] = fixed_name
        else:
            idx = counters[bucket]
            counters[bucket] += 1
            public_name_map[model_name] = f"{bucket}_{idx:0{widths[bucket]}d}"

    if len(set(public_name_map.values())) != len(public_name_map):
        sys.exit(f"FAIL: duplicate public names assigned: {public_name_map}")

    return public_name_map


# ---------------------------------------------------------------------------
# Descriptors needed (scan sub-model dirs)
# ---------------------------------------------------------------------------

def descriptors_in_dir(parent, sub_models):
    needed = set()
    for sub in sub_models:
        sub_path = os.path.join(parent, sub)
        if not os.path.isdir(sub_path):
            continue
        for d in os.listdir(sub_path):
            if d in _DESCRIPTORS:
                needed.add(d)
    return [d for d in _DESCRIPTORS if d in needed]


# ---------------------------------------------------------------------------
# metadata.yml patcher
# ---------------------------------------------------------------------------

# Captures the full species name from the existing Title line (e.g. "Antinicrobial
# activity prediction against Acinetobacter baumannii from public ChEMBL data") —
# consistently the full binomial name across all 15 pathogens' published metadata.yml,
# unlike the Interpretation line, which mixes full and abbreviated forms depending on
# who wrote it. DOTALL is required because Title wraps across lines for most pathogens.
_TITLE_NAME_RE = re.compile(
    r"^Title:.*?against\s+(.+?)\s+from",
    flags=re.MULTILINE | re.DOTALL,
)


def patch_metadata(existing_yaml, n_models, has_pubchem, dr_activity_type, sp_activity_type):
    """Patch Output Dimension, Deployment, Description, and Interpretation. Everything else
    is preserved byte-for-byte.

    Description and Interpretation are DRAFTS: regenerated each refresh from the same
    building blocks (full species name, ChEMBL/PubChem mix, sub-model count, a
    representative DR/SP activity type each), closely mirroring the existing wording
    style — meant to be reviewed and edited per pathogen before push, not treated as
    already-approved copy. dr_activity_type/sp_activity_type are each a single type
    (e.g. "MIC", "Inhibition") picked by the caller — "MIC"/"INHIBITION" if present
    among kept sub-models, else whichever type is most common — or None if that
    category has no kept ChEMBL sub-models at all.
    """
    name_match = _TITLE_NAME_RE.search(existing_yaml)
    if not name_match:
        sys.exit(
            "FAIL: could not parse the full species name from the existing metadata.yml's "
            "Title line. Expected '... against <X> from public ... data'."
        )
    # Normalize whitespace — the published file may wrap the name across lines as a
    # YAML folded scalar (e.g. "Enterococcus\n  faecium"), and we need a single-line
    # name to splice into the new Description/Interpretation.
    full_name = re.sub(r"\s+", " ", name_match.group(1)).strip()

    # Output Dimension — no consensus column when there's only one sub-model
    # (mirrors consensus.py's own single-model shortcut).
    output_dim = n_models if n_models == 1 else 1 + n_models
    new_yaml, n_subs = re.subn(
        r"^(Output Dimension:\s*).*$",
        rf"\g<1>{output_dim}",
        existing_yaml,
        count=1,
        flags=re.MULTILINE,
    )
    if n_subs != 1:
        sys.exit("FAIL: no 'Output Dimension:' line found in metadata.yml.")

    # Deployment — preserve block-style (matches the rest of the file); handle
    # both the original two-line `- Local / - Online` and the already-updated
    # single-line `- Local` form idempotently.
    new_yaml, n_subs = re.subn(
        r"^Deployment:.*?(?=\n[A-Z])",
        "Deployment:\n  - Local",
        new_yaml,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    if n_subs != 1:
        sys.exit("FAIL: no 'Deployment:' line found in metadata.yml.")

    sources_str        = "ChEMBL and PubChem" if has_pubchem else "ChEMBL"
    sources_trained_str = "ChEMBL- and PubChem" if has_pubchem else "ChEMBL"  # dangling hyphen for "-trained"

    # Description — same folded-scalar block style as Interpretation; DRAFT, see docstring.
    # Closely mirrors the existing wording template (species name + source list are the
    # only parts that vary), rather than describing the internal scoring mechanism.
    assay_clauses = []
    if sp_activity_type:
        assay_clauses.append(f"single-point ({sp_activity_type})")
    if dr_activity_type:
        assay_clauses.append(f"dose-response ({dr_activity_type})")
    assay_clause = " and ".join(assay_clauses) if assay_clauses else "single-point and dose-response"

    new_description = (
        f"Description: Bioactivity prediction of growth inhibition in {full_name}, "
        f"trained as binary (active/inactive) classifiers from publicly available data "
        f"in {sources_str}. Independent models are trained on multiple bioactivity "
        f"datasets, corresponding to {assay_clause} assays, among others. A ranking "
        f"score is provided for each model alongside a combined consensus score."
    )
    new_yaml, n_subs = re.subn(
        r"^Description:.*?(?=\n[A-Z])",
        new_description,
        new_yaml,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    if n_subs != 1:
        sys.exit("FAIL: no 'Description:' line found in metadata.yml.")

    # Interpretation — match the whole block (the original may be wrapped across
    # multiple indented continuation lines, which YAML treats as a folded scalar).
    # No consensus clause (and singular "sub-model") when there's only one.
    if n_models == 1:
        new_interp = (
            f"Interpretation: Probability of antimicrobial activity against {full_name} "
            f"from {n_models} {sources_trained_str}-trained sub-model."
        )
    else:
        new_interp = (
            f"Interpretation: Probability of antimicrobial activity against {full_name} "
            f"from {n_models} {sources_trained_str}-trained sub-models, plus a "
            f"quality-weighted consensus score."
        )
    new_yaml, n_subs = re.subn(
        r"^Interpretation:.*?(?=\n[A-Z])",
        new_interp,
        new_yaml,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    if n_subs != 1:
        sys.exit("FAIL: no 'Interpretation:' line found in metadata.yml.")

    return new_yaml


# ---------------------------------------------------------------------------
# Runtime env + staging run
# ---------------------------------------------------------------------------

def find_conda_sh():
    """conda.sh from $CONDA_SH, else next to the `conda` found on the PATH."""
    path = os.environ.get("CONDA_SH")
    if not path:
        conda = shutil.which("conda")
        if conda:
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(conda))), "etc", "profile.d", "conda.sh")
    if not path or not os.path.exists(path):
        sys.exit(
            "FAIL: conda.sh not found. Export CONDA_SH to your conda.sh, "
            "e.g. export CONDA_SH=~/miniconda3/etc/profile.d/conda.sh"
        )
    return path


def _run_in_runtime_env(conda_sh, cmd, cwd):
    full = f"source {conda_sh} && conda activate {HUB_RUNTIME_ENV} && {cmd}"
    return subprocess.run(["bash", "-c", full], cwd=cwd, capture_output=True, text=True)


def check_runtime_env(conda_sh):
    """The staging run must use the lazyqsar that the shipped install.yml pins."""
    res = _run_in_runtime_env(
        conda_sh, "python -c \"import importlib.metadata as m; print(m.version('lazyqsar'))\"", REPO_ROOT)
    version = res.stdout.strip().splitlines()[-1] if res.returncode == 0 and res.stdout.strip() else None
    if version != _LAZYQSAR_VERSION:
        sys.exit(
            f"FAIL: conda env '{HUB_RUNTIME_ENV}' has lazyqsar {version}, the shipped install.yml pins "
            f"{_LAZYQSAR_VERSION}. {res.stderr.strip()[-300:]}"
        )
    return version


def generate_run_output(conda_sh, repo_dir, pathogen, sub_models, public_name_map, reports_csv_path,
                        run_columns_csv_path, consensus_py_path, dest_csv, keep_staging, parity_smiles):
    """Build a staging dir, run bash run.sh on the example input and copy run_output.csv to dest_csv.
    Runs it a second time on `parity_smiles` (not shipped) and returns that output as a DataFrame."""
    os.makedirs(TMP_DIR, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f"update_emh_{pathogen}_", dir=TMP_DIR)
    print(f"      staging dir: {staging}")
    try:
        framework_src = os.path.join(repo_dir, "model", "framework")
        framework_dst = os.path.join(staging, "model", "framework")
        os.makedirs(os.path.join(framework_dst, "code"),     exist_ok=True)
        os.makedirs(os.path.join(framework_dst, "columns"),  exist_ok=True)
        os.makedirs(os.path.join(framework_dst, "examples"), exist_ok=True)
        os.makedirs(os.path.join(framework_dst, "fit"),      exist_ok=True)
        open(os.path.join(framework_dst, "fit", ".gitkeep"), "a").close()

        for rel in ("code/main.py", "run.sh", "examples/run_input.csv"):
            src = os.path.join(framework_src, rel)
            if not os.path.exists(src):
                sys.exit(f"FAIL: missing {src} in --repo-dir.")
            dst = os.path.join(framework_dst, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        ckpt = os.path.join(staging, "model", "checkpoints")
        os.makedirs(os.path.join(ckpt, "models"), exist_ok=True)
        for sub in sub_models:
            src = os.path.join(MODELS_DIR, pathogen, sub)
            if not os.path.isdir(src):
                sys.exit(f"FAIL: missing checkpoints for sub-model '{sub}' at {src}.")
            # Destination dir name must match the public name main.py reads from
            # run_columns.csv, even though the source dir keeps its original name.
            shutil.copytree(src, os.path.join(ckpt, "models", public_name_map[sub]))
        shutil.copy2(reports_csv_path, os.path.join(ckpt, "reports.csv"))

        shutil.copy2(run_columns_csv_path, os.path.join(framework_dst, "columns", "run_columns.csv"))
        shutil.copy2(consensus_py_path,    os.path.join(framework_dst, "code", "consensus.py"))

        out_rel = "model/framework/examples/run_output.csv"
        res = _run_in_runtime_env(
            conda_sh,
            f"bash model/framework/run.sh model/framework "
            f"model/framework/examples/run_input.csv {out_rel}",
            cwd=staging,
        )
        if res.returncode != 0:
            sys.stdout.write(res.stdout)
            sys.stderr.write(res.stderr)
            sys.exit("FAIL: run.sh failed in staging dir.")

        produced = os.path.join(staging, out_rel)
        if not os.path.exists(produced):
            sys.exit(f"FAIL: run_output.csv not produced at {produced}.")
        shutil.copy2(produced, dest_csv)

        # Same model, on DrugBank molecules whose scores steps 12b and 14 already hold.
        parity_in = os.path.join(framework_dst, "examples", "parity_input.csv")
        pd.DataFrame({"smiles": parity_smiles}).to_csv(parity_in, index=False)
        parity_rel = "model/framework/examples/parity_output.csv"
        res = _run_in_runtime_env(
            conda_sh,
            f"bash model/framework/run.sh model/framework model/framework/examples/parity_input.csv {parity_rel}",
            cwd=staging,
        )
        if res.returncode != 0 or not os.path.exists(os.path.join(staging, parity_rel)):
            sys.stdout.write(res.stdout)
            sys.stderr.write(res.stderr)
            sys.exit("FAIL: run.sh failed in staging dir on the DrugBank parity molecules.")
        return pd.read_csv(os.path.join(staging, parity_rel))

    finally:
        if not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)
        else:
            print(f"      (kept staging dir for inspection: {staging})")


def parity_sample(pathogen, has_consensus):
    """DrugBank molecules to run through the staged model, with the scores steps 12b and 14 hold for them.

    The _PARITY_N_TOP molecules with the highest consensus of step 14, where the calibration matters
    (with a single sub-model, the highest rank of that model), and _PARITY_N_RANDOM seeded random
    others. Returns {"smiles", "db12", "db14"}: the molecules and the two frames restricted to them,
    in the same order (db14 is None without a consensus).
    """
    db12 = pd.read_csv(os.path.join(DRUGBANK_RANKS, f"{pathogen}.csv"))
    db14 = None
    if has_consensus:
        db14 = pd.read_csv(os.path.join(CONSENSUS_DIR, pathogen, "drugbank_rank.csv"))
        if len(db14) != len(db12) or not db14["smiles"].equals(db12["smiles"]):
            sys.exit(f"FAIL: the DrugBank molecules of step 14 and step 12b differ for {pathogen}. Re-run step 14.")
        order = np.argsort(-db14["consensus_rank"].to_numpy(dtype=float), kind="stable")
    else:
        order = np.argsort(-db12.iloc[:, 1].to_numpy(dtype=float), kind="stable")   # the single model's rank
    top = order[:_PARITY_N_TOP]
    others = np.setdiff1d(np.arange(len(db12)), top)
    rand = np.random.default_rng(RANDOM_SEED).choice(others, size=_PARITY_N_RANDOM, replace=False)
    idx = np.concatenate([top, rand])
    return {
        "smiles": db12["smiles"].iloc[idx].tolist(),
        "db12": db12.iloc[idx].reset_index(drop=True),
        "db14": db14.iloc[idx].reset_index(drop=True) if db14 is not None else None,
    }


def check_run_output(rout, pout, parity, df, public_name_map, anchors):
    """Parity checks on the staged model. Returns (report lines, flags); stops on a hard mismatch.

    1. On the example output: consensus_score equals src/consensus.py (the code of step 14) applied to
       the sub-model ranks printed in the same output (hard check).
    2. On the DrugBank sample: the staged consensus_score equals step 14's consensus_rank within
       _PARITY_TOL_PIPELINE (hard check), and the staged sub-model ranks equal those of step 12b; a cell
       beyond _PARITY_TOL_PIPELINE is FLAGGED, not fatal (see the comment at the constants).
    """
    lines, flags = [], []
    pub = [public_name_map[m] for m in df["model_name"]]
    has_consensus = anchors is not None

    if has_consensus:
        ranks = rout[pub].to_numpy(dtype=float)
        ok = ~np.isnan(ranks).any(axis=1)
        raw = cs.weighted_consensus(
            ranks[ok], df[QUALITY_WEIGHT_COLS].to_numpy(dtype=float),
            df["decision_cutoff_rank"].to_numpy(dtype=float), np.ones(len(QUALITY_WEIGHT_COLS) + 1))
        expected = cs.apply_anchor_table(raw, anchors["x"], anchors["y"])
        diff = float(np.max(np.abs(rout.loc[ok, "consensus_score"].to_numpy(dtype=float) - expected)))
        if diff > _PARITY_TOL_SELF:
            sys.exit(f"FAIL: consensus_score differs from src/consensus.py by {diff:.1e} (> {_PARITY_TOL_SELF}).")
        lines.append(f"consensus_score vs src/consensus.py on the printed ranks of the {len(rout)} example molecules: "
                     f"max difference {diff:.1e}")

    n = len(parity["smiles"])
    if len(pout) != n:
        sys.exit(f"FAIL: the staged model returned {len(pout)} rows for the {n} DrugBank parity molecules.")
    originals = df["model_name"].tolist()
    rank_diffs = np.abs(pout[pub].to_numpy(dtype=float) - parity["db12"][originals].to_numpy(dtype=float))
    lines.append(f"{n} DrugBank molecules ({_PARITY_N_TOP} top of the consensus + {_PARITY_N_RANDOM} random), sub-model ranks "
                 f"vs step 12b: max difference {np.nanmax(rank_diffs):.1e}; cells above 1e-3: {int((rank_diffs > 1e-3).sum())} "
                 f"of {rank_diffs.size}")
    flagged = np.argwhere(rank_diffs > _PARITY_TOL_PIPELINE)
    if len(flagged):
        n_models = len({int(j) for _, j in flagged})
        flags.append(f"{len(flagged)} sub-model cell(s) in {n_models} sub-model(s) differ from step 12b by more than "
                     f"{_PARITY_TOL_PIPELINE} (step 12b ran on other CPUs; check that it is CPU noise, not another model):")
        for i, j in flagged[:20]:
            flags.append(f"  {originals[j]} ({pub[j]}), sample row {int(i)} ({'top-scoring' if i < _PARITY_N_TOP else 'random'} "
                         f"molecule): staged {pout.at[i, pub[j]]:.4f} vs step 12b {parity['db12'].at[i, originals[j]]:.4f}")

    if has_consensus:
        cons_diff = float(np.nanmax(np.abs(pout["consensus_score"].to_numpy(dtype=float)
                                           - parity["db14"]["consensus_rank"].to_numpy(dtype=float))))
        if cons_diff > _PARITY_TOL_PIPELINE:
            sys.exit(f"FAIL: staged consensus_score differs from step 14's consensus_rank by {cons_diff:.1e} "
                     f"(> {_PARITY_TOL_PIPELINE}) on {n} DrugBank molecules.")
        hits_staged = int((pout["consensus_score"] >= anchors["decision_rank"]).sum())
        hits_step14 = int((parity["db14"]["consensus_rank"] >= anchors["decision_rank"]).sum())
        lines.append(f"consensus_score vs step 14 consensus_rank on those molecules: max difference {cons_diff:.1e}; "
                     f"at or above {anchors['decision_rank']:.2f}: {hits_staged} staged vs {hits_step14} in step 14")
    return lines, flags


# ---------------------------------------------------------------------------
# Diff summary
# ---------------------------------------------------------------------------

def write_diff_summary(path, pathogen, eosXXXX, new_df, new_columns, anchors, old_reports_csv,
                       new_descriptors, repo_dir, parity_lines, flags):
    lines = []
    lines.append(f"Refresh package for {pathogen} ({eosXXXX})")
    lines.append(f"Source clone: {repo_dir}")
    lines.append("")

    # Sub-models are compared by original_name (the stable training-pipeline identifier); the public
    # column names are compared separately below.
    new_set = set(new_df["model_name"])  # new_df is `df`, pre-remap: model_name IS the original name here
    if os.path.exists(old_reports_csv):
        old_df  = pd.read_csv(old_reports_csv)
        old_key_col = "original_name" if "original_name" in old_df.columns else "model_name"
        old_set = set(old_df[old_key_col])
        added    = sorted(new_set - old_set)
        removed  = sorted(old_set - new_set)
        common   = sorted(new_set & old_set)
        old_indexed = old_df.set_index(old_key_col)
    else:
        lines.append("(no existing reports.csv in repo-dir — treating all sub-models as new)")
        added, removed, common = sorted(new_set), [], []
        old_indexed = None

    lines.append(f"Sub-models: {len(new_set)} kept  |  +{len(added)} added  |  -{len(removed)} removed  |  ={len(common)} unchanged set")
    if added:
        lines.append(f"  added:   {added}")
    if removed:
        lines.append(f"  REMOVED: {removed}   (breaking: column disappears from run_columns.csv)")
    lines.append("")

    # The public column names (and their order) are the model's API.
    old_run_columns = os.path.join(repo_dir, "model", "framework", "columns", "run_columns.csv")
    if os.path.exists(old_run_columns):
        old_columns = pd.read_csv(old_run_columns)["name"].tolist()
        if old_columns == new_columns:
            lines.append(f"Public columns: identical to the published run_columns.csv ({len(new_columns)}, same order)")
        else:
            lines.append("Public columns: BREAKING CHANGE of the published API")
            lines.append(f"  removed:   {[c for c in old_columns if c not in new_columns]}")
            lines.append(f"  added:     {[c for c in new_columns if c not in old_columns]}")
            if sorted(old_columns) == sorted(new_columns):
                lines.append("  same names, different order")
            lines.append(f"  published: {old_columns}")
            lines.append(f"  new:       {new_columns}")
    else:
        lines.append(f"Public columns (new): {new_columns}")
    lines.append("")

    new_indexed = new_df.set_index("model_name")
    if old_indexed is not None and common:
        lines.append("Per sub-model decision_cutoff_rank drift:")
        lines.append(f"  {'original_name':40s}  {'cutoff_old':>10s} -> {'cutoff_new':>10s}")
        for m in common:
            co = float(old_indexed.loc[m, "decision_cutoff_rank"])
            cn = float(new_indexed.loc[m, "decision_cutoff_rank"])
            lines.append(f"  {m:40s}  {co:>10.4f} -> {cn:>10.4f}")
        lines.append("")

    if anchors is None:
        new_thresh_str = "N/A (single sub-model — no consensus score)"
    else:
        new_thresh_str = f"{anchors['decision_rank']:.3f} (rank scale, as every sub-model)"
    old_thresh = None
    if os.path.exists(old_run_columns):
        with open(old_run_columns) as f:
            for line in f:
                if line.startswith("consensus_score"):
                    m = re.search(r"Recommended threshold:\s*(\d+\.\d+)", line)
                    old_thresh = f"{float(m.group(1)):.3f}" if m else None
                    break
    lines.append(f"Consensus threshold: {old_thresh} (old)  ->  {new_thresh_str} (new)" if old_thresh
                 else f"Consensus threshold (new): {new_thresh_str}")
    if anchors is not None:
        ref = anchors["reference"]
        lines.append(f"Consensus calibration: reference library {ref['id']} ({ref['n']:,} molecules), table built by "
                     f"step 14 with lazyqsar {anchors['lazyqsar_version']}")
        lines.append(f"  raw consensus x : {[round(float(v), 4) for v in anchors['x']]}")
        lines.append(f"  rank y          : {[round(float(v), 4) for v in anchors['y']]}")
    lines.append("")

    old_install = os.path.join(repo_dir, "install.yml")
    old_lq = None
    if os.path.exists(old_install):
        m = re.search(r'lazyqsar\[all\]",\s*"([^"]+)"', open(old_install).read())
        old_lq = m.group(1) if m else None
    lines.append(f"lazyqsar: {old_lq} (old)  ->  {_LAZYQSAR_VERSION} (new)")

    old_models_dir = os.path.join(repo_dir, "model", "checkpoints", "models")
    if os.path.isdir(old_models_dir):
        old_descs = descriptors_in_dir(old_models_dir, os.listdir(old_models_dir))
        lines.append(f"Descriptors: old={old_descs}  ->  new={new_descriptors}")
    else:
        lines.append(f"Descriptors (new): {new_descriptors}")
    lines.append("")

    lines.append("Checks on the staged model (the hard checks passed):")
    lines.extend(f"  {l}" for l in parity_lines)
    if flags:
        lines.append("")
        lines.append("FLAGGED FOR MANUAL REVIEW (not blocking):")
        lines.extend(f"  {l}" for l in flags)

    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# Copy instructions
# ---------------------------------------------------------------------------

def write_copy_instructions(path, pathogen, repo_dir, out_dir):
    rel_repo = repo_dir
    rel_out  = out_dir
    rel_models = os.path.join(MODELS_DIR, pathogen)
    text = f"""\
After reviewing DIFF_SUMMARY.txt, copy the refresh package into the clone
(step 19 does the same, then runs `ersilia fetch` on the clone):

# 1. Checkpoints (replaces models/ entirely; --delete drops removed sub-models)
rsync -a --delete {rel_models}/ {rel_repo}/model/checkpoints/models/

# 2. reports.csv
cp {rel_out}/reports.csv {rel_repo}/model/checkpoints/reports.csv

# 3. run_columns.csv
cp {rel_out}/run_columns.csv {rel_repo}/model/framework/columns/run_columns.csv

# 4. consensus.py
cp {rel_out}/consensus.py {rel_repo}/model/framework/code/consensus.py

# 5. metadata.yml
cp {rel_out}/metadata.yml {rel_repo}/metadata.yml

# 6. install.yml (bumps lazyqsar to {_LAZYQSAR_VERSION})
cp {rel_out}/install.yml {rel_repo}/install.yml

# 7. run_output.csv
cp {rel_out}/run_output.csv {rel_repo}/model/framework/examples/run_output.csv

Then:
    cd {rel_repo}
    git diff                              # visual review
    git add -A
    git commit -m "Refresh checkpoints for {pathogen}"
    git push origin main
"""
    with open(path, "w") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# Steps of main()
# ---------------------------------------------------------------------------

def build_reports_csv(df, public_name_map, path):
    """reports.csv's model_name must match run_columns.csv's "name" (main.py looks weights/cutoffs up
    in reports.csv by the same name it read from run_columns.csv), so model_name becomes the public
    name here too. original_name keeps the training-pipeline identifier (also the on-disk checkpoint
    dir name under output/09_models/) for traceability and for 19_apply_and_fetch.py's checkpoint sync."""
    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    df_out = df.drop(columns=cols_to_drop).copy()
    df_out["original_name"] = df_out["model_name"]
    df_out["model_name"] = df_out["model_name"].map(public_name_map)
    df_out = df_out[
        ["model_name", "original_name"]
        + [c for c in df_out.columns if c not in ("model_name", "original_name")]
    ]
    df_out.to_csv(path, index=False)


def build_run_columns(df, path_meta, public_name_map, anchors, path):
    """run_columns.csv: the consensus column first (when there is one), then one per sub-model."""
    rows = []
    if anchors is not None:
        rows.append({
            "name":        "consensus_score",
            "type":        "float",
            "direction":   "high",
            "description": consensus_description(len(df), anchors["decision_rank"]),
        })
    for _, model_row in df.iterrows():
        dataset_name = model_row["name"]
        if dataset_name not in path_meta.index:
            sys.exit(f"FAIL: dataset '{dataset_name}' missing from 07_datasets_metadata.csv.")
        rows.append({
            "name":        public_name_map[model_row["model_name"]],
            "type":        "float",
            "direction":   "high",
            "description": build_description(path_meta.loc[dataset_name], dataset_name, model_row["decision_cutoff_rank"]),
        })
    for r in rows:
        _check_description(r["name"], r["description"])
    pd.DataFrame(rows, columns=["name", "type", "direction", "description"]).to_csv(path, index=False)
    return [r["name"] for r in rows]


def build_metadata_yml(df, meta_df, pathogen, repo_dir, path):
    kept_meta = meta_df[(meta_df["pathogen"] == pathogen) & (meta_df["name"].isin(df["name"]))]
    kept_sources = set(kept_meta["source"])
    has_pubchem = "pubchem" in kept_sources
    print(f"      sources in kept sub-models: {sorted(kept_sources)}")

    # Pick one representative ChEMBL activity type per category (DR/SP) for the
    # Description draft: "MIC" / "INHIBITION" if present anywhere among kept
    # sub-models (the conventional defaults), else whichever type most often tops
    # an individual pool's own frequency-ordered activity_types (from
    # 01_download_datasets_chembl.py, itself ordered by real per-assay frequency).
    dr_types_seen, sp_types_seen = set(), set()
    dr_top_votes, sp_top_votes = {}, {}
    for _, row in kept_meta[kept_meta["source"] == "chembl"].iterrows():
        types = [] if pd.isna(row.get("activity_types")) else str(row["activity_types"]).split("|")
        if not types:
            continue
        is_dr = row.get("label") == "DR"
        (dr_types_seen if is_dr else sp_types_seen).update(types)
        votes = dr_top_votes if is_dr else sp_top_votes
        votes[types[0]] = votes.get(types[0], 0) + 1  # types[0] = this pool's most common type

    def _pick_default(types_seen: set, votes: dict, default: str) -> str | None:
        if default in types_seen:
            return default
        if not votes:
            return None
        return sorted(votes, key=lambda t: (-votes[t], t))[0]

    dr_activity_type = _pick_default(dr_types_seen, dr_top_votes, "MIC")
    sp_activity_type = _pick_default(sp_types_seen, sp_top_votes, "INHIBITION")
    if sp_activity_type:
        sp_activity_type = _display_activity_type(sp_activity_type)

    metadata_src = os.path.join(repo_dir, "metadata.yml")
    if not os.path.exists(metadata_src):
        sys.exit(f"FAIL: {metadata_src} not found.")
    with open(metadata_src) as f:
        new_yaml = patch_metadata(f.read(), len(df), has_pubchem, dr_activity_type, sp_activity_type)
    with open(path, "w") as f:
        f.write(new_yaml)


def check_example_output(run_output_path, run_columns_path):
    """Column order = run_columns.csv, every value in [0, 1] (NaN allowed: a molecule that cannot be featurised)."""
    rout = pd.read_csv(run_output_path)
    rcols = pd.read_csv(run_columns_path)["name"].tolist()
    if list(rout.columns) != rcols:
        sys.exit(f"FAIL: run_output column order {list(rout.columns)} != run_columns {rcols}")
    vmin, vmax = float(np.nanmin(rout.values)), float(np.nanmax(rout.values))
    if not (0.0 <= vmin and vmax <= 1.0):
        sys.exit(f"FAIL: run_output values out of [0,1]: min={vmin} max={vmax}")
    print(f"      {len(rout)} rows x {len(rout.columns)} columns, all in [0,1].")
    return rout


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--pathogen", required=True)
    parser.add_argument("--repo-dir", default=None,
                        help="Path to the clone of ersilia-os/{eosXXXX} (default: HUB_CLONES_DIR/{eosXXXX}).")
    parser.add_argument("--keep-staging", action="store_true",
                        help="Don't delete the tmp/ staging dir (debugging).")
    args = parser.parse_args()

    pathogen = args.pathogen
    if pathogen not in ERSILIA_MODEL_IDS:
        sys.exit(f"Unknown pathogen '{pathogen}'. Known: {sorted(ERSILIA_MODEL_IDS)}")
    eosXXXX = ERSILIA_MODEL_IDS[pathogen]
    repo_dir = os.path.abspath(args.repo_dir) if args.repo_dir else os.path.join(HUB_CLONES_DIR, eosXXXX)
    if not os.path.isdir(repo_dir):
        sys.exit(f"Repo dir does not exist: {repo_dir} (run scripts/18a_clone_hub_repos.py first).")
    conda_sh = find_conda_sh()
    out_dir = os.path.join(OUTPUT_DIR, pathogen)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/8] Load inputs; runtime env '{HUB_RUNTIME_ENV}'")
    reports_df = pd.read_csv(REPORTS_PATH)
    meta_df    = pd.read_csv(METADATA_PATH)
    print(f"      lazyqsar {check_runtime_env(conda_sh)} in the runtime env (pinned {_LAZYQSAR_VERSION})")

    print(f"[2/8] Filter + sort reports for {pathogen} ({eosXXXX}); calibration of the consensus")
    df = filter_and_sort(reports_df, meta_df, pathogen)
    sub_models = df["model_name"].tolist()
    print(f"      {len(sub_models)} sub-models kept: {sub_models}")
    path_meta = meta_df[meta_df["pathogen"] == pathogen].set_index("name")
    public_name_map = assign_public_names(df, path_meta)
    print(f"      public names: {[public_name_map[m] for m in sub_models]}")

    # A single surviving sub-model has nothing to build a consensus from — mirrors the template's
    # own shortcut (len(model_names) == 1); its own rank is already calibrated against the reference.
    anchors = load_anchors(pathogen, df) if len(sub_models) > 1 else None
    if anchors is None:
        print("      single sub-model — no consensus, no anchors")
    else:
        print(f"      anchors of step 14 match 10_reports.csv (fingerprint); x={[round(float(v), 4) for v in anchors['x']]}")

    print(f"[3/8] Build reports.csv and run_columns.csv")
    reports_out = os.path.join(out_dir, "reports.csv")
    build_reports_csv(df, public_name_map, reports_out)
    run_columns_out = os.path.join(out_dir, "run_columns.csv")
    new_columns = build_run_columns(df, path_meta, public_name_map, anchors, run_columns_out)

    print(f"[4/8] Emit consensus.py (src/hub_consensus.py with the anchor table baked in)")
    consensus_out = os.path.join(out_dir, "consensus.py")
    with open(consensus_out, "w") as f:
        f.write(render_consensus_py(anchors) if anchors is not None else open(TEMPLATE_PATH).read())

    print(f"[5/8] Patch metadata.yml")
    build_metadata_yml(df, meta_df, pathogen, repo_dir, os.path.join(out_dir, "metadata.yml"))

    print(f"[6/8] Emit install.yml (lazyqsar=={_LAZYQSAR_VERSION})")
    new_descriptors = descriptors_in_dir(os.path.join(MODELS_DIR, pathogen), sub_models)
    with open(os.path.join(out_dir, "install.yml"), "w") as f:
        f.write(render_install_yml(new_descriptors))
    print(f"      descriptors: {new_descriptors}")

    print(f"[7/8] Run the model in a staging dir (examples + DrugBank sample), and check it against the pipeline")
    run_output_out = os.path.join(out_dir, "run_output.csv")
    parity = parity_sample(pathogen, anchors is not None)
    pout = generate_run_output(
        conda_sh=conda_sh, repo_dir=repo_dir, pathogen=pathogen, sub_models=sub_models,
        public_name_map=public_name_map, reports_csv_path=reports_out, run_columns_csv_path=run_columns_out,
        consensus_py_path=consensus_out, dest_csv=run_output_out, keep_staging=args.keep_staging,
        parity_smiles=parity["smiles"],
    )
    rout = check_example_output(run_output_out, run_columns_out)
    parity_lines, flags = check_run_output(rout, pout, parity, df, public_name_map, anchors)
    for line in parity_lines:
        print(f"      {line}")
    for line in flags:
        print(f"      FLAG: {line}" if not line.startswith("  ") else f"      {line}")

    print(f"[8/8] Write DIFF_SUMMARY.txt + COPY_INSTRUCTIONS.txt")
    diff_text = write_diff_summary(
        path=os.path.join(out_dir, "DIFF_SUMMARY.txt"), pathogen=pathogen, eosXXXX=eosXXXX, new_df=df,
        new_columns=new_columns, anchors=anchors,
        old_reports_csv=os.path.join(repo_dir, "model", "checkpoints", "reports.csv"),
        new_descriptors=new_descriptors, repo_dir=repo_dir, parity_lines=parity_lines, flags=flags,
    )
    copy_text = write_copy_instructions(
        path=os.path.join(out_dir, "COPY_INSTRUCTIONS.txt"), pathogen=pathogen, repo_dir=repo_dir, out_dir=out_dir,
    )

    print()
    print("=" * 72)
    print(diff_text)
    print("=" * 72)
    print(copy_text)


if __name__ == "__main__":
    main()
