"""
Step 18b — Refresh an already-incorporated Ersilia Hub model with newly trained checkpoints.

Prereq: `python scripts/18a_clone_hub_repos.py` has cloned the target eos-id's repo
        (or it already exists at --repo-dir).

The published model's repo is `ersilia-os/{eosXXXX}`. The user clones it
directly, runs this script to produce a refresh package under
`output/18_emh_files/{pathogen}/`, then manually copies the package into
the clone, reviews the diff, commits, and pushes to `main`.

Inputs:
  output/10_reports/10_reports.csv         — per-sub-model metrics + weights
  output/07_datasets/07_datasets_metadata.csv — assay context for descriptions
  output/09_models/{pathogen}/             — new sub-model checkpoints
  output/12_drugbank/12b_k_star.json       — pathogen's fixed k_star for consensus tanh
  {repo-dir}/metadata.yml                  — current published metadata, patched in place
  {repo-dir}/model/framework/{code,run.sh,examples}/ — seeds the staging dir

Outputs (all under output/18_emh_files/{pathogen}/):
  reports.csv
  run_columns.csv
  consensus.py
  metadata.yml
  install.yml
  run_output.csv
  DIFF_SUMMARY.txt
  COPY_INSTRUCTIONS.txt

Usage:
    python scripts/18b_update_ersilia_model.py \\
        --pathogen abaumannii \\
        --repo-dir /path/to/clone/eos21dr
"""

import argparse
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
from default import MIN_AUROC, ERSILIA_MODEL_IDS

REPO_ROOT     = os.path.abspath(os.path.join(root, ".."))
REPORTS_PATH  = os.path.join(REPO_ROOT, "output", "10_reports", "10_reports.csv")
METADATA_PATH = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")
MODELS_DIR    = os.path.join(REPO_ROOT, "output", "09_models")
K_STAR_PATH   = os.path.join(REPO_ROOT, "output", "12_drugbank", "12b_k_star.json")
OUTPUT_DIR    = os.path.join(REPO_ROOT, "output", "18_emh_files")
TMP_DIR       = os.path.join(REPO_ROOT, "tmp")

CONDA_SH    = os.environ.get("CONDA_SH")
RUNTIME_ENV = "cam-models-runtime"

_DROP_COLS    = ["predict_rank_actives", "predict_rank_inactives", "member_assay_ids"]
_DESCRIPTORS  = ("chemeleon", "clamp", "cddd")
_W_COLS       = ["w1", "w2", "w3", "w4", "w5", "w6"]  # six stored quality weights (w7 ramp is per-compound)

# Pinned versions for the refreshed install.yml. ersilia-pack-utils version
# matches what was shipped at initial incorporation (02_init_pathogen.py); the
# lazyqsar bump from 3.3.0 -> 3.4.2 is the reason this refresh is being done.
_ERSILIA_PACK_UTILS_VERSION = "0.1.5"
_LAZYQSAR_VERSION           = "3.4.2"


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
# consensus.py — canonical family template. Identical to CONSENSUS_PY in
# chembl-antimicrobial-hub-incorporation/scripts/02_init_pathogen.py. If the
# two ever diverge, `diff` against the clone will surface it loudly.
# ---------------------------------------------------------------------------

CONSENSUS_PY = '''\
"""Quality-weighted consensus across LazyQSAR sub-models.

Mirrors chembl-antimicrobial-models/scripts/14_consensus_scoring.py:
- W1..W6 are per-sub-model quality weights from reports.csv.
- W7 is a per-compound weight that ramps 0->1 above each sub-model's
  decision_cutoff_rank.
- All 7 weights are uniformly averaged into an effective per-compound,
  per-sub-model weight; the consensus is the weighted mean of prob_ranks;
  a tanh transform (fixed steepness k_star, solved for this pathogen by
  scripts/12b_fit_transformation.py) then restores the IQR that averaging
  shrinks (variance reduction).

NaN policy: if any sub-model returns NaN for a given compound, that
compound's consensus_score is NaN (no weighting, no averaging). The
per-sub-model columns keep the real prob_ranks where prediction succeeded
and NaN where it did not.
"""

import os
import numpy as np
import pandas as pd

_W_COLS = ["w1", "w2", "w3", "w4", "w5", "w6"]
_K_STAR = __K_STAR__


def compute_consensus(R, cols_ordered, model_names, checkpoints_dir):
    """Build the model's output matrix.

    Args:
        R:               (n, K) prob_rank matrix returned by lqsar_predict.
        cols_ordered:    list of K column names matching R's columns (also from lqsar_predict).
        model_names:     canonical sub-model order for this pathogen (length M, M <= K).
        checkpoints_dir: path to model/checkpoints/ (must contain reports.csv).

    Returns:
        results: (n, 1+M) float array, rounded to 4 decimals.
                 results[:, 0] is the tanh-transformed consensus score.
                 results[:, 1:] is the per-sub-model prob_rank reordered to match model_names.
        header:  ["consensus_score", *model_names].
    """
    name_to_idx = {c: i for i, c in enumerate(cols_ordered)}
    prob_ranks = R[:, [name_to_idx[m] for m in model_names]].astype(float)

    # Single sub-model: the consensus would just be a tanh-rescaled copy of
    # the sole prob_rank. Skip it and emit only the sub-model column.
    if len(model_names) == 1:
        return np.round(prob_ranks, 4), list(model_names)

    reports = pd.read_csv(os.path.join(checkpoints_dir, "reports.csv")).set_index("model_name")
    w_quality = np.array([reports.loc[m, _W_COLS].values for m in model_names], dtype=float)
    cutoffs   = np.array([reports.loc[m, "decision_cutoff_rank"] for m in model_names], dtype=float)

    # Compounds with any NaN prob_rank get consensus = NaN; the rest go
    # through the full weighting+tanh pipeline. Splitting like this keeps
    # the NaN policy explicit and avoids invalid-value runtime warnings.
    n, M = prob_ranks.shape
    nan_rows  = np.isnan(prob_ranks).any(axis=1)
    consensus = np.full(n, np.nan)

    if (~nan_rows).any():
        pr = prob_ranks[~nan_rows]
        c  = np.clip(cutoffs[np.newaxis, :], 0.0, 1.0 - 1e-9)
        w7 = np.where(pr <= c, 0.0, (pr - c) / (1.0 - c))

        n_good = pr.shape[0]
        n_w    = len(_W_COLS) + 1
        w_all = np.empty((n_good, M, n_w))
        w_all[:, :, :len(_W_COLS)] = w_quality
        w_all[:, :,  len(_W_COLS)] = w7
        w_eff = np.average(w_all, axis=-1, weights=np.ones(n_w))

        denom = w_eff.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = (pr * w_eff).sum(axis=1) / denom
        zero = denom == 0.0          # all weights 0 for a compound -> fall back to plain mean
        if zero.any():
            raw[zero] = pr[zero].mean(axis=1)
        consensus[~nan_rows] = 0.5 + 0.5 * np.tanh(_K_STAR * (raw - 0.5)) / np.tanh(_K_STAR / 2)

    results = np.round(np.column_stack([consensus, prob_ranks]), 4)
    header  = ["consensus_score", *model_names]
    return results, header
'''


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
    """Drop sub-models with auroc_mean < MIN_AUROC, then order them per
    _run_columns_rank (compound count descending within each group)."""
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

    df = df[df["auroc_mean"] >= MIN_AUROC].copy()
    if df.empty:
        sys.exit(f"All sub-models for {pathogen} fall below AUROC>={MIN_AUROC}.")

    df["_rank"] = df.apply(_run_columns_rank, axis=1)
    df = df.sort_values(["_rank", "n_compounds"], ascending=[True, False]).reset_index(drop=True)

    return df.drop(columns=["_rank", "source", "label", "assay_type", "is_merged"])


# ---------------------------------------------------------------------------
# Consensus threshold — faithful main.py formula at the per-sub-model boundary
# ---------------------------------------------------------------------------

def consensus_threshold(cutoffs, w_quality, k_star):
    """Apply consensus.py's formula to per-sub-model decision_cutoff_rank values,
    treating them as the prob_ranks of a compound sitting exactly on each
    sub-model's boundary. W7 = 0 at the boundary by construction.
    """
    cutoffs   = np.asarray(cutoffs,   dtype=float)
    w_quality = np.asarray(w_quality, dtype=float)
    M = len(cutoffs)
    prob_ranks = cutoffs.reshape(1, -1)
    c  = np.clip(cutoffs[None, :], 0.0, 1.0 - 1e-9)
    w7 = np.where(prob_ranks <= c, 0.0, (prob_ranks - c) / (1.0 - c))
    n_w = w_quality.shape[1] + 1
    w_all = np.empty((1, M, n_w))
    w_all[:, :, :w_quality.shape[1]] = w_quality
    w_all[:, :,  w_quality.shape[1]] = w7
    w_eff = np.average(w_all, axis=-1, weights=np.ones(n_w))
    raw = (prob_ranks * w_eff).sum(axis=1) / w_eff.sum(axis=1)
    return float((0.5 + 0.5 * np.tanh(k_star * (raw - 0.5)) / np.tanh(k_star / 2)).item())


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
    persisted across refreshes -- retraining can shift compound counts, which can
    reorder the sort and therefore renumber a sub-model; that's expected, not a bug,
    since only a mapping within this one run is needed, not stability across time.

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
# Stage + run.sh
# ---------------------------------------------------------------------------

def _run_in_runtime_env(cmd, cwd):
    full = f"source {CONDA_SH} && conda activate {RUNTIME_ENV} && {cmd}"
    return subprocess.run(["bash", "-c", full], cwd=cwd, capture_output=True, text=True)


def generate_run_output(repo_dir, pathogen, sub_models, public_name_map, reports_csv_path,
                       run_columns_csv_path, consensus_py_path, dest_csv, keep_staging):
    """Build a staging dir, run bash run.sh, copy run_output.csv to dest_csv."""
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

    finally:
        if not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)
        else:
            print(f"      (kept staging dir for inspection: {staging})")


# ---------------------------------------------------------------------------
# Diff summary
# ---------------------------------------------------------------------------

def write_diff_summary(path, pathogen, eosXXXX, new_df, old_reports_csv,
                       new_threshold, new_descriptors, repo_dir):
    lines = []
    lines.append(f"Refresh package for {pathogen} ({eosXXXX})")
    lines.append(f"Source clone: {repo_dir}")
    lines.append("")

    # Compared by original_name (the stable training-pipeline identifier), not the
    # public model_name -- the public name's counter can shift between refreshes
    # (e.g. compound counts reordering the sort) even when the same dataset survives,
    # so keying by model_name here would misreport pure renumbering as churn.
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

    new_indexed = new_df.set_index("model_name")
    if old_indexed is not None and common:
        lines.append("Per sub-model decision_cutoff_rank drift:")
        lines.append(f"  {'original_name':40s}  {'cutoff_old':>10s} -> {'cutoff_new':>10s}")
        for m in common:
            co = float(old_indexed.loc[m, "decision_cutoff_rank"])
            cn = float(new_indexed.loc[m, "decision_cutoff_rank"])
            lines.append(f"  {m:40s}  {co:>10.4f} -> {cn:>10.4f}")
        lines.append("")

    new_thresh_str = (
        f"{new_threshold:.3f}" if new_threshold is not None
        else "N/A (single sub-model — no consensus score)"
    )
    old_run_columns = os.path.join(repo_dir, "model", "framework", "columns", "run_columns.csv")
    if os.path.exists(old_run_columns):
        with open(old_run_columns) as f:
            for line in f:
                if line.startswith("consensus_score"):
                    m = re.search(r"Recommended threshold:\s*(\d+\.\d+)", line)
                    if m:
                        lines.append(f"Consensus threshold: {float(m.group(1)):.3f} (old)  ->  {new_thresh_str} (new)")
                    break
    else:
        lines.append(f"Consensus threshold (new): {new_thresh_str}")
    lines.append("")

    old_models_dir = os.path.join(repo_dir, "model", "checkpoints", "models")
    if os.path.isdir(old_models_dir):
        old_descs = descriptors_in_dir(old_models_dir, os.listdir(old_models_dir))
        lines.append(f"Descriptors: old={old_descs}  ->  new={new_descriptors}")
    else:
        lines.append(f"Descriptors (new): {new_descriptors}")

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
After reviewing DIFF_SUMMARY.txt, copy the refresh package into the clone:

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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--pathogen", required=True)
    parser.add_argument("--repo-dir", required=True,
                        help="Path to the user's clone of ersilia-os/{eosXXXX}.")
    parser.add_argument("--keep-staging", action="store_true",
                        help="Don't delete the tmp/ staging dir (debugging).")
    args = parser.parse_args()

    if not CONDA_SH:
        sys.exit(
            "FAIL: CONDA_SH environment variable is not set. Export it to your "
            "machine's conda.sh, e.g. export CONDA_SH=~/miniconda3/etc/profile.d/conda.sh"
        )
    if not os.path.exists(CONDA_SH):
        sys.exit(f"FAIL: CONDA_SH does not exist: {CONDA_SH}")

    pathogen = args.pathogen
    repo_dir = os.path.abspath(args.repo_dir)
    if not os.path.isdir(repo_dir):
        sys.exit(f"--repo-dir does not exist: {repo_dir}")
    if pathogen not in ERSILIA_MODEL_IDS:
        sys.exit(f"Unknown pathogen '{pathogen}'. Known: {sorted(ERSILIA_MODEL_IDS)}")
    eosXXXX = ERSILIA_MODEL_IDS[pathogen]

    out_dir = os.path.join(OUTPUT_DIR, pathogen)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/8] Load inputs")
    reports_df = pd.read_csv(REPORTS_PATH)
    meta_df    = pd.read_csv(METADATA_PATH)
    with open(K_STAR_PATH) as f:
        k_star_map = json.load(f)

    print(f"[2/8] Filter + sort reports for {pathogen} ({eosXXXX})")
    df = filter_and_sort(reports_df, meta_df, pathogen)
    sub_models = df["model_name"].tolist()
    print(f"      {len(sub_models)} sub-models kept: {sub_models}")

    path_meta = meta_df[meta_df["pathogen"] == pathogen].set_index("name")
    public_name_map = assign_public_names(df, path_meta)
    print(f"      public names: {[public_name_map[m] for m in sub_models]}")

    # A single surviving sub-model has nothing to build a consensus from — mirrors
    # consensus.py's own compute_consensus() shortcut (len(model_names) == 1). These
    # pathogens correctly have no k_star fit by scripts/12b_fit_transformation.py.
    has_consensus = len(sub_models) > 1
    if has_consensus:
        if pathogen not in k_star_map:
            sys.exit(
                f"FAIL: no k_star entry for '{pathogen}' in {K_STAR_PATH}. "
                "Run scripts/12b_fit_transformation.py first."
            )
        k_star = float(k_star_map[pathogen]["k_star"])
        print(f"      k_star: {k_star:.4f}")
    else:
        k_star = None
        print(f"      single sub-model — consensus/k_star not applicable")

    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    reports_out = os.path.join(out_dir, "reports.csv")
    # reports.csv's model_name must match run_columns.csv's "name" (main.py looks
    # weights/cutoffs up in reports.csv by the same name it read from run_columns.csv),
    # so model_name becomes the new public_name_map value here too. original_name keeps
    # the training-pipeline identifier (also the on-disk checkpoint dir name under
    # output/09_models/) for traceability and for 19_apply_and_fetch.py's checkpoint sync.
    df_out = df.drop(columns=cols_to_drop).copy()
    df_out["original_name"] = df_out["model_name"]
    df_out["model_name"] = df_out["model_name"].map(public_name_map)
    df_out = df_out[
        ["model_name", "original_name"]
        + [c for c in df_out.columns if c not in ("model_name", "original_name")]
    ]
    df_out.to_csv(reports_out, index=False)

    print(f"[3/8] Build run_columns.csv (faithful threshold + per-assay descriptions)")
    rows = []
    if has_consensus:
        cutoffs   = [float(df.set_index("model_name").loc[m, "decision_cutoff_rank"]) for m in sub_models]
        w_quality = [df.set_index("model_name").loc[m, _W_COLS].values for m in sub_models]
        cons_thresh = round(consensus_threshold(cutoffs, w_quality, k_star), 3)
        print(f"      consensus threshold (new): {cons_thresh}")
        rows.append({
            "name":      "consensus_score",
            "type":      "float",
            "direction": "high",
            "description": (
                f"Tanh-transformed quality-weighted consensus probability across the "
                f"{len(sub_models)} sub-models. Recommended threshold: {cons_thresh}."
            ),
        })
    else:
        cons_thresh = None

    for _, model_row in df.iterrows():
        dataset_name = model_row["name"]
        if dataset_name not in path_meta.index:
            sys.exit(f"FAIL: dataset '{dataset_name}' missing from 07_datasets_metadata.csv.")
        meta_row = path_meta.loc[dataset_name]
        rows.append({
            "name":      public_name_map[model_row["model_name"]],
            "type":      "float",
            "direction": "high",
            "description": build_description(meta_row, dataset_name, model_row["decision_cutoff_rank"]),
        })

    run_columns_out = os.path.join(out_dir, "run_columns.csv")
    pd.DataFrame(rows, columns=["name", "type", "direction", "description"]).to_csv(
        run_columns_out, index=False
    )

    print(f"[4/8] Emit consensus.py (canonical template)")
    consensus_out = os.path.join(out_dir, "consensus.py")
    # Bake the pathogen's fixed k_star into the shipped template so the production
    # transform matches the recommended threshold (both derive from 12b_k_star.json).
    consensus_src = CONSENSUS_PY.replace("__K_STAR__", repr(k_star))
    if "__K_STAR__" in consensus_src:
        sys.exit("FAIL: k_star placeholder not substituted in consensus.py template.")
    with open(consensus_out, "w") as f:
        f.write(consensus_src)

    print(f"[5/8] Patch metadata.yml")
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
        new_yaml = patch_metadata(
            f.read(), len(sub_models), has_pubchem, dr_activity_type, sp_activity_type
        )
    metadata_out = os.path.join(out_dir, "metadata.yml")
    with open(metadata_out, "w") as f:
        f.write(new_yaml)

    print(f"[6/8] Emit install.yml (lazyqsar=={_LAZYQSAR_VERSION})")
    new_descriptors = descriptors_in_dir(os.path.join(MODELS_DIR, pathogen), sub_models)
    install_out = os.path.join(out_dir, "install.yml")
    with open(install_out, "w") as f:
        f.write(render_install_yml(new_descriptors))
    print(f"      descriptors: {new_descriptors}")

    print(f"[7/8] Run model in staging dir to generate run_output.csv")
    run_output_out = os.path.join(out_dir, "run_output.csv")
    generate_run_output(
        repo_dir=repo_dir,
        pathogen=pathogen,
        sub_models=sub_models,
        public_name_map=public_name_map,
        reports_csv_path=reports_out,
        run_columns_csv_path=run_columns_out,
        consensus_py_path=consensus_out,
        dest_csv=run_output_out,
        keep_staging=args.keep_staging,
    )

    rout = pd.read_csv(run_output_out)
    rcols = pd.read_csv(run_columns_out)["name"].tolist()
    if list(rout.columns) != rcols:
        sys.exit(f"FAIL: run_output column order {list(rout.columns)} != run_columns {rcols}")
    vmin, vmax = float(rout.values.min()), float(rout.values.max())
    if not (0.0 <= vmin and vmax <= 1.0):
        sys.exit(f"FAIL: run_output values out of [0,1]: min={vmin} max={vmax}")
    print(f"      {len(rout)} rows x {len(rout.columns)} columns, all in [0,1].")

    print(f"[8/8] Write DIFF_SUMMARY.txt + COPY_INSTRUCTIONS.txt")
    diff_text = write_diff_summary(
        path=os.path.join(out_dir, "DIFF_SUMMARY.txt"),
        pathogen=pathogen,
        eosXXXX=eosXXXX,
        new_df=df,
        old_reports_csv=os.path.join(repo_dir, "model", "checkpoints", "reports.csv"),
        new_threshold=cons_thresh,
        new_descriptors=new_descriptors,
        repo_dir=repo_dir,
    )
    copy_text = write_copy_instructions(
        path=os.path.join(out_dir, "COPY_INSTRUCTIONS.txt"),
        pathogen=pathogen,
        repo_dir=repo_dir,
        out_dir=out_dir,
    )

    print()
    print("=" * 72)
    print(diff_text)
    print("=" * 72)
    print(copy_text)


if __name__ == "__main__":
    main()
