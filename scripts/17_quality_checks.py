"""
Script 17 — Per-pathogen data and model quality summary.

For each pathogen, reads:
  - output/07_datasets/{pathogen}/ — raw compound CSVs
  - output/07_datasets/07_datasets_metadata.csv — dataset metadata
  - output/10_reports/10_reports.csv — model CV performance
  - output/10_reports/10_discarded_models.csv — models below MIN_AUROC

Outputs per pathogen in output/17_quality_checks/{pathogen}/:
  all_smiles_no_decoys.csv — unique InChIKeys excluding added decoys
                             inchikey | canonical_smiles | n_active | n_inactive | found_in | in_drugbank
  all_smiles_decoys.csv    — same but including added decoys, with quality flags
                             + n_decoy | label_conflict | decoy_inactive_dup | intra_dataset_conflicts
  data_summary.csv         — one row per dataset: counts + per-dataset label/DrugBank flags
                             name | source | label | n_compounds | n_positives | n_decoys | final_ratio
                             | active_label_conflict | inactive_label_conflict | n_drugbank_overlap
  model_summary.csv        — one row per model (kept + discarded)
                             name | model_name | n_compounds | n_positives | auroc_mean | auroc_std
                             | final_weight | fold_unstable | low_weight | discarded

Top-level output:
  output/17_quality_checks/summary.csv — one row per pathogen

Quality flags:
  label_conflict      — compound is active in ≥1 dataset AND inactive/decoy in ≥1 dataset
  decoy_inactive_dup  — added decoy also appears as a real inactive
  in_drugbank         — training compound appears in DrugBank screening set
  fold_unstable       — model auroc_std > FOLD_UNSTABLE_AUROC_STD (src/default.py)
  low_weight          — model final_weight < LOW_WEIGHT_THRESHOLD (src/default.py)
  discarded           — model excluded by MIN_AUROC filter in script 10

Usage:
    python scripts/17_quality_checks.py
    python scripts/17_quality_checks.py --pathogen ecoli
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "src"))
from default import FOLD_UNSTABLE_AUROC_STD, LOW_WEIGHT_THRESHOLD

REPO_ROOT      = os.path.abspath(os.path.join(root, ".."))
DATASETS_DIR   = os.path.join(REPO_ROOT, "output", "07_datasets")
METADATA_PATH  = os.path.join(DATASETS_DIR, "07_datasets_metadata.csv")
DRUGBANK_PATH  = os.path.join(REPO_ROOT, "data", "processed", "11_drugbank_smiles.csv")
REPORTS_PATH   = os.path.join(REPO_ROOT, "output", "10_reports", "10_reports.csv")
DISCARDED_PATH = os.path.join(REPO_ROOT, "output", "10_reports", "10_discarded_models.csv")
OUT_DIR        = os.path.join(REPO_ROOT, "output", "17_quality_checks")
os.makedirs(OUT_DIR, exist_ok=True)

_smi_cache: dict[str, tuple[str, str] | None] = {}


def _process_smiles(smi: str) -> tuple[str, str] | None:
    """Return (inchikey_or_fallback, canonical_smiles), or None if unparseable."""
    if smi in _smi_cache:
        return _smi_cache[smi]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        _smi_cache[smi] = None
        return None
    canonical = Chem.MolToSmiles(mol)
    key = MolToInchiKey(mol) or canonical
    result = (key, canonical)
    _smi_cache[smi] = result
    return result


def load_drugbank_inchikeys(path: str) -> set[str]:
    if not os.path.exists(path):
        print(f"[WARN] DrugBank file not found: {path} — in_drugbank will be False for all")
        return set()
    df = pd.read_csv(path)
    keys: set[str] = set()
    n_invalid = 0
    for smi in df["smiles"].dropna():
        result = _process_smiles(str(smi))
        if result:
            keys.add(result[0])
        else:
            n_invalid += 1
    print(f"DrugBank: {len(keys)} unique InChIKeys loaded ({n_invalid} unparseable skipped)")
    return keys


def collect_pathogen(
    pathogen: str,
    dataset_names: list[str],
) -> tuple[dict[str, dict], int]:
    """
    Collect all compound records for a pathogen across all its datasets.

    Returns:
        records   — inchikey -> {canonical_smiles, datasets: {name -> {n_active, n_inactive, n_decoy}}}
        n_invalid — count of SMILES that could not be parsed
    """
    records: dict[str, dict] = {}
    n_invalid = 0

    for name in tqdm(dataset_names, desc=f"  {pathogen}", unit="dataset", leave=False):
        csv_path = os.path.join(DATASETS_DIR, pathogen, f"{name}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        has_decoy_col = "decoy" in df.columns

        for _, row in df.iterrows():
            smi = str(row["smiles"])
            bin_val = int(row["bin"])
            is_decoy = bool(row["decoy"]) if has_decoy_col else False

            result = _process_smiles(smi)
            if result is None:
                n_invalid += 1
                continue
            key, canonical = result

            if key not in records:
                records[key] = {"canonical_smiles": canonical, "datasets": {}}
            rec = records[key]

            if name not in rec["datasets"]:
                rec["datasets"][name] = {"n_active": 0, "n_inactive": 0, "n_decoy": 0}
            ds = rec["datasets"][name]

            if is_decoy:
                ds["n_decoy"] += 1
            elif bin_val == 1:
                ds["n_active"] += 1
            else:
                ds["n_inactive"] += 1

    return records, n_invalid


def build_df(
    records: dict[str, dict],
    include_decoys: bool,
    drugbank_keys: set[str],
) -> pd.DataFrame:
    rows = []
    for key, rec in records.items():
        datasets = rec["datasets"]

        n_active   = sum(ds["n_active"]   for ds in datasets.values())
        n_inactive = sum(ds["n_inactive"] for ds in datasets.values())
        n_decoy    = sum(ds["n_decoy"]    for ds in datasets.values())
        n_orig     = n_active + n_inactive

        if not include_decoys and n_orig == 0:
            continue

        row: dict = {
            "inchikey":         key,
            "canonical_smiles": rec["canonical_smiles"],
            "n_active":         n_active,
            "n_inactive":       n_inactive,
            "found_in":         ";".join(sorted(datasets.keys())),
            "in_drugbank":      key in drugbank_keys,
        }

        if include_decoys:
            intra = [
                name for name, ds in datasets.items()
                if (ds["n_active"] > 0 and (ds["n_inactive"] + ds["n_decoy"]) > 0)
                or (ds["n_decoy"] > 0 and ds["n_inactive"] > 0)
            ]
            row["n_decoy"]                 = n_decoy
            row["label_conflict"]          = n_active > 0 and (n_inactive + n_decoy) > 0
            row["decoy_inactive_dup"]      = n_decoy > 0 and n_inactive > 0
            row["intra_dataset_conflicts"] = ";".join(sorted(intra)) if intra else ""

        rows.append(row)

    if include_decoys:
        cols = ["inchikey", "canonical_smiles", "n_active", "n_inactive", "n_decoy",
                "label_conflict", "decoy_inactive_dup", "intra_dataset_conflicts",
                "found_in", "in_drugbank"]
    else:
        cols = ["inchikey", "canonical_smiles", "n_active", "n_inactive",
                "found_in", "in_drugbank"]

    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values("inchikey")
        .reset_index(drop=True)
    )


def _in_dataset(found_in_series: pd.Series, name: str) -> pd.Series:
    return found_in_series.apply(lambda x: name in str(x).split(";"))


def build_data_summary(
    pathogen: str,
    dataset_names: list[str],
    meta_df: pd.DataFrame,
    df_wd: pd.DataFrame,
    records: dict,
    reports_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per dataset: basic stats plus per-dataset label conflict and DrugBank counts."""
    meta_p = meta_df[meta_df["pathogen"] == pathogen].set_index("name")
    name_to_model = (
        reports_df[reports_df["pathogen"] == pathogen]
        .set_index("name")["model_name"]
        .to_dict()
        if not reports_df.empty else {}
    )
    rows = []
    for name in dataset_names:
        if name not in meta_p.index:
            continue
        m = meta_p.loc[name]
        in_ds = _in_dataset(df_wd["found_in"], name)

        n_active_conflict = 0
        n_inactive_conflict = 0
        for rec in records.values():
            ds = rec["datasets"]
            if name not in ds:
                continue
            this = ds[name]
            others = [v for k, v in ds.items() if k != name]
            other_active   = sum(v["n_active"]             for v in others)
            other_inactive = sum(v["n_inactive"] + v["n_decoy"] for v in others)
            if this["n_active"] > 0 and other_inactive > 0:
                n_active_conflict += 1
            if this["n_inactive"] > 0 and other_active > 0:
                n_inactive_conflict += 1

        rows.append({
            "name":                    name,
            "future_model_name":       name_to_model.get(name, ""),
            "source":                  m.get("source", ""),
            "label":                   m.get("label", ""),
            "n_compounds":             int(m.get("final_compounds", m.get("compounds", 0))),
            "n_positives":             int(m.get("positives", 0)),
            "n_decoys":                int(m.get("decoys", 0)),
            "final_ratio":             round(float(m.get("final_ratio", 0.0)), 4),
            "active_label_conflict":   n_active_conflict,
            "inactive_label_conflict": n_inactive_conflict,
            "n_drugbank_overlap":      int(df_wd.loc[in_ds, "in_drugbank"].sum()),
        })
    return pd.DataFrame(rows)


def build_model_summary(
    pathogen: str,
    reports_df: pd.DataFrame,
    discarded_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per model (kept and discarded) with fold-stability and weight flags."""
    keep_cols = ["name", "model_name", "n_compounds", "n_positives", "auroc_mean", "auroc_std", "final_weight"]

    if not reports_df.empty:
        kept = reports_df[reports_df["pathogen"] == pathogen][keep_cols].copy()
    else:
        kept = pd.DataFrame(columns=keep_cols)

    kept["fold_unstable"] = kept["auroc_std"] > FOLD_UNSTABLE_AUROC_STD
    kept["low_weight"]    = kept["final_weight"] < LOW_WEIGHT_THRESHOLD
    kept["discarded"]     = False

    if not discarded_df.empty:
        disc = discarded_df[discarded_df["pathogen"] == pathogen].copy()
        if not disc.empty:
            disc = disc.rename(columns={"mean_auroc": "auroc_mean"})
            for col in [c for c in keep_cols if c not in disc.columns]:
                disc[col] = float("nan")
            disc["fold_unstable"] = False
            disc["low_weight"]    = False
            disc["discarded"]     = True
            kept = pd.concat([kept[kept.columns], disc[kept.columns]], ignore_index=True)

    return kept.round({"auroc_mean": 4, "auroc_std": 4, "final_weight": 4})


def run(
    pathogen: str,
    dataset_names: list[str],
    drugbank_keys: set[str],
    meta_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    discarded_df: pd.DataFrame,
) -> dict:
    """Process one pathogen. Returns a summary row dict for the top-level summary.csv."""
    records, n_invalid = collect_pathogen(pathogen, dataset_names)

    out_path = os.path.join(OUT_DIR, pathogen)
    os.makedirs(out_path, exist_ok=True)

    df_nd = build_df(records, include_decoys=False, drugbank_keys=drugbank_keys)
    df_nd.to_csv(os.path.join(out_path, "all_smiles_no_decoys.csv"), index=False)

    df_wd = build_df(records, include_decoys=True, drugbank_keys=drugbank_keys)
    df_wd.to_csv(os.path.join(out_path, "all_smiles_decoys.csv"), index=False)

    ds_df = build_data_summary(pathogen, dataset_names, meta_df, df_wd, records, reports_df)
    ds_df.to_csv(os.path.join(out_path, "data_summary.csv"), index=False)

    ms_df = build_model_summary(pathogen, reports_df, discarded_df)
    ms_df.to_csv(os.path.join(out_path, "model_summary.csv"), index=False)

    n_label_conflict = int(df_wd["label_conflict"].sum())
    n_db_overlap     = int(df_wd["in_drugbank"].sum())
    n_models         = int((ms_df["discarded"] == False).sum())
    n_discarded      = int(ms_df["discarded"].sum())
    n_fold_unstable  = int(ms_df["fold_unstable"].sum())
    n_low_weight     = int(ms_df["low_weight"].sum())
    auroc_vals       = ms_df.loc[ms_df["discarded"] == False, "auroc_mean"].dropna()
    auroc_median     = round(float(auroc_vals.median()), 4) if len(auroc_vals) > 0 else float("nan")
    n_decoys_total   = int(meta_df[meta_df["pathogen"] == pathogen]["decoys"].sum())

    print(
        f"[{pathogen}]  compounds={len(df_nd)}  actives={int(df_nd['n_active'].gt(0).sum())}  "
        f"decoys={n_decoys_total}  drugbank_overlap={n_db_overlap}  label_conflicts={n_label_conflict}  |  "
        f"models={n_models}  discarded={n_discarded}  auroc_median={auroc_median}  "
        f"fold_unstable={n_fold_unstable}  low_weight={n_low_weight}"
    )

    return {
        "pathogen":           pathogen,
        "n_datasets":         len(dataset_names),
        "n_compounds_unique": len(df_nd),
        "n_actives":          int(df_nd["n_active"].gt(0).sum()),
        "n_inactives":        int(df_nd["n_inactive"].gt(0).sum()),
        "n_decoys":           n_decoys_total,
        "n_drugbank_overlap": n_db_overlap,
        "n_label_conflicts":  n_label_conflict,
        "n_models":           n_models,
        "n_discarded":        n_discarded,
        "auroc_median":       auroc_median,
        "n_fold_unstable":    n_fold_unstable,
        "n_low_weight":       n_low_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-pathogen data and model quality summary."
    )
    parser.add_argument("--pathogen", default=None, help="Process a single pathogen only.")
    args = parser.parse_args()

    meta         = pd.read_csv(METADATA_PATH)
    drugbank_keys = load_drugbank_inchikeys(DRUGBANK_PATH)

    reports_df = pd.read_csv(REPORTS_PATH) if os.path.isfile(REPORTS_PATH) else pd.DataFrame()
    try:
        discarded_df = pd.read_csv(DISCARDED_PATH)
    except Exception:
        discarded_df = pd.DataFrame(columns=["pathogen", "name", "mean_auroc"])

    pathogens = [args.pathogen] if args.pathogen else list(dict.fromkeys(meta["pathogen"]))

    summary_rows = []
    for pathogen in pathogens:
        dataset_names = meta.loc[meta["pathogen"] == pathogen, "name"].tolist()
        row = run(pathogen, dataset_names, drugbank_keys, meta, reports_df, discarded_df)
        summary_rows.append(row)

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary → {summary_path}")
    print(f"Outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
