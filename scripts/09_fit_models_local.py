"""
Step 09 (local) — Train LazyQSAR models for all datasets on a local machine.

Equivalent to 09_run_models.py / 09_run_models.sh but runs all datasets
sequentially without SLURM. Uses LazyQSAR's default weight paths.

Phase 1 — CV + final models
  For each dataset in 07_datasets/07_datasets_metadata.csv:
  1. 5-fold stratified CV with full metrics
       → output/09_reports/{pathogen}/{name}.csv         (step-10 compatible)
       → output/09_reports/{pathogen}/{name}_folds.json  (raw fold arrays for plots)
  2. Final model fit
       → output/09_models/{pathogen}/{model_name}/

Phase 2 — Plots
  Per dataset: class balance + ROC folds + score boxplot
  Aggregate:   ROC grid, ROC grid colored
       → output/09_plots/

Existing outputs are skipped so the script is safe to re-run after interruption.

Usage:
    python scripts/09_fit_models_local.py
    python scripts/09_fit_models_local.py --pathogens abaumannii saureus
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import stylia as st
from lazyqsar.qsar import LazyClassifierQSAR
from lazyqsar.utils.metrics import bedroc_random_baseline, bedroc_score
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from stylia import FadingColormap

root      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(root, ".."))
sys.path.append(os.path.join(root, "..", "src"))

from default import DESCRIPTORS, N_FOLDS, RANDOM_SEED
from model_name import compute_model_name

METADATA_PATH = os.path.join(REPO_ROOT, "output", "07_datasets", "07_datasets_metadata.csv")
DATASETS_DIR  = os.path.join(REPO_ROOT, "output", "07_datasets")
REPORTS_DIR   = os.path.join(REPO_ROOT, "output", "09_reports")
MODELS_DIR    = os.path.join(REPO_ROOT, "output", "09_models")
PLOTS_DIR     = os.path.join(REPO_ROOT, "output", "09_plots")

MODE = "slow"

st.set_format("print")
st.set_style("ersilia")
nc = st.NamedColors()


# ---------------------------------------------------------------------------
# CV + reports
# ---------------------------------------------------------------------------

def run_cv(smiles: list, y: list, pathogen: str, name: str, model_name: str) -> None:
    """5-fold stratified CV. Saves CSV (metrics) and JSON (raw fold arrays)."""
    records = []
    fold_data = {}
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_idx, test_idx) in enumerate(kf.split(smiles, y)):
        smiles_train = [smiles[i] for i in train_idx]
        y_train      = [y[i]      for i in train_idx]
        smiles_test  = [smiles[i] for i in test_idx]
        y_test       = [y[i]      for i in test_idx]

        model = LazyClassifierQSAR(mode=MODE)
        model.fit(smiles_list=smiles_train, y=y_train)
        scores_proba = model.predict_proba(smiles_list=smiles_test)[:, 1]
        scores_rank  = model.predict_rank(smiles_list=smiles_test)[:, 1]

        auroc           = roc_auc_score(y_test, scores_rank)
        auprc           = average_precision_score(y_test, scores_rank)
        baseline_auroc  = 0.5
        baseline_auprc  = sum(y_test) / len(y_test)
        bedroc          = bedroc_score(np.array(y_test), scores_rank)
        baseline_bedroc = bedroc_random_baseline(np.array(y_test))

        oof_auc_map = dict(zip(model.descriptor_types, model.oof_aucs_))
        oof_per_descriptor = {
            f"oof_auc_{desc}": round(oof_auc_map[desc], 4) if desc in oof_auc_map else np.nan
            for desc in DESCRIPTORS
        }
        num_batches = len(model.models[0]._model.models) if model.models else np.nan

        y_arr = np.array(y_test)

        def fmt(arr, mask):
            return ";".join(str(round(float(v), 3)) for v in arr[mask])

        records.append({
            "pathogen":                pathogen,
            "name":                    name,
            "model_name":              model_name,
            "fold":                    fold,
            "compounds_train":         len(y_train),
            "compounds_test":          len(y_test),
            "positives_train":         sum(y_train),
            "positives_test":          sum(y_test),
            "auroc":                   round(auroc, 4),
            "auprc":                   round(auprc, 4),
            "baseline_auroc":          baseline_auroc,
            "baseline_auprc":          round(baseline_auprc, 4),
            "bedroc":                  round(bedroc, 4),
            "baseline_bedroc":         round(baseline_bedroc, 4),
            "num_batches":             num_batches,
            **oof_per_descriptor,
            "predict_proba_actives":   fmt(scores_proba, y_arr == 1),
            "predict_proba_inactives": fmt(scores_proba, y_arr == 0),
            "predict_rank_actives":    fmt(scores_rank,  y_arr == 1),
            "predict_rank_inactives":  fmt(scores_rank,  y_arr == 0),
        })

        fold_data[str(fold)] = {
            "y_true":  y_test,
            "y_hat":   scores_proba.tolist(),
            "y_rank":  scores_rank.tolist(),
            "roc_auc": round(auroc, 4),
        }

        print(f"  fold {fold}: auroc={auroc:.3f}  auprc={auprc:.3f}  bedroc={bedroc:.3f}")

    report_dir = os.path.join(REPORTS_DIR, pathogen)
    os.makedirs(report_dir, exist_ok=True)

    pd.DataFrame(records).to_csv(os.path.join(report_dir, f"{model_name}.csv"), index=False)
    with open(os.path.join(report_dir, f"{model_name}_folds.json"), "w") as f:
        json.dump(fold_data, f)
    print(f"  Report saved: {os.path.join(report_dir, model_name)}.csv / _folds.json")


# ---------------------------------------------------------------------------
# Data loading for plots
# ---------------------------------------------------------------------------

def _load_fold_data(pathogen: str, model_name: str) -> dict:
    path = os.path.join(REPORTS_DIR, pathogen, f"{model_name}_folds.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot helpers (adapted from h3d-mtb-bioactivity/src/plotting_utils.py)
# ---------------------------------------------------------------------------

def _plot_class_balance(ax, data: dict, title: str = "") -> None:
    all_y = []
    for v in data.values():
        all_y.extend(v["y_true"])
    all_y = np.array(all_y, dtype=int)
    n_pos = int(all_y.sum())
    n_neg = int((all_y == 0).sum())
    bars = ax.bar([0, 1], [n_neg, n_pos], color=[nc.gray, nc.orange])
    for bar, count in zip(bars, [n_neg, n_pos]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(count), ha="center", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Neg", "Pos"])
    ax.set_ylim(0, max(n_neg, n_pos) * 1.1)
    st.label(ax, ylabel="Count", xlabel="", title=title)


def _plot_roc_folds(ax, data: dict) -> None:
    cmap = FadingColormap("plum")
    cmap.fit([0.5, 1])
    auroc_values = [v["roc_auc"] for v in data.values()]
    for v in data.values():
        color = cmap.transform([v["roc_auc"]])[0]
        fpr, tpr, _ = roc_curve(np.array(v["y_true"]), np.array(v["y_hat"]))
        ax.plot(fpr, tpr, color=color)
    mean_auroc = float(np.mean(auroc_values))
    std_auc    = float(np.std(auroc_values))
    ax.plot([], [], linestyle="none",
            label=f"Mean AUC={mean_auroc:.3f}±{std_auc:.2f}")
    ax.legend(loc="lower right")


def _plot_scores(ax, data: dict, title: str, fold: int = 0) -> None:
    v = data[str(fold)]
    y_true = np.asarray(v["y_true"], dtype=int)
    y_hat  = np.asarray(v["y_hat"],  dtype=float)
    y_rank = np.asarray(v["y_rank"], dtype=float)
    proba_neg, proba_pos = y_hat[y_true == 0], y_hat[y_true == 1]
    rank_neg,  rank_pos  = y_rank[y_true == 0], y_rank[y_true == 1]
    bplot = ax.boxplot(
        [proba_neg, proba_pos, rank_neg, rank_pos],
        positions=[0, 1, 3, 4],
        patch_artist=True,
        medianprops=dict(color=nc.plum, linewidth=0.5),
        boxprops=dict(color=nc.plum, linewidth=0.5),
        whiskerprops=dict(color=nc.plum, linewidth=0.5),
        capprops=dict(color=nc.plum, linewidth=0.5),
        flierprops=dict(marker="o", markeredgecolor=nc.plum,
                        markerfacecolor="none", markersize=2, markeredgewidth=0.5),
        manage_ticks=False,
    )
    for patch, color in zip(bplot["boxes"], [nc.gray, nc.orange, nc.gray, nc.orange]):
        patch.set_facecolor(color)
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels(["Neg", "Pos", "Neg", "Pos"])
    ax.set_xlim(-0.8, 5.3)
    st.label(ax, ylabel="proba / rank", xlabel="Proba vs Rank", title=title)


def _plot_roc_grid(ax, data: dict) -> tuple:
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    auroc_values = [v["roc_auc"] for v in data.values()]
    for v in data.values():
        y_true = np.array(v["y_true"])
        y_hat  = np.array(v["y_hat"])
        mask   = np.isfinite(y_hat) & np.isfinite(y_true)
        fpr, tpr, _ = roc_curve(y_true[mask], y_hat[mask])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.vstack(tprs).mean(axis=0)
    mean_tpr[0] = 0.0
    ax.plot(mean_fpr, mean_tpr, color=nc.plum, linewidth=0.8)
    ax.plot([0, 1], [0, 1], "--", color=nc.gray, linewidth=0.5)
    return float(np.mean(auroc_values)), float(np.std(auroc_values))


def _plot_roc_grid_colored(ax, data: dict) -> tuple:
    cmap = FadingColormap("plum")
    cmap.fit([0, 1])
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    auroc_values = [v["roc_auc"] for v in data.values()]
    for v in data.values():
        y_true = np.array(v["y_true"])
        y_hat  = np.array(v["y_hat"])
        mask   = np.isfinite(y_hat) & np.isfinite(y_true)
        fpr, tpr, _ = roc_curve(y_true[mask], y_hat[mask])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
    mean_tpr = np.vstack(tprs).mean(axis=0)
    mean_tpr[0] = 0.0
    mean_auroc = float(np.mean(auroc_values))
    ax.fill_between(mean_fpr, mean_tpr, color=cmap.transform([mean_auroc])[0], alpha=0.7)
    ax.plot(mean_fpr, mean_tpr, color=nc.plum)
    ax.plot([0, 1], [0, 1], "--", color=nc.gray, linewidth=0.5)
    return mean_auroc, float(np.std(auroc_values))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(pathogens: list | None) -> None:
    for path in [REPORTS_DIR, MODELS_DIR, PLOTS_DIR]:
        os.makedirs(path, exist_ok=True)

    meta = pd.read_csv(METADATA_PATH)
    if pathogens:
        meta = meta[meta["pathogen"].isin(pathogens)].reset_index(drop=True)

    # Precompute model names for all rows (needed in both phases)
    model_name_map = {task_id: compute_model_name(meta, task_id) for task_id in meta.index}

    # -----------------------------------------------------------------------
    # Phase 1: CV + final models
    # -----------------------------------------------------------------------
    for task_id, row in meta.iterrows():
        pathogen, name = row["pathogen"], row["name"]
        model_name = model_name_map[task_id]

        print(f"\n{'='*60}")
        print(f"  [{task_id}] {pathogen}/{name} ({model_name})")
        print(f"{'='*60}")

        report_csv = os.path.join(REPORTS_DIR, pathogen, f"{model_name}.csv")
        model_path = os.path.join(MODELS_DIR,  pathogen, model_name)
        report_done = os.path.exists(report_csv)
        model_done  = os.path.exists(os.path.join(model_path, "metadata.json"))

        if report_done and model_done:
            print("  Report and model exist — skipping")
            continue

        try:
            df     = pd.read_csv(os.path.join(DATASETS_DIR, pathogen, f"{name}.csv"))
            smiles = df["smiles"].tolist()
            y      = df["bin"].tolist()

            if not report_done:
                run_cv(smiles, y, pathogen, name, model_name)
            else:
                print("  Report exists — skipping CV")

            if not model_done:
                print("  Training final model")
                model = LazyClassifierQSAR(mode=MODE)
                model.fit(smiles_list=smiles, y=y)
                os.makedirs(model_path, exist_ok=True)
                model.save(model_path)
                print(f"  Model saved: {model_path}")
            else:
                print("  Final model exists — skipping")
        except Exception as e:
            print(f"  [ERROR] {pathogen}/{name}: {e} — skipping")

    # -----------------------------------------------------------------------
    # Phase 2: Plots
    # -----------------------------------------------------------------------
    print("\nGenerating plots...")

    available = [
        (task_id, row)
        for task_id, row in meta.iterrows()
        if os.path.exists(os.path.join(REPORTS_DIR, row["pathogen"],
                                       f"{model_name_map[task_id]}_folds.json"))
    ]

    # Per-dataset: class balance + ROC folds + score boxplot
    for task_id, row in available:
        pathogen, name = row["pathogen"], row["name"]
        model_name = model_name_map[task_id]
        try:
            data = _load_fold_data(pathogen, model_name)
            _, axs = st.create_figure(1, 3, width_ratios=[0.3, 1, 1])
            ax = axs.next()
            _plot_class_balance(ax, data, title=name)
            ax = axs.next()
            _plot_roc_folds(ax, data)
            st.label(ax, xlabel="FPR", ylabel="TPR", title=name)
            ax = axs.next()
            _plot_scores(ax, data, title=name)
            out = os.path.join(PLOTS_DIR, f"09_{pathogen}_{name}_performance.png")
            st.save_figure(out)
            print(f"  Saved: {os.path.basename(out)}")
        except Exception as e:
            print(f"  [ERROR] plot {pathogen}/{name}: {e} — skipping")

    # Aggregate ROC grids sorted by mean AUROC descending
    auroc_map = {
        (row["pathogen"], row["name"]): float(np.mean(
            [v["roc_auc"] for v in _load_fold_data(row["pathogen"], model_name_map[task_id]).values()]
        ))
        for task_id, row in available
    }
    sorted_available = sorted(
        available,
        key=lambda x: auroc_map[(x[1]["pathogen"], x[1]["name"])],
        reverse=True,
    )

    n = len(sorted_available)
    if n > 0:
        ncols = 10
        nrows = math.ceil(n / ncols)

        for colored in [False, True]:
            _, axs = st.create_figure(nrows, ncols, height=1.5, width=2)
            for i, (task_id, row) in enumerate(sorted_available):
                data = _load_fold_data(row["pathogen"], model_name_map[task_id])
                ax   = axs.next()
                if colored:
                    _plot_roc_grid_colored(ax, data)
                else:
                    _plot_roc_grid(ax, data)
                st.label(ax,
                         title=f"{row['pathogen']}\n{row['name']}",
                         ylabel="TPR" if i % ncols == 0 else "",
                         xlabel="FPR" if i >= (nrows - 1) * ncols else "")
            suffix = "_colored" if colored else ""
            out = os.path.join(PLOTS_DIR, f"09_roc_summary{suffix}.png")
            st.save_figure(out)
            print(f"  Saved: {os.path.basename(out)}")

    # Summary CSV
    pd.DataFrame([
        {"pathogen": row["pathogen"], "name": row["name"],
         "auroc": auroc_map[(row["pathogen"], row["name"])]}
        for _, row in sorted_available
    ]).to_csv(os.path.join(PLOTS_DIR, "09_summary.csv"), index=False)
    print(f"  Saved: 09_summary.csv")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LazyQSAR models locally (sequential, no SLURM)."
    )
    parser.add_argument(
        "--pathogens",
        nargs="+",
        default=None,
        help="Restrict to specific pathogens (e.g. --pathogens abaumannii saureus)",
    )
    args = parser.parse_args()
    main(args.pathogens)
