# Scripts

Each script is numbered to match its position in the pipeline. Outputs are written to `output/<step>_*/` and data to `data/raw/` or `data/processed/`.

---

## 01_download_datasets_chembl.py

Downloads ChEMBL antimicrobial datasets for all 15 pathogens from the sibling `chembl-antimicrobial-tasks` repo (default) or from the remote EOS service (`--eosvc`).

For each pathogen, the `no_pubchem` variant of the general assay files is preferred when available — these exclude compounds already covered by PubChem assays. Only datasets at the `middle` activity cutoff are retained to avoid overly permissive or stringent definitions of activity.

Individual general datasets are named `ORG_{activity_type}_{cutoff}` (e.g. `ORG_MIC_10.0`), matching the filename stem in the source zip. In addition, two aggregate datasets are built per pathogen by merging all middle-cutoff assays: `G_ORG_DR` (dose-response types: IC50, MIC, EC50, …) and `G_ORG_SP` (single-point types: INHIBITION, ACTIVITY, GI, …). Deduplication is activity-conservative: if a compound appears active in any constituent assay, it is kept as active. Aggregates with fewer than 50 actives are discarded.

**Cutoff:** individual general datasets must also have AUROC ≥ 0.7 and ≥ 50 actives to be included.

---

## 02a_download_datasets_pubchem.py

Downloads the PubChem bioassay summary from the sibling `pubchem-antimicrobial-tasks` repo (default) or from the remote EOS service (`--eosvc`). Filters to organism-level assays only (excludes single-protein assays and discarded labels), then downloads the per-assay compound CSVs for all retained assays.

---

## 02b_plot_datasets.py

Stacked horizontal bars of ChEMBL + PubChem dataset counts per pathogen alongside a scatter of active ratios per dataset and a per-pathogen breakdown of dataset types. PubChem rows labelled `discarded` are excluded. Output: `output/02_datasets/02_datasets.png`.

---

## 03_select_positives.py

Reads the combined ChEMBL and PubChem dataset metadata and extracts all active compounds (bin = 1) across every assay. SMILES are deduplicated by InChIKey — all variants of the same molecule collapse into one row — and RDKit canonical SMILES are assigned.

Produces two files in `output/03_select_positives/`:
- `03_selected_positives.csv` — full compound table with provenance (`found_in`), activity count (`n_active`), and a `split` index used by the HPC decoy run.
- `selected_positive_smiles.csv` — SMILES-only file (single `smiles` column) for direct input to Ersilia models.

**Split size:** 500 compounds per split (from `SPLIT_SIZE` in `src/default.py`).

---

## 04_setup_decoy_run.py *(HPC only)*

Prepares the HPC environment for decoy generation. Splits the positives into per-split CSVs (`output/04_positives_splits/split_XXX.csv`) and builds a Singularity/Apptainer SIF image for model `eos3e6s`. Prints the `sbatch` command to submit the SLURM array job.

Requires the project conda env at `envs/camm/` with `ersilia_apptainer` installed.

---

## 05_run_decoys.sh *(HPC only)*

SLURM array job script. Each task reads one split CSV and runs `eos3e6s` via `ersilia_apptainer` to generate 100 decoy SMILES per compound. Outputs one CSV per split to `output/05_decoys/eos3e6s_XXX.csv`.

---

## 05_run_decoys_ersilia.sh *(local only)*

Local alternative to scripts 04 and 05. Runs `ersilia` directly on `selected_positive_smiles.csv` in one shot — no splitting, no Singularity image required. Writes a single `output/05_decoys/eos3e6s_all.csv`. Run from the repo root: `bash scripts/05_run_decoys_ersilia.sh`.

---

## 06_aggregate_decoys.py

Collects decoy results into `output/06_decoys/06_eos3e6s_v1.csv`.

- **HPC path:** streams all per-split `eos3e6s_XXX.csv` files into one file. Warns if the number of completed splits is fewer than expected (incomplete job array).
- **Local path:** if `eos3e6s_all.csv` is present, copies it directly with no further processing.

With `--cleanup`, removes the intermediate `04_positives_splits/`, `05_decoys/`, and `05_logs/` directories. The SIF image (`04_decoys_sif_image/`) is kept even with cleanup as it is expensive to rebuild.

---

## 07_prepare_datasets.py

Prepares the final compound datasets for model training. Runs four stages in sequence:

1. **Metadata normalisation.** Loads ChEMBL and PubChem dataset summaries, aligns column names, and flags any ChEMBL dataset that is superseded by a PubChem assay covering the same ChEMBL ID (`keep=False`). Only `keep=True` datasets proceed to extraction and augmentation.

2. **Compound extraction.** For each kept dataset, extracts SMILES and binary activity labels (`smiles`, `bin`) from the source files — ChEMBL zip archives or PubChem flat CSVs — and writes them to `output/07_datasets/{pathogen}/{name}.csv`. Each dataset is then deduplicated at the **InChIKey** level: one row per molecule, SMILES stored as RDKit canonical SMILES. ChEMBL is already deduplicated upstream (≈no-op); PubChem is not (CID-only, non-canonical) and carries real duplicates and label conflicts that this step resolves.

3. **Decoy augmentation.** Datasets with an active ratio above 0.5 are augmented with decoy compounds generated by `eos3e6s` to bring the ratio down to ~0.1. Decoys are drawn from `output/06_decoys/06_eos3e6s_v1.csv` and assigned to actives by **InChIKey** matching. A decoy is excluded if its **InChIKey** matches a compound already in the dataset (including previously added decoys) or an active in any assay of the same pathogen; selected decoys are stored as canonical SMILES. Augmented datasets gain a `decoy` column (`True` for added rows).

4. **Metadata output.** Saves `output/07_datasets/07_datasets_metadata.csv` with compound/positive counts and active ratios recomputed from the deduplicated datasets, plus decoy counts and achieved final ratios.

**Conflict rule:** when one InChIKey appears as both active and inactive within a dataset, the **active label wins** (`bin=1`) — consistent with the ChEMBL positive selection (script 01) and the upstream PubChem CID priority.

**Thresholds:** augmentation triggered at ratio > 0.5; target ratio 0.1; up to 20 decoys per active compound (all from `src/default.py`).

---

## 07b_quality_checks.py

Per-dataset InChIKey-deduplication audit of the post-07 datasets. For each dataset, reports total rows, unparsable SMILES, unique compounds, duplicate compounds (and how many of those duplicate via more than one distinct SMILES string), label conflicts (same InChIKey carrying both `bin=0` and `bin=1`), and decoy/active collisions across other datasets of the same pathogen. Output: `output/07_datasets/07_dup_report.csv`.

---

## 07c_plot_datasets.py

Three-panel figure built from the post-augmentation metadata: datasets per pathogen, per-dataset active ratio, and total decoys added per pathogen (log-scaled). Datasets that received decoys are drawn as hollow markers in the active-ratio scatter. Output: `output/07_datasets/07_datasets.png`.

---

## 08_download_weights.py *(HPC only)*

Downloads the LazyQSAR descriptor model weights needed by step 09. Run once from the login node before submitting the SLURM array job. Re-running is safe — each file is skipped if already present. Weights are saved to `output/08_weights/` and include chemeleon, cddd (encoder + FPSim index), and CLAMP.

---

## 09_run_models.py / 09_run_models.sh *(HPC only)*

Trains a LazyQSAR model for one dataset per SLURM array task. Each task reads `output/07_datasets/{pathogen}/{name}.csv`, runs 5-fold stratified cross-validation, and saves per-fold metrics (AUROC, AUPRC, BEDROC and baselines, OOF AUCs per descriptor, raw score arrays) to `output/09_reports/{pathogen}/{name}.csv`. Then trains a final model on all data and saves it to `output/09_models/{pathogen}/{model_name}/`.

**Local alternative:** `09_fit_models_local.py` — runs all datasets sequentially using a local `ersilia` installation. Produces the same report CSV plus a `_folds.json` per dataset (raw fold arrays for plotting) and publication-ready figures in `output/09_plots/`. Accepts an optional `--pathogens` flag to restrict to a subset.

---

## 10a_aggregate_reports.py

Reads all per-dataset CV reports from `output/09_reports/` and collapses them into `output/10_reports/`. Applies a hard filter: datasets with mean CV AUROC < 0.7 are excluded and recorded in `10_discarded_models.csv`. Retained datasets are written to `10_reports.csv` with one row per dataset.

Beyond aggregated metrics (mean/std of AUROC, AUPRC, BEDROC), each dataset gets a composite quality weight from seven components: dataset type (individual > merged > general), decoy contamination fraction, mean CV AUROC, AUPRC enrichment over prevalence, BEDROC enrichment over random, total compound count, and active compound count. The `final_weight` is the mean of these seven scores; `final_normalized_weight` rescales within each pathogen so weights sum to 100 — this is used in downstream consensus scoring.

**Threshold:** `MIN_AUROC = 0.7` (from `src/default.py`).

---

## 10b_training_results.py

Per-pathogen three-panel figure: AUROC bars with cross-fold std error, out-of-fold rank-score distributions (jittered scatter + boxplot) for actives vs inactives with the `decision_cutoff_rank` overlaid, and final-weight bars. Decoy-augmented datasets render with a white bar face in the AUROC and weight panels. Accepts `--pathogen <code>` (single) or iterates all pathogens in `10_reports.csv`. Output: `output/10_reports/plots/10_training_{pathogen}.png`.

---

## 11_download_drugbank.py

Downloads DrugBank SMILES from a public GitHub mirror, validates them with RDKit, drops inorganic molecules (no carbon) and entries above the molecular-weight cap, and writes a single canonical-SMILES column sorted alphabetically to `data/processed/11_drugbank_smiles.csv`.

**Threshold:** `MW_CAP = 1000 Da`.

---

## 12a_predict_drugbank.py / 12a_predict_drugbank_local.py

Predicts DrugBank ranks for each pathogen using the trained LazyQSAR models. `12a_predict_drugbank.py` is the cluster variant — it points LazyQSAR at the project `output/08_weights/` directory; `12a_predict_drugbank_local.py` uses the descriptor weights from a local LazyQSAR install. Both accept `--pathogen <code>` (single output) or `--all_pathogens` (one pass, then split per pathogen). Output: `output/12_drugbank/{pathogen}.csv` with `smiles` + one column per sub-model.

---

## 12b_fit_transformation.py

For each pathogen, solves the tanh steepness `k*` such that the transformed consensus IQR matches the average per-model IQR, then fits a saturating-exponential meta-curve `k(M) = 2·(1 + a·(1 − e^{−M/τ}))` over the empirical `(M, k*)` points. The fitted `a` and `τ` are written to `output/12_drugbank/12b_tanh_fit.json` and consumed by script 14. A diagnostic figure and per-pathogen table are also written alongside.

---

## 13_predict_drugbank_ersilia.sh

Runs a single Ersilia Hub model on the DrugBank SMILES file (`data/processed/11_drugbank_smiles.csv`) and writes predictions to `output/13_drugbank_ersilia/<model_id>.csv`. Accepts a model ID and an optional batch size argument (default 100). Must be run in an `ersilia` conda environment — not `camm` — because ersilia and lazyqsar have conflicting numpy requirements.

---

## 14_consensus_scoring.py

Computes a weighted consensus score per DrugBank compound for each pathogen. Reads per-model prob_rank predictions from `output/12_drugbank/` and quality weights (w1–w7) from `output/10_reports/10_reports.csv`. A per-compound, per-model eighth weight (w8) linearly rewards predictions above each model's decision cutoff. The weighted mean prob_rank is passed through a tanh transformation that restores IQR compression caused by averaging; steepness k depends on the number of models via a saturating-exponential fit. Outputs weighted and unweighted variants (with and without the tanh transform) to `output/14_consensus/`. Accepts `--pathogen <code>` for a single pathogen.

---

## 15_recapitulate_models.py

Quantifies pairwise agreement between individual models on DrugBank for each pathogen. For every ordered model pair (A, B) it reports Spearman and Pearson correlation, raw hit overlap at top 10/100/500, and AUROC of A scoring against B binarized at three thresholds (0.1%, 1%, 5%). Output goes to `output/15_recapitulate_models/{pathogen}.csv`. Accepts `--pathogen <code>` for a single pathogen.

---

## 16_recapitulate_consensus.py

Measures how well the consensus score (weighted and unweighted, from step 14) recapitulates each individual model. Runs in two modes — leave-one-out (model excluded from consensus) and full (model included) — producing four CSV files per pathogen in `output/16_recapitulate_consensus/`. Metrics match step 15: Spearman, Pearson, hit overlap, and threshold AUROC. Accepts `--pathogen <code>` for a single pathogen.

---

## 16b_consensus_results.py

Per-pathogen 12-panel consensus dashboard combining DrugBank rank distributions per sub-model (with `decision_cutoff_rank` lines), AUROC histograms from step 15, weighted/unweighted consensus scores ±tanh-transformed (from step 14, split into leave-one-out and global), RMSE importance per model, and consensus-with-vs-without AUROC scatter and histograms (from step 16). Accepts `--pathogen <code>` (single) or iterates all pathogens in `config/pathogens.csv`. Output: `output/16_recapitulate_consensus/plots/16_consensus_{pathogen}.png`.

---

## 17_quality_checks.py

Generates a data and model quality dashboard per pathogen. For each pathogen it produces four files under `output/17_quality_checks/{pathogen}/`: `all_smiles_no_decoys.csv` and `all_smiles_decoys.csv` (unique InChIKeys with label-conflict, decoy-duplication, and DrugBank-overlap flags), `data_summary.csv` (one row per dataset with compound counts and per-dataset conflict/DrugBank counts), and `model_summary.csv` (one row per model with AUROC, weight, and fold-stability flags, including discarded models). A top-level `summary.csv` with one row per pathogen is also written.

**Thresholds:** `FOLD_UNSTABLE_AUROC_STD = 0.05` and `LOW_WEIGHT_THRESHOLD = 0.3` (both in `src/default.py`).

---

## 18_update_ersilia_model.py

Refreshes an already-incorporated Ersilia Hub model with newly trained checkpoints. Per pathogen, reads `output/10_reports/10_reports.csv`, `output/07_datasets/07_datasets_metadata.csv`, `output/09_models/{pathogen}/`, and `output/12_drugbank/12b_tanh_fit.json`; writes a complete refresh package to `output/18_emh_files/{pathogen}/`:

- `reports.csv` — quality report, filtered to `auroc_mean >= MIN_AUROC` and sorted by `(source, label, n_compounds desc)`.
- `run_columns.csv` — `consensus_score` row + one row per kept sub-model. Descriptions are built from assay type, activity type, units, cutoff, source-assay count, and compound count.
- `consensus.py` — canonical family template with `(a, τ)` from the 12b fit baked in.
- `metadata.yml` — patched in place from the clone's existing file: `Output Dimension`, `Deployment` (block-style `- Local` only), and `Interpretation` (regenerated with new N).
- `install.yml` — pins `ersilia-pack-utils` and `lazyqsar` versions and the per-pathogen descriptor `--only` list.
- `run_output.csv` — generated by running `bash run.sh` in a temp staging dir built from the clone + new artifacts.
- `DIFF_SUMMARY.txt` — sub-model set diff, per-sub-model `decision_cutoff_rank` drift, consensus threshold old→new, descriptor set old→new.
- `COPY_INSTRUCTIONS.txt` — paste-ready `rsync`/`cp` commands to drop the package into the clone, plus the commit/push lines.

Requires `--pathogen <p>` and `--repo-dir <path-to-clone>`. After the script, the user (with Claude) reviews `DIFF_SUMMARY.txt`, runs the copy commands, `git diff` reviews, and pushes direct to `main` on `ersilia-os/{eosXXXX}` (no PR).

**Consensus threshold (faithful):** computed by applying `consensus.py`'s W1-W7+W8 formula to the per-sub-model `decision_cutoff_rank` values (W8 = 0 at the boundary by construction), then the tanh transform with `(a, τ)` from `output/12_drugbank/12b_tanh_fit.json`. Same `(a, τ)` baked into the shipped `consensus.py`, so the recommended threshold and the production transform always agree.

---

## 19_apply_and_fetch.py

Applies the refresh package produced by `18_update_ersilia_model.py` to a local clone, then runs `ersilia fetch --from_dir` on it as a structural verification. Per pathogen:

1. `rsync -a --delete` checkpoints from `output/09_models/{pathogen}/` into `{repo-dir}/model/checkpoints/models/`.
2. Copies the 6 generated artifacts (`reports.csv`, `run_columns.csv`, `consensus.py`, `metadata.yml`, `install.yml`, `run_output.csv`) from `output/18_emh_files/{pathogen}/` into the right paths under `{repo-dir}/`.
3. `ersilia delete {eosXXXX}` — clears any cached install.
4. `ersilia fetch {eosXXXX} --from_dir {repo-dir}` — validates `metadata.yml` schema, runs `install.yml` end-to-end, and executes `run.sh` once. Fails loud on non-zero exit.

Requires `--pathogen <p>` and `--repo-dir <path-to-clone>`. After this, the user (with Claude) reviews `cd {repo-dir} && git diff`, commits, and pushes direct to `main` on `ersilia-os/{eosXXXX}`.
