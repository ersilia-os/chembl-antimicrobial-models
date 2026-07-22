# Scripts

Each script is numbered to match its position in the pipeline. Outputs are written to `output/<step>_*/` and data to `data/raw/` or `data/processed/`.

---

## 01_download_datasets_chembl.py

Loads the **signal-based pooled** ChEMBL datasets (stage4) for all 15 pathogens from the sibling `chembl-antimicrobial-tasks` repo (default) or from the remote EOS service (`--eosvc`). Reads `output/stage4/<pathogen>/` in the tasks repo; the old `17/19/20_*` "final"/"general" files no longer exist upstream.

Two pool families are ingested, both DR (dose-response) and SP (single-point) categories:
- **Grown pools** — `25_pools/{DR,SP}/<pool_id>.csv.gz`, enumerated from `25_pool_summary.csv`. These are exactly the pools flagged `modelled=True` in `24_cv_summary.csv`.
- **Catch-all pools** — `26_pools/{DR,SP}/<category>_catchall.csv.gz`, enumerated from `26_cv_summary.csv` (low-data aggregation, present for a subset of pathogens).

Each pool CSV has columns `inchikey, compound_chembl_id, smiles, value, unit, bin`. Compound/positive counts are recomputed from the actual files (not trusted from the summary). Metadata is emitted to `data/processed/chembl/<pathogen>/01_chembl_datasets.csv` (+ combined `01_chembl_datasets_all.csv`) with columns compatible with the contract read by scripts 03/07a: `label` = category (DR/SP), `assay_type` = `pool`/`catchall`, `name` = pool id, `auroc` = grown/CV AUROC, `cutoff` = CV Youden score cutoff, `n_assays` = constituent assay count, plus `compounds/positives/ratio` and a `pool_step` (25/26) provenance flag. `activity_type`/`unit` are left blank because pools mix activity types.

**Selection (decided 2026-07-13):** keep *all* pools present in 25_pools + 26_pools — no AUROC/size filter here (that is re-applied downstream in 10a). First-pass `23_pools` are **not** used, even for pathogens whose growth step produced no grown pools (hpylori, ngonorrhoeae end up with only catch-all pools). Result: 145 pools total (137 grown + 8 catch-all) across all 15 pathogens.

---

## 02a_download_datasets_pubchem.py

Loads the final **organism (whole-cell) pooled datasets** from the `pubchem-antimicrobial-tasks` **step 08** (`output/08_transfer_pool_organism/`), from the sibling repo (default) or the remote EOS service (`--eosvc`). Upstream, step 07 folds near-duplicate organism assays and step 08 transfer-pools those datasets (the PubChem analogue of ChEMBL stage4 pooling), so this script just ingests the finished pools — no re-derivation from `06_summary.csv`, no own dedup.

Reads `08_pool_summary.csv` (one row per pool) and `08_pool_members.csv` (pool → member assay AIDs), copies each pool file `08_transfer_pool_organism/{code}/{pool_id}.csv` → `data/raw/pubchem/{code}/{pool_id}.csv`, and writes metadata to `data/processed/pubchem/02_pubchem_datasets_organism.csv`. `pool_id` (`name`) is a string (e.g. `1242` or `485275_merged3`). Per pool, `member_aids`/`n_members` are the union of underlying assay AIDs from the members file, and `is_merged` = spans >1 assay. Compound/positive counts are recomputed from the copied files. single_protein assays are not used.

**Selection:** all step-08 organism pools are kept — the ChEMBL-overlap and ≥100-datapoint filtering is applied upstream in the PubChem pipeline. Result (current upstream): 40 pools (27 single-assay + 13 multi-assay) across 6 pathogens.

---

## 02b_plot_datasets.py

Four-panel figure from the ChEMBL + PubChem dataset metadata: datasets per pathogen (stacked bars, ChEMBL vs PubChem), active ratio per dataset, compounds per dataset (log), and a dataset-type breakdown per pathogen. Types are ChEMBL pool category (DR/SP) plus PubChem organism datasets split into merged vs single-assay. Output: `output/02_datasets/02_datasets.png`.

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

## 06b_plot_decoys.py

Decoy-quality figure comparing eos3e6s decoys against random reference compounds. Loads `06_eos3e6s_v1.csv`, selects N reference rows (`--compounds`), assigns each reference 10 random decoys drawn from its own `smi_*` columns (→ N×10), and samples N×10 random reference compounds (the `input` column) across the whole table as a baseline. Frees the table from memory, prints a summary, and saves a 2×3 stylia figure to `output/06_decoys/06b_decoys.png`:

- **Panel 1:** Tanimoto similarity to the reference (Morgan/ECFP4), ref–decoy vs ref–random.
- **Panels 2–6:** absolute per-pair difference in MW, LogP, HBA, HBD, and rotatable bonds, ref–decoy vs ref–random.

Each reference is paired with its own 10 decoys and 10 randoms, so both distributions per panel are size N×10 (symmetric).

**Defaults:** `--compounds` 100; 10 decoys per reference; fingerprint Morgan radius 2 / 2048 bits (ECFP4). Row selection, decoy assignment, and random sampling all use `RANDOM_SEED` from `src/default.py`. The random baseline is drawn from the reference compounds (`input`), not from the decoy pool.

---

## 07a_prepare_datasets.py

Prepares the final compound datasets for model training. Runs in three stages:

1. **Metadata.** Concatenates the ChEMBL (`01`) and PubChem (`02a`) metadata — both already share `pathogen`/`source`/`name`/`compounds`/`positives`/`target_type` — and adds `inactives`. No ChEMBL/PubChem overlap flagging: that de-duplication is now handled upstream in the PubChem pipeline's assay-selection criteria.

2. **Compound extraction.** For each dataset, reads `[inchikey, smiles, bin]` from the source (`data/raw/chembl/{pathogen}/{name}.csv.gz` or `data/raw/pubchem/{pathogen}/{name}.csv` — both carry a standard `inchikey`), validates `bin ∈ {0,1}`, and deduplicates at the **InChIKey** level (one row per molecule; active wins on a conflict). Writes `output/07_datasets/{pathogen}/{name}.csv`. Re-extraction overwrites, so the step is idempotent.

3. **Balancing with proven negatives.** Datasets with an active ratio above **0.5** are balanced down to exactly **0.5** by adding *real measured negatives* — compounds inactive (`bin=0`) in another dataset of the same pathogen (pool spans ChEMBL + PubChem). A candidate is excluded if its InChIKey is already in the dataset or is a proven active (`bin=1`) anywhere in the pathogen. If the proven-negative pool is exhausted, the shortfall is topped up with decoys from `output/06_decoys/06_eos3e6s_v1.csv` (loaded lazily). Added rows get `bin=0` and `added_negative=True`.

Saves `output/07_datasets/07_datasets_metadata.csv` with recomputed counts plus `added_negatives`, `added_decoys`, `final_ratio`, `final_compounds`.

**Conflict rule:** when one InChIKey appears as both active and inactive within a dataset, the **active label wins** (`bin=1`).

**Threshold:** balance datasets with ratio > 0.5 down to 0.5 (`HIGH_RATIO_THRESHOLD` in `src/default.py`); decoys are fallback-only. In the current run all 51 imbalanced datasets were balanced with proven negatives alone (0 decoys needed).

---

## 07b_quality_checks.py

Per-dataset InChIKey-deduplication audit of the post-07 datasets, using the `inchikey` column carried by each file (no RDKit recompute; `source` read from the metadata). For each dataset, reports total rows, rows without an InChIKey, unique compounds, duplicate compounds (and how many duplicate via >1 distinct SMILES), label conflicts (same InChIKey with both `bin=0` and `bin=1`), added-negative count, `added_neg_shared`/`added_neg_shared_frac` (added negatives also added to another dataset of the same pathogen), and added-negative/active collisions across other datasets of the same pathogen (should be 0). Prints a clean/non-clean summary plus a per-pathogen **added-negative reuse** table (reuse factor and % reused — the same proven negative can land in several datasets since the pool is shared per pathogen). Output: `output/07_datasets/07_dup_report.csv`.

---

## 07c_plot_datasets.py

Four-panel figure from the post-balancing metadata: datasets per pathogen (stacked — solid = as-is, white fill / colored edge = balanced with added negatives), final active ratio per dataset (jittered scatter; balanced datasets are hollow and sit at the 0.5 reference line), total negatives added per pathogen (log-scaled), and **negative reuse across datasets** (% of a pathogen's added negatives that were added to more than one dataset — surfaces shared-pool duplication). Output: `output/07_datasets/07_datasets.png`.

---

## 08_download_weights.py *(HPC only)*

Downloads the LazyQSAR descriptor model weights needed by step 09. Run once from the login node before submitting the SLURM array job. Re-running is safe — each file is skipped if already present. Weights are saved to `output/08_weights/` and include chemeleon, cddd (encoder + FPSim index), and CLAMP.

---

## 09_run_models.py / 09_run_models.sh *(HPC only)*

Trains a LazyQSAR model (`slow` mode: cddd, chemeleon, clamp, morgan, rdkit) for one dataset per SLURM array task. Each task reads `output/07_datasets/{pathogen}/{name}.csv` (`smiles`, `bin`), runs 5-fold stratified cross-validation, and saves per-fold metrics (AUROC, AUPRC, BEDROC and baselines, OOF AUCs per descriptor, raw score arrays) to `output/09_reports/{pathogen}/{model_name}.csv` + a `_folds.json`. Then trains a final model on all data and saves it to `output/09_models/{pathogen}/{model_name}/`. The **model name is the dataset `name`** (unique per pathogen — ChEMBL pool ids / PubChem dataset ids), so report and model files match the dataset file on disk.

Datasets whose minority class is smaller than `N_FOLDS` (too few actives/inactives, or all-active/all-inactive) are **skipped** as untrainable, with a message.

**Local alternative:** `09_fit_models_local.py` — runs all datasets sequentially. Same report CSV + `_folds.json` per dataset. Accepts `--pathogens` to restrict to a subset. Existing outputs are skipped, so it is safe to re-run after interruption.

---

## 10a_aggregate_reports.py

Reads all per-dataset CV reports from `output/09_reports/` (keyed by dataset `name` — no positional recomputation) and collapses them into `output/10_reports/`. Applies a hard filter: datasets with mean CV AUROC < `MIN_AUROC` are excluded and recorded in `10_discarded_models.csv`. Retained datasets are written to `10_reports.csv`, one row per dataset.

`10_discarded_models.csv` also reports, per discarded dataset, a compound-overlap breakdown against that pathogen's accepted datasets: `compounds` (total), `actives` (bin=1 count), `lost` (present nowhere else), `%_lost` (lost as a fraction of the accepted chemical space's own size), `ambiguous` (accepted datasets already disagree amongst themselves), `concordant` (accepted label agrees with the discarded dataset's own label), `conflict` (accepted label disagrees) — with `lost + ambiguous + concordant + conflict == compounds`.

Beyond aggregated metrics (mean/std of AUROC, AUPRC, BEDROC), each dataset gets a composite quality weight from six 0–1 components: **w1** real-negative fraction = 1 − (added_negatives + added_decoys)/n_negatives (penalises negatives borrowed from other assays / decoy fallback), **w2** mean CV AUROC, **w3** AUPRC enrichment over prevalence, **w4** BEDROC enrichment over random, **w5** total compound count, **w6** active count. (The old flat dataset-type weight was removed and the rest renumbered.) All components are guarded against NaN/inf (baseline clamped away from 0/1, zero-negative and zero-sum guards). `final_weight` is their mean; `final_normalized_weight` rescales within each pathogen to sum to 100. A per-compound seventh weight (**w7**, the decision-cutoff ramp) is added only in the consensus (steps 12b/14).

**Threshold:** `MIN_AUROC = 0.7` (from `src/default.py`).

---

## 10b_training_results.py

Per-pathogen four-panel figure: AUROC bars with cross-fold std error, out-of-fold rank-score distributions (jittered scatter + boxplot) for actives vs inactives with the `decision_cutoff_rank` overlaid, training-set composition (actives / original inactives / added negatives, log scale), and final-weight bars. Datasets balanced with added negatives render with a white bar face in the AUROC and weight panels. Accepts `--pathogen <code>` (single) or iterates all pathogens in `10_reports.csv`. Output: `output/10_reports/plots/10_training_{pathogen}.png`.

---

## 11_download_drugbank.py

Downloads DrugBank SMILES from a public GitHub mirror, validates them with RDKit, drops inorganic molecules (no carbon) and entries above the molecular-weight cap, and writes a single canonical-SMILES column sorted alphabetically to `data/processed/11_drugbank_smiles.csv`.

**Threshold:** `MW_CAP = 1000 Da`.

---

## 12a_predict_drugbank.py / 12a_run_array.sh *(HPC only)*

Predicts DrugBank scores for one (pathogen, predict type) pair per SLURM array task — 90 tasks total (15 pathogens × 6 `PREDICT_TYPES`: `rank`, `proba`, `score`, `logit`, `lift`, `binary`). `task_id` maps to the pair via `divmod(task_id, 6)` over the fixed `PATHOGENS` list in `src/default.py`. Points LazyQSAR at the project `output/08_weights/` directory (via a `HOME` override). Output: `output/12_drugbank/{type}/{pathogen}.csv`, with `smiles` + one column per sub-model. Skips instantly if its target CSV already exists, so it can run safely alongside a concurrent local run. Submit via `sbatch --chdir=<repo_root> --array=0-89%20 scripts/12a_run_array.sh` (requires `output/12_logs/` to already exist — created by script 11).

**Local alternative:** `12a_predict_drugbank_local.py` — runs every predict type in series in one process, sharing descriptors across all models within each type (cheaper per-model, but no cluster parallelism). Accepts `--pathogen <code>` (single output) or `--all_pathogens` (one descriptor pass per type, scoring every pathogen's models at once, then split per pathogen).

**Descriptor note:** the lazy-qsar multi-model `predict()` API computes descriptors once per featurizer *within* a call but discards them afterwards, so descriptors are recomputed once per predict type (N types ⇒ N× descriptor cost) in the local script, and once per (pathogen, type) task in the array version. The `_ensemble_cache` from lazy-qsar issue #26 is a separate single-model code path not used here.

---

## 12b_fit_transformation.py

For each pathogen, reads the per-model **rank** predictions from `output/12_drugbank/rank/{pathogen}.csv` (asserts they are on the [0,1] scale). Before any weighting or fitting, computes the pairwise Pearson correlation between the pathogen's models (folded in from the former script 20) as a diagnostic for how much averaging actually reduces variance. Then solves the tanh steepness `k*` such that the transformed consensus IQR matches the average per-model IQR — once for the full M-model consensus, and once per leave-one-out exclusion (`k_star_loo`, targeting the average IQR of the remaining M-1 models). There is **no meta-curve / global fit** across pathogens: each pathogen, and each of its LOO exclusions, keeps its own directly-solved `k*`.

If the target IQR is unreachable for a given consensus (same-side IQR ceiling below target — see the `_solve_k_star` docstring), the peak-achievable `k` is used instead of failing outright, since it is provably the closest approximation possible; this is flagged via a `*_exact` boolean rather than silently passed off as an exact match. Every pathogen and every one of its LOO exclusions therefore always gets a usable k — nothing is omitted or left null.

Outputs to `output/12_drugbank/`:
- `12b_fit_transformation.csv` — one row per pathogen: `M`, `avg_model_iqr`, `consensus_iqr`, `k_star`, `k_star_exact`, `mean_pairwise_corr`, `median_pairwise_corr`.
- `12b_k_star_loo.csv` — one row per `(pathogen, excluded_model)`: `avg_model_iqr_loo`, `consensus_iqr_loo`, `k_star_loo`, `k_star_loo_exact`.
- `12b_k_star.json` — `{pathogen: {k_star, k_star_exact, M, k_star_loo: {model: k}, k_star_loo_exact: {model: bool}}}`, consumed by scripts 14 and 18.
- `12b_fit_transformation.png` — (i) tanh curve for the min-`k*` and max-`k*` pathogens, (ii) `k*` vs `M` scatter, pathogen-labeled (no fitted line).
- `12b_pairwise_correlation_distributions.png` — KDE of pairwise correlations, one curve per pathogen (formerly script 20's output; script 20 has been removed).

---

## 13_predict_drugbank_ersilia.sh

Runs a single Ersilia Hub model on the DrugBank SMILES file (`data/processed/11_drugbank_smiles.csv`) and writes predictions to `output/13_drugbank_ersilia/<model_id>.csv`. Accepts a model ID and an optional batch size argument (default 100). Must be run in an `ersilia` conda environment — not `camm` — because ersilia and lazyqsar have conflicting numpy requirements.

---

## 14_consensus_scoring.py

Computes a weighted consensus score per DrugBank compound for each pathogen. Reads per-model prob_rank predictions from `output/12_drugbank/rank/{pathogen}.csv` (asserts [0,1] scale) and quality weights (w1–w6) from `output/10_reports/10_reports.csv`. A per-compound, per-model seventh weight (w7) linearly rewards predictions above each model's decision cutoff. The weighted mean prob_rank is passed through a tanh transformation that restores IQR compression caused by averaging; the steepness `k_star` (full consensus) and `k_star_loo` (per leave-one-out exclusion) are loaded directly per pathogen from `output/12_drugbank/12b_k_star.json` — no shared meta-curve across pathogens. A `k_star`/`k_star_loo` that isn't an exact IQR match (`*_exact = False` — the target was unreachable, see 12b) is still applied and used as-is, since it's the closest achievable approximation; this is only printed, not treated as a failure. Only if the whole pathogen is missing from `12b_k_star.json` are its outputs left untransformed. Outputs weighted and unweighted variants (with and without the tanh transform) to `output/14_consensus/`. Accepts `--pathogen <code>` for a single pathogen.

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

Generates a data and model quality dashboard per pathogen. For each pathogen it produces four files under `output/17_quality_checks/{pathogen}/`: `all_smiles_no_added.csv` and `all_smiles_with_added.csv` (unique InChIKeys with label-conflict, added-negative/inactive-overlap, and DrugBank-overlap flags), `data_summary.csv` (one row per dataset with compound counts, `n_added_negatives`/`n_added_decoys`, and per-dataset conflict/DrugBank counts), and `model_summary.csv` (one row per model with AUROC, weight, and fold-stability flags, including discarded models). A top-level `summary.csv` with one row per pathogen is also written.

**Thresholds:** `FOLD_UNSTABLE_AUROC_STD = 0.05` and `LOW_WEIGHT_THRESHOLD = 0.3` (both in `src/default.py`).

---

## 18a_clone_hub_repos.py

Clones every pathogen's Ersilia Hub repo (`ersilia-os/{eosXXXX}`, from `ERSILIA_MODEL_IDS` in `src/default.py`) into `chembl-models-tmp/{eosXXXX}/`, one per pathogen. Skips any that already exist (safe to re-run). Prerequisite for both `18b_update_ersilia_model.py` and `19_apply_and_fetch.py`, which both take `--repo-dir` pointing at one of these clones.

---

## 18b_update_ersilia_model.py

Refreshes an already-incorporated Ersilia Hub model with newly trained checkpoints. Per pathogen, reads `output/10_reports/10_reports.csv`, `output/07_datasets/07_datasets_metadata.csv`, `output/09_models/{pathogen}/`, and `output/12_drugbank/12b_k_star.json`; writes a complete refresh package to `output/18_emh_files/{pathogen}/`:

- `reports.csv` — quality report, filtered to `auroc_mean >= MIN_AUROC` and sorted by `(source, label, n_compounds desc)`. `model_name` values are lowercased for Ersilia's schema requirements.
- `run_columns.csv` — `consensus_score` row + one row per kept sub-model (also lowercased). Descriptions describe the dataset family (ChEMBL DR/SP signal-based pool or low-data catch-all, or PubChem whole-cell organism screen), its constituent-assay count, compound count, and any added negatives — no per-assay concentration cutoff (pool `cutoff` is a Youden score).
- `consensus.py` — canonical family template with the pathogen's fixed `k_star` baked in.
- `metadata.yml` — patched in place from the clone's existing file: `Output Dimension`, `Deployment` (block-style `- Local` only), `Description`, and `Interpretation` (both drafted/regenerated — reviewed and edited per pathogen before push, not final copy).
- `install.yml` — pins `ersilia-pack-utils` and `lazyqsar` versions and the per-pathogen descriptor `--only` list.
- `run_output.csv` — generated by running `bash run.sh` in a temp staging dir built from the clone + new artifacts.
- `DIFF_SUMMARY.txt` — sub-model set diff, per-sub-model `decision_cutoff_rank` drift, consensus threshold old→new, descriptor set old→new.
- `COPY_INSTRUCTIONS.txt` — paste-ready `rsync`/`cp` commands to drop the package into the clone, plus the commit/push lines.

Requires `--pathogen <p>` and `--repo-dir <path-to-clone>`. After the script, the user (with Claude) reviews `DIFF_SUMMARY.txt`, runs the copy commands, `git diff` reviews, and pushes direct to `main` on `ersilia-os/{eosXXXX}` (no PR).

**Consensus threshold (faithful):** computed by applying `consensus.py`'s W1-W6 (quality) + W7 (per-compound ramp) formula to the per-sub-model `decision_cutoff_rank` values (W7 = 0 at the boundary by construction), then the tanh transform with the pathogen's fixed `k_star` from `output/12_drugbank/12b_k_star.json`. The same `k_star` is baked into the shipped `consensus.py`, so the recommended threshold and the production transform always agree.

**Lowercase sub-model names:** Ersilia's schema requires lowercase output column names, so `reports.csv`'s `model_name` and `run_columns.csv`'s `name` are lowercased here even though the on-disk checkpoint dirs under `output/09_models/{pathogen}/` (and the local staging test's copies of them) keep their original training-pipeline casing (e.g. `DR_0001`).

---

## 19_apply_and_fetch.py

Applies the refresh package produced by `18b_update_ersilia_model.py` to a local clone, then runs `ersilia fetch --from_dir` on it as a structural verification. Per pathogen:

1. `rsync -a --delete` checkpoints from `output/09_models/{pathogen}/` into `{repo-dir}/model/checkpoints/models/`, matching each (lowercased) kept sub-model name from `reports.csv` back to its actual (originally-cased) source directory case-insensitively.
2. Copies the 6 generated artifacts (`reports.csv`, `run_columns.csv`, `consensus.py`, `metadata.yml`, `install.yml`, `run_output.csv`) from `output/18_emh_files/{pathogen}/` into the right paths under `{repo-dir}/`.
3. `ersilia delete {eosXXXX}` — clears any cached install.
4. `ersilia fetch {eosXXXX} --from_dir {repo-dir}` — validates `metadata.yml` schema, runs `install.yml` end-to-end, and executes `run.sh` once. Fails loud on non-zero exit.

Pass `--no-fetch` to stop after steps 1-2 and skip the `ersilia delete`/`ersilia fetch` validation (also skips the `CONDA_SH` requirement).

Requires `--pathogen <p>` and `--repo-dir <path-to-clone>`. After this, the user (with Claude) reviews `cd {repo-dir} && git diff`, commits, and pushes direct to `main` on `ersilia-os/{eosXXXX}` — one pathogen at a time, with an explicit go-ahead before each push.

---

## 20_cut_releases.py

Cuts the next GitHub release (`vN+1.0.0`) for a pathogen's Hub repo — a separate, deliberate step run only after that pathogen's post-push CI has actually succeeded (never bundled into the push itself). Per `--pathogen`:

1. Checks the most recent `main`-branch workflow run (`upload-model.yml`) via `gh run list` — exits loudly if it doesn't match the commit just pushed, or hasn't completed successfully.
2. Reads the current latest release tag via `gh release list` and bumps it by one (`v2.0.0 -> v3.0.0`; per-pathogen, since pathogens are at different versions — 13 at v2, abaumannii and efaecium at v3).
3. Creates the new release via `gh release create`, which in turn triggers `retag-release-docker.yml` (republishes the model's Docker image) — the only thing that actually does that.
