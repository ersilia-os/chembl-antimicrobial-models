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

## 08_download_weights_and_reference.py *(HPC only)*

Downloads everything step 09 needs from the network, in two halves, then prints the `sbatch` command to submit it. Run once from the login node. Re-running is safe — cached files are not fetched again, though the reference-library checks always re-run.

1. **Descriptor weights** (~610 MB) — chemeleon, cddd (encoder + FPSim index + SMILES list) and CLAMP, saved to `output/08_weights/.lazyqsar/`. Downloaded by LazyQSAR itself, so the URLs live in one place rather than being copied here, and every file is verified against a published sha256 — including files already cached from an earlier run.
2. **Reference library** (~260 MB) — the fixed 50,000-molecule drug-like library that LazyQSAR ≥ 3.6 ranks against, saved to `output/08_weights/.lazyqsar/reference/`. Fetching it here once means the step-09 array tasks can run with `LAZYQSAR_REFERENCE_OFFLINE=1` instead of all racing into the same cache. Verified on arrival: structural check, sha256 against the published manifest, and a descriptor-drift canary that recomputes 16 molecules with the local install. Any failure exits non-zero.

The weights must come first: the drift canary in (2) recomputes descriptors using the checkpoints from (1).

**Dataset size split:** the printed `sbatch` commands separate datasets at **30,000 compounds** — below that, 16 GB of memory and 20 concurrent tasks; above, 64 GB and 5. The threshold reflects observed memory use during featurization, not a property of the data.

---

## 09_run_models.py / 09_run_models.sh *(HPC only)*

Trains a LazyQSAR model (`slow` mode: cddd, chemeleon, clamp, morgan, rdkit) for one dataset per SLURM array task. Each task reads `output/07_datasets/{pathogen}/{name}.csv` (`smiles`, `bin`), runs 5-fold stratified cross-validation, and saves per-fold metrics (AUROC, AUPRC, BEDROC and baselines, OOF AUCs per descriptor, raw score arrays) to `output/09_reports/{pathogen}/{model_name}.csv` + a `_folds.json`. Then trains a final model on all data and saves it to `output/09_models/{pathogen}/{model_name}/`. The **model name is the dataset `name`** (unique per pathogen — ChEMBL pool ids / PubChem dataset ids), so report and model files match the dataset file on disk.

**LazyQSAR version:** 3.6.1 (`environment.yml`). 3.6.1 is 3.6.0 plus one change, the fit-time memory fix (lazy-qsar PR #48, merged 2026-10-05): 3.6.0 keeps an onnxruntime session arena per preprocessor, so the memory of a fit grows with the number of batches times the rows, and the largest datasets exceed 100 GB. The fix changes no computed value. Provenance of the models in `output/09_models/`: 188 of the 196 were fit with the released 3.6.0, and the last 8 (calbicans 1242 and 485275_merged3, ecoli 1053175, mtuberculosis 449762_merged3, pfalciparum 743093_merged2, 504832 and 504834, saureus 1259309_merged3) were fit after that file was patched into the environment, which is the code released as 3.6.1, with 64 GB or 96 GB of memory requested instead of the script's default. All the scaffold-split reports of step 09b were also produced with the patched code. That the two give identical results was checked on one dataset (pfalciparum/SP_0006: predictions of all six types, cutoffs, out-of-fold AUCs, rank knots, `screening_auc` and ONNX weights identical to the released 3.6.0), and the installed package is byte-identical to the 3.6.1 sources.

Datasets whose minority class is smaller than `N_FOLDS` (too few actives/inactives, or all-active/all-inactive) are **skipped** as untrainable, with a message.

**Local alternative:** `09_fit_models_local.py` — runs all datasets sequentially. Same report CSV + `_folds.json` per dataset. Accepts `--pathogens` to restrict to a subset. Existing outputs are skipped, so it is safe to re-run after interruption.

---

## 09b_run_models_scaffold.py / 09b_run_models_scaffold.sh *(HPC only)*

Companion to step 09: same 5-fold CV evaluation, same array indexing over `07_datasets/07_datasets_metadata.csv` (one task per dataset, same trainability skip rule), but folds are assigned by **scaffold** instead of at random, and **no final model is trained or saved** — that model already exists from step 09 and would be identical here. Purely a diagnostic report, not consumed by any downstream script (10a onward untouched).

**Key decisions:**
- **Scaffold:** standard (non-generic) Bemis-Murcko scaffold via RDKit `MurckoScaffoldSmiles` (`includeChirality=False`). Acyclic compounds all share the empty-string scaffold and are grouped together — the standard convention for this function.
- **Fold assembly:** `sklearn.model_selection.StratifiedGroupKFold` (`N_FOLDS=5`, `RANDOM_SEED=42`) — scaffold groups are never split across a train/test boundary, with best-effort class balancing per fold.
- **Degenerate folds:** a scaffold-constrained fold can end up with a single-class test set even when the dataset passes the min-class-size trainability check. When that happens, only that fold is skipped (no model fit, since AUROC/AUPRC/BEDROC would be undefined): its report row keeps compound/positive counts and `baseline_auroc=0.5`, all other metric columns are `NaN`, and it is omitted from `_folds.json`.

Output: `output/09b_reports/{pathogen}/{name}.csv` + `_folds.json`, same column layout as `output/09_reports/` plus two extra columns, `scaffolds_train`/`scaffolds_test`, giving the number of distinct scaffold groups in each split (always defined, including for skipped folds). `output/09b_logs/` must exist before `sbatch` submission (not created automatically).

---

## 09c_plot_scaffold_vs_random.py

Diagnostic scatter comparing the two CV strategies: for every dataset with a completed report in both `output/09_reports/` (random split) and `output/09b_reports/` (scaffold split), computes `delta_auroc = auroc_random_mean - auroc_scaffold_mean` per model and plots it against pathogen (x-jittered, one dot per model, dashed reference line at 0). A positive delta means the random split was optimistic relative to the scaffold-grouped one. Datasets missing a report on either side (e.g. still pending in 09b's array job) are excluded and counted per pathogen in the console output rather than guessed. Output: `output/09c_scaffold_vs_random/09c_delta_auroc.csv` + `09c_scaffold_vs_random.png`.

---

## 10a_aggregate_reports.py

Reads all per-dataset CV reports from `output/09_reports/` (keyed by dataset `name` — no positional recomputation) and collapses them into `output/10_reports/`. Applies a hard filter: datasets with mean CV AUROC < `MIN_AUROC` are excluded and recorded in `10_discarded_models.csv`, together with the untrainable datasets (a class with fewer than `N_FOLDS` members, which step 09 skips, so they have no report). A trainable dataset with no report, a report whose folds are not exactly 0 to `N_FOLDS`−1, or a NaN mean AUROC stops the script with an error instead of becoming a silent discard, so an unfinished or failed training run cannot drop a model out of the consensus unnoticed. Retained datasets are written to `10_reports.csv`, one row per dataset.

`10_discarded_models.csv` also reports, per discarded dataset, a compound-overlap breakdown against that pathogen's accepted datasets: `compounds` (total), `actives` (bin=1 count), `lost` (present nowhere else), `%_lost` (lost as a fraction of the accepted chemical space's own size), `ambiguous` (accepted datasets already disagree amongst themselves), `concordant` (accepted label agrees with the discarded dataset's own label), `conflict` (accepted label disagrees) — with `lost + ambiguous + concordant + conflict == compounds`.

Beyond aggregated metrics (mean/std of AUROC, AUPRC, BEDROC), each dataset gets a composite quality weight from seven 0–1 components: **w1** real-negative fraction = 1 − (added_negatives + added_decoys)/n_negatives (penalises negatives borrowed from other assays / decoy fallback), **w2** mean CV AUROC, **w3** AUPRC enrichment over prevalence, **w4** BEDROC enrichment over random, **w5** total compound count, **w6** active count, **w_screen** screening AUC (0 at ≤0.7, linear to 1 at 1.0 — the same shape as w2; see below). (The old flat dataset-type weight was removed and the rest renumbered.) The baselines are clamped away from 0 and 1 so a degenerate prevalence cannot give inf/NaN, and the zero-negative and zero-sum cases are guarded. `final_weight` is their mean; `final_normalized_weight` rescales within each pathogen to sum to 100. A per-compound weight (**w7**, the decision-cutoff ramp) is added only in the consensus (step 14); it is unrelated to **w_screen**, which is a model-level weight and is named differently on purpose.

`10_reports.csv` also carries two out-of-fold diagnostics read from each model's `metadata.json` (LazyQSAR >= 3.6): `screening_auc` (probability that an active ranks above a molecule of the fixed 50K drug-like reference library) and `sensitivity_at_cutoff` (fraction of actives at or above the fixed 0.65 decision cutoff). `screening_auc` feeds **w_screen**; `sensitivity_at_cutoff` is reported only. A model without `screening_auc` (for example a pre-3.6 checkpoint, or a dataset whose final fit has not finished) stops the script with an error instead of getting a blank or zero weight.

**Thresholds** (all in `src/default.py`): `MIN_AUROC = 0.7`, `W_AUROC_FLOOR = 0.7` (w2 is 0 at or below it) and `W_SCREEN_FLOOR = 0.7`. `w_screen` is 0 at a screening AUC of `W_SCREEN_FLOOR` or below and rises linearly to 1 at 1.0, mirroring w2; it enters the mean as an equal component. Chosen over a floor of 0.5 (chance) or 0.6 because it separates models more: across the 193 retained models the median `w_screen` is 0.63 and 9 models are at 0.

**Decision on w3 and w4 (2026-10-05, kept as is):** the fold-enrichment half of each is capped by the baseline, because `value / baseline` cannot exceed `1 / baseline`. At prevalence 0.5 the cap is 2x, which limits that half to about 0.06, against 0.5 at prevalence 0.1. Datasets balanced with added negatives are therefore down-weighted three times over, by `w1`, `w3` and `w4`: among the 54 balanced models `w3` reaches at most 0.54 and `w4` 0.56, against 0.93 and 0.98 for the 139 others, and the median `final_weight` is 0.52 against 0.58. The penalty was reviewed and kept deliberately. Two related notes: `w5` counts the added negatives as compounds (so it partly offsets `w1`), and the overlap breakdown in `10_discarded_models.csv` counts the added negatives of accepted datasets as part of the accepted chemical space.

---

## 10b_training_results.py

Per-pathogen four-panel figure: 5-fold CV AUROC bars with cross-fold std error and one cross per dataset for its `screening_auc` (how well the model separates its actives from the reference library of drug-like chemistry, as opposed to from the dataset's own inactives); out-of-fold rank-score distributions (jittered scatter + boxplot) for actives vs inactives with the `decision_cutoff_rank` overlaid; training-set composition (actives / original inactives / added negatives, log scale); and final-weight bars. Datasets balanced with added negatives render in a lighter tint in the AUROC and weight panels, and their added-negatives bars are lighter in the composition panel. The AUROC axis starts one 0.05 step below `MIN_AUROC` (0.65) and extends further down only when a screening-AUC cross would otherwise be cut off; the weight axis always starts at 0, so bar heights are proportional to the weights. Each panel's legend sits in a band just above the data, sized to the legend (about 13% of the panel) (the axis ticks stop at the real range, so the band does not read as part of the scale). The figure uses stylia's default width and a taller height, since four stacked panels need it; this and the small scatter marker (thousands of points per dataset) are documented exceptions to stylia's default sizes. Accepts `--pathogen <code>` (single) or iterates all pathogens in `10_reports.csv`. Output: `output/10_reports/plots/10_training_{pathogen}.png`.

---

## 11_download_drugbank.py

Downloads DrugBank SMILES from a public GitHub mirror, validates them with RDKit, drops inorganic molecules (no carbon) and entries above the molecular-weight cap, and writes a single `smiles` column to `data/processed/11_drugbank_smiles.csv`, sorted alphabetically with identical strings removed (11,347 molecules). The SMILES are kept as they are in the source: they are not re-canonicalised, and salts and mixtures stay whole (343 entries have more than one fragment), so the same structure written as two different strings is scored twice (3 of the 11,347 rows share an InChIKey with another row).

**Threshold:** `MW_CAP = 1000 Da`.

---

## 12a_predict_reference.py / 12a_run_array.sh *(HPC only)*

The twin of 12b, and the step that comes first because it defines the scale the DrugBank scores are placed on: same models, same six predict types, same `divmod(task_id, 6)` mapping, same output layout — only the molecules differ. Instead of DrugBank it scores the 50,000 drug-like molecules of the LazyQSAR ≥ 3.6 reference library, the library every model's `rank` is already anchored on. Output: `output/12_reference/{type}/{pathogen}.csv`. Submit via `sbatch --chdir=<repo_root> --array=0-89%20 scripts/12a_run_array.sh`; in practice only the `rank` tasks are needed (`--array=0,6,12,…`, i.e. pathogen index × 6).

**Retired:** the tanh steepness fit that used to be step 12b (`12b_fit_transformation.py`, now in `tmp/`) assumed the middle of the rank scale is 0.5. Under LazyQSAR 3.6 a typical molecule scores about 0.26 per model, the transform can no longer reach its target spread, and it pushes ~99% of DrugBank compounds down. The replacement for the consensus scale is being decided; step 14 now anchors the consensus on the reference library instead (see its entry), and step 18b still reads `12b_k_star.json` until it is rewritten (plan: `docs/2026-10-02_consensus_reference_anchoring_plan.md`).

**Why:** a sub-model's rank is a *position* against that library, so its 0.65 cutoff means "beats 99% of drug-like chemistry" and admits about 1% of it: 0.98% to 1.21% of the library per model once it is re-scored at inference, with 178 of the 193 models within 2% (relative) of 1%. The small deviation probably comes from the bundle storing its descriptors as float16 while inference featurizes in float32 (a hypothesis, not tested). It does not affect the consensus, which step 14 re-anchors so that exactly 1% of the library reaches 0.65. The consensus is a weighted *mean* of several such positions, which is not itself a position against anything — with weakly-correlated models, far less than 1% of generic chemistry reaches a mean of 0.65 (measured on DrugBank: ~2% of compounds per sub-model, but 1/11,347 and 10/11,347 for the abaumannii and kpneumoniae consensus). Scoring the library itself gives the consensus the same reference distribution its sub-models have, so a consensus cutoff can be defined by the generic hit rate it admits rather than by transforming 0.65 through the tanh.

**Molecule list:** read from the cached bundle through lazyqsar's own `reference.identity`, never from a path spelled out in the script, so it follows the bundle the checkpoints were fitted against (`manifest_sha256`). Script 08 fetches it; this script never downloads (`LAZYQSAR_REFERENCE_OFFLINE=1`), so a compute node without network fails loudly instead of silently diverging.

**Resources:** 32 GB and a 12 h limit, up from 12b's 16 GB / 6 h, for 4.4× the molecules (50,000 vs 11,347). Requires `output/12a_logs/` to already exist.

**Reproducibility and provenance:** the same applies as for 12b (see its note below). In 12a the types of a pathogen are not bit-consistent: 89 of 193 models differ somewhere between files, 13,874 of the 38.6 million compared values by more than 1e-4 (mostly mtuberculosis rank, up to 0.088, and calbicans `lift` and `logit`), and `binary` disagrees with "rank ≥ 0.65" for 11 molecule-model pairs in 4 models. Rebuilding the step-14 consensus from the rank implied by the other types for five pathogens moved the four anchors by at most 2e-5 and the per-molecule raw consensus by at most 0.0066. Seven rank files (tasks 0, 24, 30, 42, 60, 78 and 84: abaumannii, efaecium, enterobacter, kpneumoniae, paeruginosa, smansoni, spneumoniae) were written on 2026-10-01 by an earlier version of the wrapper and skipped on 2026-10-04 because they already existed; their models are unchanged since, and a recompute of one pathogen matches them to 1.5e-6.

---

## 12b_predict_drugbank.py / 12b_run_array.sh *(HPC only)*

Predicts DrugBank scores, the set the consensus is applied to (12a comes first because it defines the scale they are placed on), for one (pathogen, predict type) pair per SLURM array task — 90 tasks total (15 pathogens × 6 `PREDICT_TYPES`: `rank`, `proba`, `score`, `logit`, `lift`, `binary`). `task_id` maps to the pair via `divmod(task_id, 6)` over the fixed `PATHOGENS` list in `src/default.py`. Points LazyQSAR at the project `output/08_weights/` directory (via a `HOME` override). Output: `output/12_drugbank/{type}/{pathogen}.csv`, with `smiles` + one column per sub-model. Skips instantly if its target CSV already exists, so it can run safely alongside a concurrent local run. Submit via `sbatch --chdir=<repo_root> --array=0-89%20 scripts/12b_run_array.sh` (requires `output/12b_logs/` to already exist — created by script 11).

**Local alternative:** moved to `tmp/12b_predict_drugbank_local.py` (no longer part of the pipeline). It runs every predict type in series in one process, sharing descriptors across all models within each type (cheaper per-model, but no cluster parallelism), via `--pathogen <code>` or `--all_pathogens`.

**Descriptor note:** the lazy-qsar multi-model `predict()` API computes descriptors once per featurizer *within* a call but discards them afterwards, so descriptors are recomputed once per predict type (N types ⇒ N× descriptor cost) in the local script, and once per (pathogen, type) task in the array version. The `_ensemble_cache` from lazy-qsar issue #26 is a separate single-model code path not used here.

**Reproducibility and provenance (12b, and by extension 12a):** each (pathogen, type) is a separate task that recomputes descriptors and predictions on whichever of the three allowed nodes it lands on. Different CPU types give inputs that differ at about 1e-7, and the tree heads amplify this for a few molecules, so the six types of a pathogen are not bit-consistent with each other. A fresh run on one machine is bit-reproducible, and 1 versus 4 threads give identical results, so the variation comes from the node, not from randomness. In 12b the rank file differs from the rank implied by the probability file in 63 of 193 models (0.10% of the 2.19 million molecule-model pairs; 19 pairs by more than 0.05, at most 0.10), `logit` differs from the probability for 31 models, `lift` for 85 and `score` for 9, and `binary` disagrees with "rank ≥ 0.65" in 5 pairs (mtuberculosis DR_0006 and DR_0012, pfalciparum DR_0011). So the types must not be read as views of one computation: do not rely on rank, `proba`, `logit` and `lift` ordering molecules identically, and quote this tolerance. Only the `rank` files are read by steps 14 to 16b, and each of those is one consistent run. A rerun does not reproduce the files bit for bit. Computing all six types in one `predict_tasks` call per pathogen would make them consistent (15 tasks instead of 90); this was judged not worth regenerating. Seven rank files (the same tasks as in 12a) were written on 2026-10-01 by an earlier version of the wrapper and skipped on 2026-10-04; they were checked and are valid.

**Write safety:** a task writes its output in place and skips existing files, so a task killed mid-write (spot preemption) would leave a truncated file that a rerun accepts; after a kill, delete that task's file first. All 180 files of 12a and 12b were checked complete (row counts, `smiles` column, one finish time each).

**DrugBank molecules without descriptors:** 337 of the 11,347 compounds (3%), organometallics and metalloids (Bi, Os, Fe, As, Hg, Se), have all-NaN RDKit descriptors. The preprocessor imputes the median, so the 63 models that use RDKit score them from imputed values, and nothing in the output flags them; read their hits with care.

---

## 12c_plot_organism_summary.py

Per-pathogen two-panel figure comparing the score distributions of the training folds with DrugBank and the LazyQSAR reference library, one slot per model along x: out-of-fold actives (red), out-of-fold inactives including added negatives (blue), the DrugBank compounds (yellow) and the 50,000 reference-library molecules (green), each as a boxplot with jittered points (box from the 25th to the 75th percentile with a line at the median, whiskers at the 5th and 95th percentiles, a purple (orchid) circle at the 90th and a fuchsia circle at the 99th). On the rank scale the reference group's 90th and 99th percentiles fall on the 0.50 and 0.65 anchors by construction, so the markers show at a glance where the other groups sit against them. The legend gives the number of DrugBank and reference compounds scored and explains the box, whiskers and markers. Panel (a) shows predicted probabilities and panel (b) rank scores. Above each group the label is the share of its compounds at or above the cutoff, with the comparison `>=` that LazyQSAR's `binary` output uses. Output: `output/12c_organism_summary/12c_{pathogen}.png`; accepts `--pathogen <code>`, otherwise plots every pathogen in `output/12_drugbank/rank`. Inputs: the per-fold arrays of step 09 (`{name}_folds.json`, all five folds, so every compound is scored out of fold once), the models' `decision_cutoff_*`, the DrugBank and reference rank and probability files (`output/12_drugbank`, `output/12_reference`) and `config/pathogens.csv`.

**Decision (2026-10-05): in panel (a) only DrugBank has a cutoff.** The DrugBank probabilities come from the final model, so its `decision_cutoff_proba` is exact for them. The out-of-fold probabilities come from five fold models, and the same rank 0.65 sits at a different probability in each (they differ by a median of 0.05, up to 0.28), so no single probability cutoff fits them: a first version applied the final model's, and for 34 of 193 models the held-out percentage then differed from panel (b) by more than 5 points (saureus SP_0004: 19% against 58%). Panel (a) therefore shows no line or label for the held-out groups; read them in panel (b), where every model is cut at the same rank 0.65. DrugBank and the reference library are both scored by the final model, so both keep the exact cutoff in panel (a). The fold models' own probability cutoffs are not saved; they can be recovered as the smallest probability among a fold's compounds ranked at 0.65 or higher, because the rank is monotone in the probability.

**The green group is a calibration reference, not a test:** every model's rank is a position against the reference library, so about 1% of it reaches rank 0.65 by design (0.976% to 1.212% per model, median 0.996%; the labels round to one decimal, so 184 of 193 models read "1.0%"). Compare the DrugBank hit rate with it, but do not read it as an independent result.

**Notes:** the out-of-fold percentage in panel (b) is the outer 5-fold estimate, which is not the `sensitivity_at_cutoff` of `10_reports.csv` (LazyQSAR's internal out-of-fold estimate): they differ by more than 5 points for 21 of 193 models, at most 25 (kpneumoniae SP_0004: 16% against 41.5%), so do not quote one as the other. The number in the title counts the retained models (pfalciparum shows 51 models out of 54 datasets, because 3 were discarded). For pfalciparum and mtuberculosis the labels are too dense to read at the default figure size, and for models whose probabilities are tiny (calbicans 1242) panel (a) is squeezed against zero.

---

## 13_predict_drugbank_ersilia.sh

Runs a single Ersilia Hub model on the DrugBank SMILES file (`data/processed/11_drugbank_smiles.csv`) and writes predictions to `output/13_drugbank_ersilia/<model_id>.csv`. Accepts a model ID and an optional batch size argument (default 100). Must be run in an `ersilia` conda environment — not `camm` — because ersilia and lazyqsar have conflicting numpy requirements.

---

## 14_consensus_scoring.py

Computes a consensus score per DrugBank compound for each pathogen, placed on the same rank scale as a sub-model. A sub-model's rank is a position against the 50,000 drug-like molecules of the LazyQSAR reference library (rank 0.65 = "beats 99% of that library"); a weighted *mean* of such positions is not itself a position against anything, so the consensus is anchored on the library the same way. For each pathogen: (1) the consensus of the 50,000 **reference** molecules (`output/12_reference/rank/{pathogen}.csv`) and of the **DrugBank** compounds (`output/12_drugbank/rank/{pathogen}.csv`) is computed with the same function and weights (quality weights `w1`–`w6` and `w_screen` from `output/10_reports/10_reports.csv`, plus the per-compound ramp `w7`: 0 at or below the model's decision cutoff, rising to 1 at rank 1; the eight terms are averaged); (2) the raw consensus at the reference's p50 / p90 / p99 / p99.9 is read off and mapped to 0.25 / 0.50 / 0.65 / 0.75 with LazyQSAR's own piecewise-linear rule (`lazyqsar.utils.ranking`), with a straight line from the last anchor to (raw 1.0 → rank 1.0) above it; (3) each DrugBank consensus is pushed through that map. The map is monotone, so the order of compounds is unchanged, and `consensus_rank >= 0.65` flags about 1% of generic chemistry by construction. Every column has its own anchors because each averages a different set of models: the full consensus and each leave-one-out consensus, weighted and unweighted.

Outputs go in one folder per pathogen, `output/14_consensus/{pathogen}/`: `drugbank_raw.csv`, `drugbank_rank.csv`, `drugbank_unweighted_raw.csv`, `drugbank_unweighted_rank.csv` (columns `consensus_raw` / `consensus_rank` and `consensus_raw_without_{model}` / `consensus_rank_without_{model}`, 6 decimals), the same four files for the 50,000 reference molecules (`reference_*.csv`), and `anchors.json` (the anchor table of every column, the bootstrap standard error of its four anchors — diagnostic only, never changes a score — and a fingerprint of the models, weights and cutoffs it was built with, which step 18b uses to refuse to ship stale anchors). The reference files are the molecules the anchors were built from, so about 1% of `reference_rank` is at or above 0.65 by construction: they show the distribution the scale is defined on, not an independent test. The maths lives in `src/consensus.py` so 18b applies exactly the same code. Pathogens with a single retained model get no consensus. The script exits with code 1 if a pathogen with two or more models is skipped for missing inputs, and raises if the DrugBank and reference files disagree with each other or with `10_reports.csv`. Accepts `--pathogen <code>`.

**Decisions:** the consensus is an arithmetic weighted mean (of the ranks, and of the terms that make up each weight); a geometric mean was considered and not adopted (see the step-14 plan, section 2). The scale is LazyQSAR's, read from it rather than copied (anchors 50/90/99/99.9 → 0.25/0.50/0.65/0.75, decision rank 0.65), and the script stops if a later LazyQSAR release changes it. The top of the scale has no actives anchor (a consensus has no out-of-fold actives), so it follows LazyQSAR's own fallback; only the ~0.5% of compounds above the reference p99.9 depend on this, never the cutoff or the hit counts. Plan and evidence: `docs/2026-10-02_consensus_reference_anchoring_plan.md` and `docs/2026-10-02_step14_refactor_plan.md`.

---

## 14b_plot_consensus_calibration.py

Plots, for each pathogen with a consensus, what the calibration of step 14 does to the full weighted consensus. Panel (a) shows two distributions, each as points with a violin and a boxplot on top, the raw consensus (before calibration) and the consensus rank (after), each for the DrugBank compounds and the 50,000 reference molecules, with a dashed line at the decision rank 0.65 and the % of each group at or above it. Each boxplot is drawn as in 12c (box from the 25th to the 75th percentile with a line at the median, whiskers at the 5th and 95th, a purple (orchid) circle at the 90th and a fuchsia circle at the 99th); after calibration the reference group's 90th and 99th percentiles fall on the 0.50 and 0.65 anchors by construction. Panel (b) is a scatter of raw consensus against consensus rank with the Spearman correlation of each group, which is 1 because the calibration is monotone (a value below 1 would mean the 6-decimal rounding of the step-14 files created ties or the files do not match). It only reads the step-14 files; the leave-one-out and unweighted columns are not drawn. Output goes to `output/14b_consensus_calibration/14b_{pathogen}.png`. Exits with code 1 if a pathogen with two or more models has no step-14 output. Accepts `--pathogen <code>`.

---

## 15_recapitulate_models.py

Quantifies pairwise agreement between individual models for each pathogen, on DrugBank and on the 50,000-molecule reference library. For every ordered model pair (A, B) it reports Spearman and Pearson correlation, the hit overlap (molecules shared by the top-k of A and of B), and the AUROC of A scoring against B binarized at its top-k. Output goes to `output/15_recapitulate_models/{pathogen}/{drugbank,reference}.csv`. Accepts `--pathogen <code>` for a single pathogen (an unknown code is an error). Pathogens with a single retained model are skipped. The script exits with code 1 if a pathogen with two or more models has no rank file, and raises if a rank file does not hold exactly the models `10_reports.csv` lists for that pathogen. The metric code is in `src/recapitulation.py` (shared with 16a).

**Depth:** top 0.1%, 1% and 5% of the dataset, `k = max(1, ceil(t * n))`: 12 / 114 / 568 molecules on DrugBank and 50 / 500 / 2,500 on the reference. A fixed count (the old top 10 / 100 / 500) would mean a different depth on each dataset. The k of every row is stored in the `top_k_*` columns. The depths are chosen to match the AUROC thresholds, so overlap and AUROC can be read at the same depth.

**Two datasets, two questions:** DrugBank (known drugs, enriched in real bioactives) asks whether the models agree where actives may exist; the reference library (generic chemistry, almost all inactive) asks whether they agree across ordinary chemical space. All metrics are rank-based, so none depends on the anchoring of step 14.

---

## 16a_recapitulate_consensus.py

Measures how well the consensus score (weighted and unweighted, from step 14) recapitulates each individual model, on DrugBank and on the reference library. Runs in two modes — leave-one-out (model excluded from the consensus) and full (model included) — producing four CSV files per dataset in `output/16_recapitulate_consensus/{pathogen}/` (`{dataset}_exc_weighted.csv`, `_exc_unweighted`, `_weighted`, `_unweighted`). Metrics and depths match step 15. Accepts `--pathogen <code>` for a single pathogen.

By default it reads the consensus on the LazyQSAR rank scale (`consensus_rank`). `--raw` reads the raw consensus (`consensus_raw`) and adds a `_raw` suffix to the output files. The anchoring is monotone, so the two give identical Spearman, hit overlap and AUROC; only Pearson differs (checked on abaumannii: at most 0.02). The script exits with code 1 if a pathogen with two or more models has no step-14 folder (pathogens with one model are skipped), and raises if step 14 was not built from the step-12 files it reads: SMILES that differ row by row, different columns or models (also in `anchors.json`), a reference rank file whose SHA-256 is not the one recorded in `anchors.json`, or an unweighted consensus that is not the mean of the step-12 ranks. Accepts an unknown `--pathogen` as an error.

**Ties at a cut:** the top-k sets hold exactly k molecules, so when several molecules tie at the k-th score (the consensus is written with 6 decimals) which of them are in the set is arbitrary. That can move a hit overlap by one molecule (seen on DrugBank at 1% and on the reference library at 5%, e.g. 445 vs 446 of 2,500), below the 4-decimal precision of the other metrics; AUROC counts every tied molecule as a positive and is not affected.

---

## 16b_consensus_results.py

Per-pathogen, per-dataset consensus dashboard (one figure for DrugBank, one for the reference library): three full-width rows plus a split final row. [0] `rank` distribution per sub-model, with each model's `decision_cutoff_rank` as a dotted line. [1] Weighted consensus rank, one column per leave-one-out exclusion plus the global consensus ("G."). [2] How well the leave-one-out consensus recapitulates each model. [3] AUROC of the step-15 off-diagonal model pairs, as a histogram and a reversed-cumulative curve. Accepts `--pathogen <code>` (single) or iterates all pathogens in `config/pathogens.csv`. Output: `output/16_recapitulate_consensus/plots/16_consensus_{pathogen}_{drugbank,reference}.png`.

**On the reference library, panels [0] and [1] are a calibration check, not a measurement:** every rank there is a position against that same library, so each distribution is fixed by construction. Panels [2] and [3] are real agreement measurements.

**Safeguards:** the script raises if the step-12 file, the step-14 leave-one-out columns, the step-15 and step-16a tables and `10_reports.csv` do not all list the same models, and exits with code 1 if a pathogen with two or more models has no step-14 output (pathogens with one model are skipped). An unknown `--pathogen` is an error.

**Panel [2] shows only the AUROC:** for each model, one circle per depth (the model's own top 0.1 / 1 / 5% as positives, scored by the leave-one-out consensus). Colour encodes depth and the dashed line is chance (0.5). Spearman and hit overlap are in the step-16a tables but not in the figure.

---

## 17_quality_checks.py

Generates a data and model quality dashboard per pathogen. For each pathogen it produces four files under `output/17_quality_checks/{pathogen}/`: `all_smiles_no_added.csv` and `all_smiles_with_added.csv` (unique InChIKeys with label-conflict, added-negative/inactive-overlap, and DrugBank-overlap flags), `data_summary.csv` (one row per dataset with compound counts, `n_added_negatives`/`n_added_decoys`, and per-dataset conflict/DrugBank counts), and `model_summary.csv` (one row per model with AUROC, weight, and fold-stability flags, including discarded models). A top-level `summary.csv` with one row per pathogen is also written.

**Thresholds:** `FOLD_UNSTABLE_AUROC_STD = 0.05` and `LOW_WEIGHT_THRESHOLD = 0.3` (both in `src/default.py`).

---

## 18a_clone_hub_repos.py

Clones or updates the Ersilia Hub repo of each pathogen (`ersilia-os/{eosXXXX}`, from `ERSILIA_MODEL_IDS` in `src/default.py`) at `HUB_CLONES_DIR/{eosXXXX}` (`chembl-models-tmp`, next to this repo), or at `--repo-dir` for a single pathogen: the same argument 18b and 19 take. A missing repo is cloned over HTTPS (the remote the existing clones use). An existing one that is the root of its git repo, clean, on `main` and with nothing unpushed is fetched and fast-forwarded (never a merge commit). Anything else is left alone and reported: not a git repo, not on `main`, uncommitted changes, commits ahead of `origin/main`, or a failing git command. The exit code is 1 in those cases, so a refresh cannot start from a stale clone unnoticed. The Hub repos are also updated by CI, so a clone goes stale quickly. It prints the Hub commit each clone is at, which is where the refresh starts from. Accepts `--pathogen <code> [<code> ...]`. Prerequisite for `18b_update_ersilia_model.py` and `19_apply_and_fetch.py`.

---

## 18b_update_ersilia_model.py

Refreshes an already-incorporated Ersilia Hub model with newly trained checkpoints. For one pathogen it writes a refresh package to `output/18_emh_files/{pathogen}/`, runs the model once in a staging copy and checks the result. Nothing is written into the clone and nothing is pushed. Reads `output/10_reports/10_reports.csv`, `output/07_datasets/07_datasets_metadata.csv`, `output/09_models/{pathogen}/` and, for the consensus, `output/14_consensus/{pathogen}/anchors.json`. `--pathogen <p>` is required; `--repo-dir` defaults to `HUB_CLONES_DIR/{eosXXXX}`.

- `reports.csv` — the quality report of the kept sub-models, with the public `model_name` (lowercase, as Ersilia's schema requires) and the training-pipeline `original_name`.
- `run_columns.csv` — `consensus_score` row (when there is more than one sub-model) + one row per sub-model. Sub-models are ordered single-point before dose-response, pools before catch-alls, PubChem last, compound count descending. Descriptions give the dataset family, assay and compound counts and added negatives; no per-assay concentration cutoff (pool `cutoff` is a Youden score). **Descriptions must not contain commas** (nor quotes or line breaks): the Hub tooling does not read CSV quoting, so 18b stops if one does.
- `consensus.py` — `src/hub_consensus.py` with the pathogen's anchor table baked in (see below).
- `metadata.yml` — patched in place from the clone's file: `Output Dimension`, `Deployment` (block-style `- Local` only), `Description` and `Interpretation` (drafted, to be reviewed per pathogen before push).
- `install.yml` — pins `ersilia-pack-utils` 0.1.5 and `lazyqsar` 3.6.1 (the version of `environment.yml`) and the per-pathogen descriptor `--only` list.
- `run_output.csv` — produced by running `run.sh` in a temp staging dir built from the clone and the new artifacts, in the conda env `HUB_RUNTIME_ENV` (`cam-models-runtime`; `conda.sh` from `$CONDA_SH`, else next to `conda` on the PATH).
- `DIFF_SUMMARY.txt` — sub-model set diff, **public column names and order against the published ones (any change is flagged BREAKING: they are the model's API)**, per-sub-model `decision_cutoff_rank` drift, consensus threshold old → new and its calibration table, `lazyqsar` old → new, descriptor set old → new, and the checks below.
- `COPY_INSTRUCTIONS.txt` — paste-ready `rsync`/`cp` commands for the clone, plus the commit/push lines. Step 19 applies them.

**Consensus and its reference-library calibration:** each sub-model's rank (`predict_type="rank"`) already is a position against LazyQSAR's fixed 50,000-molecule reference library, stored in its checkpoint (0.65 = better than 99% of the library). The consensus, a weighted mean of such ranks, is placed on the same scale with the anchor table of step 14 (the pathogen's own, weighted, full model set): the same weighted mean on the reference library gives raw values at its p50/p90/p99/p99.9, mapped to 0.25/0.50/0.65/0.75, with (0, 0) and (1, 1) as ends. The 6 + 6 numbers are baked into `consensus.py`, which interpolates linearly through them (no lazyqsar at inference). Weights are unchanged: the seven quality weights of `reports.csv` plus the per-compound ramp w7, equal terms. NaN policy: any sub-model NaN gives a NaN consensus. **Threshold:** the consensus threshold is the decision rank 0.65, the same as every sub-model (the old pathogen-specific, tanh-derived threshold is gone). The published column keeps the name `consensus_score` (it is the API; the decision to rename it is open, see `docs/2026-10-05_steps18_refactor_plan.md`).

**Safeguards, each stops the script:** (1) the weights fingerprint stored in `anchors.json` must equal the one recomputed from `10_reports.csv` for the kept models (otherwise "re-run step 14"), and the model sets must be equal; (2) `10_reports.csv` must hold no model below `MIN_AUROC` (10a keeps only retained models; this used to be a silent filter); (3) the conda env must have the pinned `lazyqsar`; (4) the staged `consensus_score` must equal `src/consensus.py` applied to the printed sub-model ranks (tolerance 3e-4, the output is rounded to 4 decimals); (5) the staged model is also run on 25 DrugBank molecules (the 15 top of the step-14 consensus, where the calibration matters, and 10 seeded random ones; not shipped), and its consensus must be within 0.01 of step 14 or the script stops (it catches another model, another lazyqsar, a stale step 12b or another anchor table). Its sub-model ranks are compared with step 12b at the same 0.01, but a cell beyond it is only **flagged** for manual review in `DIFF_SUMMARY.txt`: the same checkpoint can give a slightly different rank on another machine (see "Machine-dependent ranks" below), so a flag needs a person's look and not a stop. The Hub's own 3 example molecules are not in DrugBank, so they cannot do this; (6) the example input in the clone must parse with RDKit (an unparseable SMILES gets no value from the model and `ersilia fetch` rejects a model with an empty output row, as happened for calbicans, whose example had two SMILES joined by `;`), and the example output has the columns of `run_columns.csv`, no empty value, and values in [0, 1].

**Machine-dependent ranks (accepted):** on rare molecules a sub-model's rank differs slightly between machines. Rounding differences in the chemeleon embedding (~1e-7) hit isotonic calibration steps finer than float32 in some logistic-regression heads. Observed: sub-model rank up to 0.017, consensus up to ~5e-4 (traced on mtuberculosis DR_0012, 2026-10-06). The fix belongs in LazyQSAR.

**Lowercase sub-model names:** Ersilia's schema requires lowercase output column names, so `reports.csv`'s `model_name` and `run_columns.csv`'s `name` are lowercase even though the checkpoint dirs under `output/09_models/{pathogen}/` keep their original training-pipeline casing (e.g. `DR_0001`).

After the script, you review `DIFF_SUMMARY.txt`; step 19 copies the package into the clone; you review `git diff` and push to `main` of `ersilia-os/{eosXXXX}` (no PR).

---

## 19_apply_and_fetch.py

Applies the refresh package produced by `18b_update_ersilia_model.py` to a local clone, then runs `ersilia fetch --from_dir` on it as a structural verification. Per pathogen:

1. `rsync -a --delete` checkpoints from `output/09_models/{pathogen}/` into `{repo-dir}/model/checkpoints/models/`, matching each (lowercased) kept sub-model name from `reports.csv` back to its actual (originally-cased) source directory case-insensitively.
2. Copies the 6 generated artifacts (`reports.csv`, `run_columns.csv`, `consensus.py`, `metadata.yml`, `install.yml`, `run_output.csv`) from `output/18_emh_files/{pathogen}/` into the right paths under `{repo-dir}/`.
3. `ersilia delete {eosXXXX}` — clears any cached install.
4. `ersilia fetch {eosXXXX} --from_dir {repo-dir}` — validates `metadata.yml` schema, runs `install.yml` end-to-end, and executes `run.sh` once. Fails loud on non-zero exit.

Pass `--no-fetch` to stop after steps 1-2 and skip the `ersilia delete`/`ersilia fetch` validation (also skips the `CONDA_SH` requirement).

Requires `--pathogen <p>`; `--repo-dir` defaults to `HUB_CLONES_DIR/{eosXXXX}`, where 18a puts the clone. After this, the user (with Claude) reviews `cd {repo-dir} && git diff`, commits, and pushes direct to `main` on `ersilia-os/{eosXXXX}` — one pathogen at a time, with an explicit go-ahead before each push.

---

## 20_cut_releases.py

Cuts the next GitHub release (`vN+1.0.0`) of one or more pathogens' Hub repos, a separate step run only after the post-push CI has succeeded. The release triggers `retag-release-docker.yml`, the only thing that republishes the model's Docker image. Per pathogen: the refresh commit is the newest commit on `main` without `[skip ci]` (CI itself adds README/metadata commits marked `[skip ci]` after it, which it never tests; every commit after the refresh must be by ersilia-bot and change only `README.md`/`metadata.yml`, otherwise no release); the newest "Test and upload model" run must be on that commit and succeeded, and a later "Test model image" run must have succeeded; a release published after the refresh commit means "already released". The latest tag is bumped by one per pathogen and the release is created on the checked HEAD (title = tag, generated notes). A pathogen that fails a check is skipped, the script exits 1. `--dry-run` runs the checks only.
