# Consensus Scoring Report

## Overview

This pipeline trains per-dataset QSAR binary classifiers (LazyQSAR) for 15 pathogens, then aggregates their predictions on DrugBank into a weighted consensus score per compound. The final output ranks every DrugBank compound by how likely it is to be active against each pathogen.

The pipeline has 17 numbered scripts. Steps 01–07 prepare training data. Steps 08–10 train models and collect cross-validation metrics. Steps 11–14 produce the consensus DrugBank ranking. Steps 15–17 are diagnostic and quality-checking steps.

---

## Pipeline steps

### Data acquisition

**01 – Download ChEMBL datasets**
Downloads binary bioactivity datasets from ChEMBL via the Ersilia Object Storage service. Each dataset is one assay for one pathogen, with compounds labelled active (bin=1) or inactive (bin=0). Stored in `data/raw/chembl/<pathogen>/`.

**02 – Download PubChem datasets**
Downloads PubChem bioassay summary tables from the `pubchem-antimicrobial-tasks` Ersilia repository. These are pre-annotated assay-to-pathogen mappings. Stored in `data/raw/pubchem/`.

**03 – Select positives**
Extracts the union of active compounds across all ChEMBL and PubChem datasets, deduplicated by InChIKey. Outputs `output/03_select_positives/03_selected_positives.csv`, one row per unique active compound, with a `found_in` column listing all source datasets and an `n_active` column counting dataset-level occurrences. This file is the input for decoy generation.

### Decoy generation

**04 – Set up decoy run**
Downloads the DUD-E style decoy generator SIF image (Ersilia model `eos3e6s`) and splits the positives file into batches for SLURM array submission. Outputs go to `output/04_positives_splits/` and `output/04_decoys_sif_image/`.

**05 – Run decoys** *(shell script, SLURM)*
Runs the Apptainer container for each batch of positives to generate property-matched decoy compounds (same MW, cLogP, rotatable bonds distribution as the actives, but structurally dissimilar). Raw outputs per batch land in `output/05_decoys/`.

**06 – Aggregate decoys**
Concatenates all per-batch decoy CSVs from step 05 into a single file `output/06_decoys/06_eos3e6s_v1.csv`. Also validates that each decoy SMILES can be parsed by RDKit and reports coverage statistics.

**07 – Prepare datasets**
The central data-assembly step. For each dataset (ChEMBL or PubChem):
1. Extracts actives and inactives from the source ZIP/CSV.
2. Optionally augments the inactive class with sampled decoys when the active:inactive ratio is below a threshold.
3. Writes one CSV per dataset to `output/07_datasets/<pathogen>/<name>.csv` with columns `smiles`, `bin`, `decoy`.
4. Writes a master metadata table `output/07_datasets/07_datasets_metadata.csv` recording pathogen, dataset name, compound counts, decoy counts, and a `label` field classifying each dataset as individual (A/B), merged (M), or general (G).

If a PubChem assay references a ChEMBL dataset by ID, the ChEMBL dataset is flagged for removal (`keep=False`) to avoid duplication.

### Model training

**08 – Download weights**
Downloads pre-trained encoder weights for the five descriptor types used by LazyQSAR: `cddd`, `chemeleon`, `clamp`, `morgan`, `rdkit`. Must be run once before the SLURM array job. Also prints the `sbatch` commands for steps 09, split into small (≤30k compounds) and large (>30k) jobs.

**09 – Run models** *(Python called by SLURM array)*
For each dataset (identified by SLURM array task ID):
1. Runs 5-fold stratified cross-validation using `LazyClassifierQSAR` (mode=slow; all five descriptor types).
2. Records per-fold AUROC, AUPRC, BEDROC, their baselines, out-of-fold AUCs per descriptor, and raw score arrays in `output/09_reports/<pathogen>/<name>.csv`.
3. Trains a final model on all data and saves it to `output/09_models/<pathogen>/<model_name>/`.

LazyQSAR trains one base classifier (RF, XGB, LR, SVC) per descriptor type and selects the best-performing portfolio; the trained model includes a `metadata.json` with the `decision_cutoff_rank` (rank threshold separating predicted actives from predicted inactives on the training set) and the `portfolio` of descriptor types used.

**10 – Aggregate reports**
Reads all per-dataset CV CSVs from step 09 and computes summary metrics and quality weights for each model. Outputs `output/09_reports/10_reports.csv` (one row per model, 395 models across 15 pathogens as of the current run). This file is the input to all downstream steps.

---

## Model quality weights (W1–W7)

Each model receives seven quality weights computed in step 10. They are all on a [0, 1] scale. The `final_weight` is their unweighted arithmetic mean.

### W1 — Dataset specificity

| Dataset label | w1 |
|---|---|
| A – individual assay (single ChEMBL assay) | 1.0 |
| B – individual assay (PubChem source) | 1.0 |
| M – merged (actives from multiple assays, same activity type) | 0.5 |
| G – general (actives from all assays regardless of type) | 0.0 |

Rationale: a model trained on a single well-defined assay (e.g., one IC50 series against one target) is expected to be more precise than one trained on a heterogeneous pool of actives. General datasets aggregate multiple activity types and thresholds, which introduces label noise.

In the current run: 97 models have w1=1.0 (individual), 213 have w1=0.5 (merged), 85 have w1=0.0 (general).

### W2 — Decoy contamination

```
w2 = max(0, 1 − n_decoys / n_inactives_test)
```

Measures what fraction of the inactive test set consists of added decoys rather than experimentally confirmed inactives. A model with no decoys scores 1.0; one where decoys make up all inactives scores 0.0.

Note: the denominator `n_inactives_test` is computed as `compounds_test − positives_test` from the CV folds, which already includes the decoys. This makes w2 somewhat lenient for heavily augmented datasets (the denominator grows with the decoys). Mean w2 in the current run is 0.293, reflecting that many datasets are heavily decoy-augmented.

### W3 — Cross-validation AUROC

```
w3 = 0                           if mean_CV_AUROC ≤ 0.70
w3 = (mean_CV_AUROC − 0.70) / 0.30   otherwise (capped at 1.0)
```

Linear ramp from 0 at AUROC=0.70 to 1.0 at AUROC=1.0. Models with AUROC at or below 0.70 receive w3=0 but are **not excluded** from the consensus — they still contribute through the other six weights. In the current run, 4 models have w3=0 (all PubChem-sourced, AUROC 0.53–0.68). Mean w3 is 0.861.

### W4 — AUPRC enrichment

Combines two equal sub-scores (each 0–0.5):
- **Absolute excess**: how far AUPRC is above the prevalence baseline (random classifier), scaled to the maximum possible excess.
- **Fold enrichment**: how many times above baseline AUPRC is, scaled from 1× (no enrichment) to 10× (full score).

```
c1 = 0.5 × clip((AUPRC − baseline) / (1 − baseline), 0, 1)
c2 = 0.5 × clip((AUPRC/baseline − 1) / 9, 0, 1)
w4 = c1 + c2
```

AUPRC is more informative than AUROC when class imbalance is high (which is typical here — most datasets have far more inactives than actives). Mean w4 is 0.805.

### W5 — BEDROC enrichment

Same formula as W4 but using BEDROC (Boltzmann-Enhanced Discrimination of ROC) in place of AUPRC. BEDROC down-weights the tail of the ranked list and emphasises whether the true actives are concentrated at the very top of the ranking — particularly relevant for virtual screening, where only the top-ranked compounds are followed up.

```
c1 = 0.5 × clip((BEDROC − baseline_BEDROC) / (1 − baseline_BEDROC), 0, 1)
c2 = 0.5 × clip((BEDROC/baseline_BEDROC − 1) / 9, 0, 1)
w5 = c1 + c2
```

Mean w5 is 0.790. Maximum achievable w5 in practice is ~0.93 (random-baseline BEDROC cannot be zero).

### W6 — Total compound count

Piecewise linear on the total number of test-set compounds across all 5 folds:

| Compounds | w6 |
|---|---|
| < 100 | 0.0 |
| 1,000 | 0.25 |
| 10,000 | 0.5 |
| ≥ 100,000 | 1.0 |

Rewards larger datasets as more statistically reliable. Mean w6 is 0.312 (most datasets are small-to-medium scale).

### W7 — Active compound count

Piecewise linear on the number of positive test-set examples:

| Actives | w7 |
|---|---|
| < 50 | 0.0 |
| 250 | 0.25 |
| 1,000 | 0.5 |
| ≥ 10,000 | 1.0 |

Rewards datasets with many confirmed actives, which constrain the positive class better. Mean w7 is 0.192 (active compounds are typically scarce).

### Final weight and normalized weight

```
final_weight = mean(w1, w2, w3, w4, w5, w6, w7)
```

`final_normalized_weight` rescales `final_weight` within each pathogen so that all models for a given pathogen sum to 100. This is used to compare relative model influence within a pathogen.

---

## Consensus scoring (steps 12–14)

### Step 11 — Download DrugBank
Downloads a reference SMILES file for all DrugBank approved drugs. This is the compound set scored by the consensus.

### Step 12 — Predict DrugBank ranks
Runs every trained model on the DrugBank SMILES set. Each model outputs a `prob_rank` for each compound: the percentile rank of its predicted probability among all DrugBank compounds under that model (0=lowest, 1=highest). Outputs one CSV per pathogen: `output/12_drugbank/<pathogen>.csv`, columns `smiles | model_name_1 | model_name_2 | ...`.

### Step 13 — Predict DrugBank (Ersilia models) *(optional)*
A supplementary SLURM script that runs the Ersilia Model Hub on DrugBank using GPU nodes (separate from LazyQSAR). Outputs go to `output/13_drugbank_ersilia/`.

### Step 14 — Consensus scoring

The consensus score for compound *i* is a weighted mean of its `prob_rank` values across all *M* models for that pathogen:

```
consensus_score[i] = Σ_m ( prob_rank[i,m] × weight[i,m] ) / Σ_m weight[i,m]
```

where `weight[i,m]` is the average of eight values — the seven model-level quality weights (W1–W7) plus one compound-level weight (W8):

```
weight[i,m] = mean(w1_m, w2_m, w3_m, w4_m, w5_m, w6_m, w7_m, w8[i,m])
```

**W8 — Decision cutoff reward**
The only weight that varies per compound. It is zero for compounds ranked at or below the model's `decision_cutoff_rank` and rises linearly to 1.0 for a compound ranked at the very top:

```
w8[i,m] = 0                                         if prob_rank[i,m] ≤ cutoff_m
w8[i,m] = (prob_rank[i,m] − cutoff_m) / (1 − cutoff_m)   otherwise
```

The `decision_cutoff_rank` is stored in `metadata.json` for each model and represents the rank at which the model switches from predicting inactive to active on its training data. W8 therefore rewards compounds that a model places confidently in the active region and penalises compounds below the decision boundary.

**Leave-one-out columns**
The output CSV also contains one `excluded_<model>` column per model, computed as the consensus over all *M−1* other models (the named model excluded). These are used in step 16 to evaluate each model's independent contribution.

**Unweighted baseline**
A parallel `_unweighted.csv` file contains the simple mean of `prob_rank` values across all models, without any weighting. This serves as a baseline for evaluating whether the weighting scheme improves the consensus.

#### IQR-restoring tanh transformation

Averaging M independent model rankings compresses the score distribution: the interquartile range (IQR) of the consensus shrinks by roughly √M relative to an individual model. This makes top compounds harder to distinguish. A tanh transformation is applied to restore the IQR toward the average IQR of the individual models:

```
f(x) = 0.5 + 0.5 × tanh(k × (x − 0.5)) / tanh(k/2)
```

The steepness `k` depends only on *M* via a saturating-exponential fit derived from 9 pathogens:

```
k(M) = 2 × (1 + 1.156 × (1 − exp(−M / 6.47)))
```

Properties of the transform:
- Centred at 0.5 (the neutral point of the prob_rank scale).
- Normalised so that f(0)=0 and f(1)=1 exactly.
- Strictly monotone (ranks are preserved).
- Output is guaranteed in [0, 1] without clipping.
- For small M (e.g., M=2), k≈2 and the correction is mild. For large M (e.g., M=50), k≈4.3 and the correction is stronger.

The transformed outputs are saved as `_transformed.csv` (weighted and unweighted variants). The tanh transform is applied after consensus scoring, as a post-processing step.

---

## Validation steps

### Step 15 — Pairwise model recapitulation
For every ordered pair (A, B) of models for a pathogen, measures how consistently they rank DrugBank compounds:
- **Spearman and Pearson correlation** of continuous prob_ranks.
- **Hit overlap** at top 10, 100, 500 compounds.
- **AUROC** of A scoring against B binarized at the top 0.1%, 1%, and 5%.

Output: `output/15_recapitulate_models/<pathogen>.csv`. High pairwise agreement across models supports the consensus; low agreement suggests models are trained on contradictory or very narrow signal.

### Step 16 — Consensus recapitulation
For each model, measures how well the consensus (weighted and unweighted) recapitulates that model's individual ranking. Computed in four combinations: model vs leave-one-out consensus (model excluded) and model vs full consensus (model included), for both weighted and unweighted variants. Uses the same metrics as step 15.

Output: `output/16_recapitulate_consensus/<pathogen>_{exc/inc}_{weighted/unweighted}.csv`.

### Step 17 — Quality checks
Produces per-pathogen deduplicated compound lists for detecting data quality issues:
- **`all_smiles_no_decoys.csv`**: one row per unique InChIKey across all training datasets, excluding added decoys. Flags DrugBank overlap (`in_drugbank`).
- **`all_smiles_decoys.csv`**: same but including decoys. Additional flags:
  - `label_conflict`: InChIKey appears as active in ≥1 dataset AND as inactive or decoy in ≥1 dataset.
  - `decoy_inactive_dup`: InChIKey appears as both an added decoy and a real inactive — the decoy redundantly duplicates an existing negative.
  - `intra_dataset_conflicts`: dataset names where the same InChIKey appears under conflicting roles within a single CSV.
- **`summary.csv`**: per-dataset count of intra-dataset duplications introduced by decoys, with a TOTAL row.

---

## Output file reference

| Output | Produced by | Contents |
|---|---|---|
| `output/09_reports/10_reports.csv` | step 10 | One row per model: CV metrics, W1–W7, final_weight, cutoff |
| `output/12_drugbank/<pathogen>.csv` | step 12 | prob_rank per compound per model |
| `output/14_consensus/<pathogen>.csv` | step 14 | Weighted consensus + leave-one-out columns |
| `output/14_consensus/<pathogen>_unweighted.csv` | step 14 | Unweighted consensus + leave-one-out |
| `output/14_consensus/<pathogen>_transformed.csv` | step 14 | Tanh-transformed weighted consensus |
| `output/14_consensus/<pathogen>_unweighted_transformed.csv` | step 14 | Tanh-transformed unweighted consensus |
| `output/15_recapitulate_models/<pathogen>.csv` | step 15 | Pairwise model agreement |
| `output/16_recapitulate_consensus/<pathogen>_*.csv` | step 16 | Model vs consensus agreement (4 variants) |
| `output/17_quality_checks/<pathogen>/all_smiles_*.csv` | step 17 | Deduplicated compound quality flags |
| `output/17_quality_checks/summary.csv` | step 17 | Intra-dataset decoy duplication summary |

---

## Current run statistics

- **15 pathogens**: abaumannii, calbicans, campylobacter, ecoli, efaecium, enterobacter, hpylori, kpneumoniae, mtuberculosis, ngonorrhoeae, paeruginosa, pfalciparum, saureus, smansoni, spneumoniae
- **395 models** total (range: 1 for campylobacter to 97 for pfalciparum)
- **4 models** with w3=0 (mean CV AUROC ≤ 0.70): calbicans/pubchem_b (0.571), calbicans/pubchem_c (0.587), ecoli/pubchem_a (0.681), pfalciparum/pubchem_h (0.535)
- **0 models** with final_weight=0 (all models contribute to the consensus)
- Mean final_weight: 0.538 (range 0.298–0.950)
