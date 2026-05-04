# Antimicrobial ML Models

This repository trains binary antimicrobial activity classification models from curated bioactivity datasets sourced from [ChEMBL](https://github.com/ersilia-os/chembl-antimicrobial-tasks) and [PubChem](https://github.com/ersilia-os/pubchem-antimicrobial-tasks). For each pathogen of interest, the pipeline downloads curated binary datasets (SMILES + activity label), optionally augments them with decoys, and trains QSAR models using [LazyQSAR](https://github.com/ersilia-os/lazy-qsar).

## Setup

Clone this repository and create a Conda environment:

```sh
git clone https://github.com/ersilia-os/chembl-antimicrobial-models.git
cd chembl-antimicrobial-models
conda env create -f environment.yml --prefix ./envs/camm
conda activate ./envs/camm
```

> **Note:** Environment creation may take 5–10 minutes.

The `--prefix ./envs/camm` flag places the environment inside the repository directory. This is intentional: on HPC clusters, only the shared filesystem is visible to compute nodes, so the environment must live alongside the code rather than in the default local conda path. Two pipeline steps (04 and 08) are designed to run directly on the cluster to allow for parallelization — see the [Pipeline overview](#pipeline-overview) section below.

### Data download

Data is stored with the [eosvc](https://github.com/ersilia-os/eosvc) tool:

```sh
eosvc download --path data
eosvc download --path output
```

## Supported pathogens

| Code | Organism |
|------|----------|
| `abaumannii` | *Acinetobacter baumannii* |
| `calbicans` | *Candida albicans* |
| `campylobacter` | *Campylobacter* spp. |
| `ecoli` | *Escherichia coli* |
| `efaecium` | *Enterococcus faecium* |
| `enterobacter` | *Enterobacter* spp. |
| `hpylori` | *Helicobacter pylori* |
| `kpneumoniae` | *Klebsiella pneumoniae* |
| `mtuberculosis` | *Mycobacterium tuberculosis* |
| `ngonorrhoeae` | *Neisseria gonorrhoeae* |
| `paeruginosa` | *Pseudomonas aeruginosa* |
| `pfalciparum` | *Plasmodium falciparum* |
| `saureus` | *Staphylococcus aureus* |
| `smansoni` | *Schistosoma mansoni* |
| `spneumoniae` | *Streptococcus pneumoniae* |

## Pipeline overview

| Step | Script | What it does |
|------|--------|-------------|
| 01 | `scripts/01_download_datasets_chembl.py` | Downloads binary datasets from `chembl-antimicrobial-tasks` outputs into `data/raw/chembl/` and `data/processed/chembl/`; optionally selects a representative subset with `--select_representatives` |
| 02 | `scripts/02_download_datasets_pubchem.py` | Downloads curated PubChem bioassay datasets from `pubchem-antimicrobial-tasks` outputs into `data/raw/pubchem/` and `data/processed/pubchem/`; per-AID metadata written to `02_pubchem_datasets.csv` |
| 03* | `scripts/02_select_positives.py` | Extracts all active compounds (bin == 1) across every dataset, deduplicates SMILES, and records provenance and split indices in `output/results/02_selected_positives.csv` |
| 04* | `scripts/03_setup_decoy_run.py` | Splits positives into batches, builds the `eos3e6s` Apptainer SIF image via `ersilia_apptainer create` (accepts `--version`, default `v1.0.0`), and prints the exact `sbatch` command to submit step 04; requires `envs/camm` from the Setup step |
| 05* | `scripts/04_run_decoys.sh` | Static SLURM array job script; submit using the command printed by step 03 (`sbatch --chdir=<repo_root> --array=0-N%M scripts/04_run_decoys.sh`); runs `eos3e6s` on each input split via `ersilia_apptainer` |
| 06* | `scripts/05_aggregate_decoys.py` | Streams all per-split CSVs into `output/results/05_eos3e6s_v1.csv`; `--cleanup` removes intermediate directories (splits, decoys, logs) only if all expected splits are present |
| 07* | `scripts/06_prepare_datasets.py` | Extracts raw compound CSVs from per-pathogen zip archives into `output/results/06_datasets/{pathogen}/{name}.csv` (columns: `smiles, bin`); augments datasets with active ratio > 0.5 with decoy compounds targeting ratio 0.1; saves enriched metadata to `output/results/06_datasets_metadata.csv` |
| 08* | `scripts/07_download_weights.py` | Downloads LazyQSAR descriptor weights (cddd, chemeleon, clamp) to `output/results/07_weights/.lazyqsar/`; run once from the login node before submitting step 08; accepts `--path` to override the cache location; prints two separate `sbatch` commands — one for small datasets (≤30k compounds, 16 GB) and one for large datasets (>30k compounds, 64 GB) |
| 09* | `scripts/08_run_models.sh` | Static SLURM array job script; submit using the commands printed by step 07; trains a LazyQSAR model for each dataset and saves CV reports and the final model |
| 10* | `scripts/09_aggregate_reports.py` | Iterates over datasets from `06_datasets_metadata.csv`, skips incomplete runs (< 5 folds), and writes one summarised row per dataset to `output/results/09_reports.csv`; reports mean/std AUROC/AUPRC/BEDROC, per-descriptor OOF AUCs, seven model-quality weights (w1–w7), `final_weight`, `final_normalized_weight`, decision cutoff, model sizes, portfolio, and aggregated predict_rank scores |
| 11* | `scripts/10_download_drugbank.py` | Downloads DrugBank SMILES from [`ersilia-os/sars-cov-2-chemspace`](https://github.com/ersilia-os/sars-cov-2-chemspace); default output `data/processed/10_drugbank_smiles.csv`; `--only_smiles` returns a single deduplicated SMILES column sorted alphabetically; `--output` overrides the destination path |
| 11 | `scripts/11_predict_drugbank.py` | Loads all trained models for a given pathogen and runs `predict_rank` on every DrugBank compound; model order follows `09_reports.csv`; outputs one CSV per pathogen with one column per model; `--pathogen <code>` for a single pathogen, `--all_pathogens` to iterate all in metadata order |

Steps 04 and 08 are static SLURM scripts designed to run on an HPC cluster; all other scripts run locally.

## Repository structure

```
chembl-antimicrobial-models/
├── config/
│   └── pathogens.csv           # Pathogen code → organism name mapping
├── data/
│   ├── raw/                    # Downloaded datasets per pathogen
│   └── processed/              # Merged dataset metadata per pathogen
├── envs/
│   └── camm/                   # Project conda env (gitignored; created with --prefix)
├── scripts/                    # Pipeline scripts (01–09)
├── notebooks/                  # Exploratory notebooks
└── output/
    └── results/
        ├── 02_selected_positives.csv        # Unique active SMILES with provenance
        ├── 03_positives_splits/             # Per-split input CSVs (split_XXX.csv) [removed by step 05 --cleanup]
        ├── 03_eos3e6s_v1.sif                # Apptainer SIF image
        ├── 04_decoys/                       # eos3e6s output CSVs per split [removed by step 05 --cleanup]
        ├── 04_logs/                         # SLURM job logs for decoy generation [removed by step 05 --cleanup]
        ├── 05_eos3e6s_v1.csv                # Aggregated eos3e6s predictions
        ├── 06_datasets/                     # Per-pathogen compound CSVs (smiles, bin)
        ├── 06_datasets_metadata.csv         # Enriched metadata with decoys, final_ratio, and final_compounds columns
        ├── 07_weights/                      # LazyQSAR descriptor weight cache (created by step 07)
        ├── 08_reports/                      # Per-dataset CV reports (one CSV per dataset)
        ├── 08_models/                       # Trained LazyQSAR models (one dir per dataset)
        ├── 08_logs/                         # SLURM job logs for model training
        ├── 09_reports.csv                   # One summarised row per dataset: model_name, metrics, weights, decision cutoff, portfolio, predict_rank scores
        ├── 11_drugbank/                     # Per-pathogen DrugBank rank predictions (one CSV per pathogen)
└── data/
    └── processed/
        └── 10_drugbank_smiles.csv           # DrugBank SMILES (downloaded by step 10)
```

## Weighting strategy

Each trained model receives a `final_weight` score in `09_reports.csv`, computed as the mean of seven independent weights (w1–w7). A higher `final_weight` indicates a model that is more reliable and ready for deployment.

### Model-dependent weights

These weights reflect intrinsic properties of the dataset and how well the model performed during cross-validation.

| Weight | Description |
|--------|-------------|
| **w1** | **Dataset type.** Individual pathogen-specific datasets score 1.0; merged (multi-source) datasets score 0.5; general datasets score 0.0. See [`chembl-antimicrobial-tasks`](https://github.com/ersilia-os/chembl-antimicrobial-tasks) for details on dataset types. |
| **w2** | **Decoy contamination.** 1.0 if no decoy compounds were added to the inactive set; decreases linearly toward 0 as the fraction of decoys among inactives increases. |
| **w3** | **Cross-validated AUROC.** 0 for mean CV AUROC ≤ 0.7; linear to 1 at AUROC = 1.0. |
| **w4** | **AUPRC enrichment.** Sum of two equal sub-scores (each 0–0.5): (i) absolute excess of mean AUPRC over the prevalence baseline, scaled from 0 at baseline to 0.5 at AUPRC = 1; (ii) fold enrichment over baseline, scaled from 0 at ≤1× to 0.5 at ≥10×. |
| **w5** | **BEDROC enrichment.** Same two-component scheme as w4, applied to mean BEDROC versus its expected random-ranking baseline (α = 20). BEDROC captures early enrichment — how well actives are concentrated at the top of the ranked list. |
| **w6** | **Total compound count.** Piecewise linear: 0 at < 100 compounds, 0.25 at 1k, 0.5 at 10k, 1.0 at ≥ 100k. |
| **w7** | **Active compound count.** Piecewise linear: 0 at < 50 actives, 0.25 at 250, 0.5 at 1k, 1.0 at ≥ 10k. |

### Sample-dependent weights

*Placeholder — to be defined.*

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
