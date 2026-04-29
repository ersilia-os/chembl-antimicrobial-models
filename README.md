# Antimicrobial ML Models from ChEMBL

This repository trains binary antimicrobial activity classification models from datasets produced by the [`chembl-antimicrobial-tasks`](https://github.com/ersilia-os/chembl-antimicrobial-tasks) pipeline. For each pathogen of interest, the pipeline downloads curated binary datasets (SMILES + activity label), optionally augments them with decoys, and trains QSAR models using [LazyQSAR](https://github.com/ersilia-os/lazy-qsar).

## Setup

Clone this repository and create a Conda environment:

```sh
git clone https://github.com/ersilia-os/chembl-antimicrobial-models.git
cd chembl-antimicrobial-models
conda env create -f environment.yml --prefix ./envs/camm
conda activate ./envs/camm
```

> **Note:** Environment creation may take 5–10 minutes.

The `--prefix ./envs/camm` flag places the environment inside the repository directory. This is intentional: on HPC clusters, only the shared filesystem is visible to compute nodes, so the environment must live alongside the code rather than in the default local conda path. Two pipeline steps (04 and 07) are designed to run directly on the cluster to allow for parallelization — see the [Pipeline overview](#pipeline-overview) section below.

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
| 01 | `scripts/01_download_datasets.py` | Downloads binary datasets from `chembl-antimicrobial-tasks` outputs; optionally selects a representative subset with `--select_representatives` |
| 02 | `scripts/02_select_positives.py` | Extracts all active compounds (bin == 1) across every dataset, deduplicates SMILES, and records provenance and split indices in `output/results/02_selected_positives.csv` |
| 03 | `scripts/03_setup_decoy_run.py` | Splits positives into batches, builds the `eos3e6s` Apptainer SIF image via `ersilia_apptainer create` (accepts `--version`, default `v1.0.0`), and prints the exact `sbatch` command to submit step 04; requires `envs/camm` from the Setup step |
| 04 | `scripts/04_run_decoys.sh` | Static SLURM array job script; submit using the command printed by step 03 (`sbatch --chdir=<repo_root> --array=0-N%M scripts/04_run_decoys.sh`); runs `eos3e6s` on each input split via `ersilia_apptainer` |
| 05 | `scripts/05_aggregate_decoys.py` | Streams all per-split CSVs into `output/results/05_eos3e6s_v1.csv`; `--cleanup` removes intermediate directories (splits, decoys, logs) only if all expected splits are present |
| 06 | `scripts/06_prepare_datasets.py` | Extracts raw compound CSVs from per-pathogen zip archives into `output/results/06_datasets/{pathogen}/{name}.csv` (columns: `smiles, bin`); augments datasets with active ratio > 0.5 with decoy compounds targeting ratio 0.1; saves enriched metadata to `output/results/06_datasets_metadata.csv`; prints the exact `sbatch` command to submit step 07 |
| 07 | `scripts/07_run_models.sh` | Static SLURM array job script; submit using the command printed by step 06 (`sbatch --chdir=<repo_root> --array=0-N%20 scripts/07_run_models.sh`); trains a LazyQSAR model for each dataset |

Steps 04 and 07 are static SLURM scripts designed to run on an HPC cluster; all other scripts run locally.

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
├── scripts/                    # Pipeline scripts (01–07)
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
        └── 07_logs/                         # SLURM job logs for model training
```

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
