# Antimicrobial ML Models

This repository trains binary antimicrobial activity classification models from curated bioactivity datasets sourced from [ChEMBL](https://github.com/ersilia-os/chembl-antimicrobial-tasks) and [PubChem](https://github.com/ersilia-os/pubchem-antimicrobial-tasks). For each pathogen of interest, the pipeline downloads curated binary datasets (SMILES + activity label), optionally augments them with decoys, and trains QSAR models using [LazyQSAR](https://github.com/ersilia-os/lazy-qsar).

## Setup

Clone this repository and create a Conda environment:

```sh
git clone https://github.com/ersilia-os/chembl-antimicrobial-models.git
cd chembl-antimicrobial-models
conda env create -f environment.yml --prefix ./envs/camm
conda activate ./envs/camm
pip install --ignore-installed "lazyqsar[all]==3.2.0"
```

> **Note:** Environment creation may take 5–10 minutes. The explicit `pip install` step after `conda env create` is required because conda's pip integration does not always resolve all transitive dependencies — running pip directly ensures LazyQSAR and all its dependencies are fully installed.

The `--prefix ./envs/camm` flag places the environment inside the repository directory. This is intentional: on HPC clusters, only the shared filesystem is visible to compute nodes, so the environment must live alongside the code rather than in the default local conda path. Two pipeline steps (05 and 09) are designed to run directly on the cluster to allow for parallelization — see the [Pipeline overview](#pipeline-overview) section below.

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

See [scripts/README.md](scripts/README.md) for a description of each step.

| Step | Script |
|------|--------|
| 01 | `scripts/01_download_datasets_chembl.py` |
| 02 | `scripts/02_download_datasets_pubchem.py` |
| 03 | `scripts/03_select_positives.py` |
| 04 | `scripts/04_setup_decoy_run.py` |
| 05 | `scripts/05_run_decoys.sh` *(HPC)* |
| 06 | `scripts/06_aggregate_decoys.py` |
| 07 | `scripts/07_prepare_datasets.py` |
| 08 | `scripts/08_download_weights.py` *(HPC)* |
| 09 | `scripts/09_run_models.sh` *(HPC)* |
| 10 | `scripts/10_aggregate_reports.py` |
| 11 | `scripts/11_download_drugbank.py` |
| 12 | `scripts/12_predict_drugbank.py` |
| 13 | `scripts/13_predict_drugbank_ersilia.sh` |
| 14 | `scripts/14_consensus_scoring.py` |
| 15 | `scripts/15_recapitulate_models.py` |
| 16 | `scripts/16_recapitulate_consensus.py` |
| 17 | `scripts/17_quality_checks.py` |
| 18 | `scripts/18_emh_files.py` |
| 19 | `scripts/19_euopenscreen_benchmark.sh` |

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
├── scripts/                    # Pipeline scripts (01–19)
├── notebooks/                  # Exploratory notebooks
└── output/
    └── results/
        ├── 03_selected_positives.csv        # Unique active SMILES with provenance
        ├── 04_positives_splits/             # Per-split input CSVs (split_XXX.csv) [removed by step 05 --cleanup]
        ├── 04_eos3e6s_v1.sif                # Apptainer SIF image
        ├── 05_decoys/                       # eos3e6s output CSVs per split [removed by step 05 --cleanup]
        ├── 05_logs/                         # SLURM job logs for decoy generation [removed by step 05 --cleanup]
        ├── 06_eos3e6s_v1.csv                # Aggregated eos3e6s predictions
        ├── 07_datasets/                     # Per-pathogen compound CSVs (smiles, bin)
        ├── 07_datasets_metadata.csv         # Normalised merged metadata sorted by pathogen; adds decoys, final_ratio, final_compounds columns
        ├── 08_weights/                      # LazyQSAR descriptor weight cache (created by step 08)
        ├── 09_reports/                      # Per-dataset CV reports (one CSV per dataset)
        ├── 09_models/                       # Trained LazyQSAR models (one dir per dataset)
        ├── 09_logs/                         # SLURM job logs for model training
        ├── 10_reports.csv                   # One summarised row per dataset: model_name, metrics, weights, decision cutoff, portfolio, predict_rank scores
        ├── 12_drugbank/                     # Per-pathogen DrugBank rank predictions (one CSV per pathogen)
        ├── 13_consensus/                    # Per-pathogen consensus scores (smiles, consensus_score)
```

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
