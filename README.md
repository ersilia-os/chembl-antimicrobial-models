This project has been financed by Project PID2023-148309OA-I00 funded by MICIU/AEI/10.13039/501100011033 and by ERDF, EU.

<img width="300" alt="miciu_cofinanciado" src="https://github.com/user-attachments/assets/65da03a5-a684-4fb9-9481-20a176cde27a" />


# Antimicrobial ML Models

This repository trains binary antimicrobial activity classification models from curated bioactivity datasets sourced from [ChEMBL](https://github.com/ersilia-os/chembl-antimicrobial-tasks) and [PubChem](https://github.com/ersilia-os/pubchem-antimicrobial-tasks). For each pathogen of interest, the pipeline downloads curated binary datasets (SMILES + activity label), optionally augments them with decoys, and trains QSAR models using [LazyQSAR](https://github.com/ersilia-os/lazy-qsar).

## Setup

Clone this repository and create a Conda environment:

```sh
git clone https://github.com/ersilia-os/chembl-antimicrobial-models.git
cd chembl-antimicrobial-models
conda env create -f environment.yml --prefix ./envs/camm
conda activate ./envs/camm
pip install --ignore-installed "lazyqsar[all]==3.4.2"
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
| 01  | `scripts/01_download_datasets_chembl.py` |
| 02a | `scripts/02a_download_datasets_pubchem.py` |
| 02b | `scripts/02b_plot_datasets.py` |
| 03  | `scripts/03_select_positives.py` |
| 04  | `scripts/04_setup_decoy_run.py` *(HPC)* |
| 05  | `scripts/05_run_decoys.sh` *(HPC)* / `scripts/05_run_decoys_ersilia.sh` *(local)* |
| 06  | `scripts/06_aggregate_decoys.py` |
| 07a | `scripts/07a_prepare_datasets.py` |
| 07b | `scripts/07b_quality_checks.py` |
| 07c | `scripts/07c_plot_datasets.py` |
| 08  | `scripts/08_download_weights.py` *(HPC)* |
| 09  | `scripts/09_run_models.sh` *(HPC)* / `scripts/09_fit_models_local.py` *(local)* |
| 10a | `scripts/10a_aggregate_reports.py` |
| 10b | `scripts/10b_training_results.py` |
| 11  | `scripts/11_download_drugbank.py` |
| 12a | `scripts/12a_predict_drugbank.py` / `scripts/12a_predict_drugbank_local.py` |
| 12b | `scripts/12b_fit_transformation.py` |
| 13  | `scripts/13_predict_drugbank_ersilia.sh` |
| 14  | `scripts/14_consensus_scoring.py` |
| 15  | `scripts/15_recapitulate_models.py` |
| 16  | `scripts/16_recapitulate_consensus.py` |
| 16b | `scripts/16b_consensus_results.py` |
| 17  | `scripts/17_quality_checks.py` |
| 18  | `scripts/18_update_ersilia_model.py` |

## Repository structure

```
chembl-antimicrobial-models/
├── config/
│   └── pathogens.csv               # Pathogen code → organism name mapping
├── data/
│   ├── raw/                        # Downloaded datasets per pathogen
│   └── processed/                  # Merged dataset metadata per pathogen
├── envs/
│   └── camm/                       # Project conda env (gitignored; created with --prefix)
├── scripts/                        # Pipeline scripts
├── notebooks/                      # Exploratory notebooks
└── output/
    ├── 02_datasets/                # 02b dataset-summary figure
    ├── 03_select_positives/        # Selected active SMILES + per-split inputs
    ├── 04_positives_splits/        # split_XXX.csv (HPC) [removed by step 06 --cleanup]
    ├── 04_decoys_sif_image/        # Apptainer SIF image for eos3e6s
    ├── 05_decoys/                  # eos3e6s output per split [removed by step 06 --cleanup]
    ├── 05_logs/                    # SLURM logs for decoy generation [removed by step 06 --cleanup]
    ├── 06_decoys/                  # 06_eos3e6s_v1.csv — aggregated decoys
    ├── 07_datasets/                # Per-pathogen compound CSVs + 07_datasets_metadata.csv + 07_dup_report.csv + 07c figure
    ├── 08_weights/                 # LazyQSAR descriptor weight cache (created by step 08)
    ├── 09_reports/                 # Per-dataset CV reports + _folds.json
    ├── 09_models/                  # Trained LazyQSAR models (one dir per dataset)
    ├── 09_logs/                    # SLURM logs for model training
    ├── 10_reports/                 # 10_reports.csv, 10_discarded_models.csv, plots/ (10b)
    ├── 12_drugbank/                # Per-pathogen DrugBank rank predictions + 12b tanh fit
    ├── 13_drugbank_ersilia/        # Single-model Ersilia Hub predictions on DrugBank
    ├── 14_consensus/               # Per-pathogen consensus + unweighted/transformed variants
    ├── 15_recapitulate_models/     # Pairwise model agreement
    ├── 16_recapitulate_consensus/  # Leave-one-out + full consensus recap + plots/ (16b)
    ├── 17_quality_checks/          # Per-pathogen QA + top-level summary.csv
    └── 18_emh_files/               # Ersilia Model Hub submission bundles
```

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
