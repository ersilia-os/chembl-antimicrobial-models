# Antimicrobial ML Models from ChEMBL

This repository trains binary antimicrobial activity classification models from datasets produced by the [`chembl-antimicrobial-tasks`](https://github.com/ersilia-os/chembl-antimicrobial-tasks) pipeline. For each pathogen of interest, the pipeline downloads curated binary datasets (SMILES + activity label), optionally augments them with decoys, and trains QSAR models using [LazyQSAR](https://github.com/ersilia-os/lazy-qsar).

## Setup

Clone this repository and create a Conda environment:

```sh
git clone https://github.com/ersilia-os/chembl-antimicrobial-models.git
cd chembl-antimicrobial-models
conda create -n camm python=3.12 -y
conda activate camm
pip install -r requirements.txt
```

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
| 01 | `scripts/01_download_datasets.py` | Downloads representative binary datasets from `chembl-antimicrobial-tasks` outputs |
| 02 | `scripts/02_add_decoys.py` | Augments active-enriched datasets with ChEMBL decoys where needed |
| 03 | `scripts/03_train_models.py` | Trains LazyQSAR binary classifiers and saves ONNX models |

## Repository structure

```
chembl-antimicrobial-models/
├── data/
│   ├── raw/          # Downloaded datasets per pathogen
│   └── processed/    # Datasets ready for training (after decoy augmentation)
├── scripts/          # Pipeline scripts
├── notebooks/        # Exploratory notebooks
├── output/
│   ├── models/       # Trained LazyQSAR models (ONNX)
│   └── plots/        # Evaluation figures
├── src/              # Shared utility code
└── docs/             # Additional documentation
```

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
