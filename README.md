# 🖥️ Antimicrobial binary ML models from ChEMBL 🤖

Train and validate antimicrobial classification models with bioactivity data gathered from ChEMBL. This repository is the subsequent continuation of [chembl-antimicrobial-tasks](https://github.com/ersilia-os/chembl-antimicrobial-tasks). 

This repository is currently **WORK IN PROGRESS**. ⚠️🚧

## Setup 🛠️

To get started, first clone this repository, avoiding large LFS-stored files:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ersilia-os/chembl-antimicrobial-models.git
cd chembl-antimicrobial-models
```

We recommend creating a Conda environment to run this code. Dependencies are minimal. 🐍

```sh
conda create -n camm python=3.10
conda activate camm
pip install -r requirements.txt
```

### Computational pipeline and output 📊

The computational pipeline to train and validate antimicrobial classification models with ChEMBL bioactivity data consists of several well defined and consecutive steps. Each step corresponds to a given Python script (see next Section).

1. `01_download_input_data.py`: Downloading binary bioactivity datasets prepared in the [chembl-antimicrobial-tasks](https://github.com/ersilia-os/chembl-antimicrobial-tasks) repository. 

2. `02_featurize_compounds.py`: Characterizing small molecules with Morgan Fingerprints (Counts, radius 3, 2048 bits).

3. `03_train_baseline_models.py`: For each pathogen and antimicrobial classification task, train baseline models: Multinomial Naive Bayes Models and Random Forest models with zero-shot hyperparameter suggestion (**TabPFN - WIP**).

4. `04_predict_drugbank.py`: For each pathogen and antimicrobial classification model, predict classification scores for 10k DrugBank compounds. These predictions are meant to identify correlations among ML models. 

5. `05_calculate_correlations`: Calculate correlations among antimicrobial classification models (inter and intra pathogen comparisons).


## Repository architecture ⛩️

- `data`: Bioactivity data used to train ML models grouped by pathogen of interest. 
- `notebooks`: Notebooks used to analyze the results.  
- `other`: Additional data e.g. DrugBank small molecules used as reference compound set. 
- `output`: Main output of each pipeline step.
- `plots`: Plots and analyses. 
- `scripts`: Scripts to perform each step of the computational pipeline. 



## TL;DR 🚩

Bla bla

## About the Ersilia Open Source Initiative 🌍🤝

This repository is developed by the [Ersilia Open Source Initiative](https://ersilia.io). Ersilia develops AI/ML tools to support drug discovery research in the Global South. To learn more about us, please visit our [GitBook Documentation](https://ersilia.gitbook.io) and our [GitHub profile](https://github.com/ersilia-os/).

