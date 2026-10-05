import os

# Column names
COL_SMILES = "smiles"
COL_CANONICAL_SMILES = "canonical_smiles"
COL_BIN = "bin"
COL_INCHIKEY = "inchikey"
COL_FOUND_IN = "found_in"
COL_DECOY = "decoy"

# Reproducibility
RANDOM_SEED = 42

# Decoy generation (model eos3e6s)
DECOY_MODEL = "eos3e6s"
SPLIT_SIZE = 500
N_DECOYS = 20

# Dataset preparation thresholds
HIGH_RATIO_THRESHOLD = 0.5  # augment with decoys when active fraction exceeds this
TARGET_RATIO = 0.1  # target active fraction after augmentation

# ChEMBL archive and metadata filenames
CHEMBL_ZIP_FINAL = "19_final_datasets.zip"
CHEMBL_ZIP_GENERAL = "20_general_datasets_middle.zip"
CHEMBL_ZIP_GENERAL_NO_PUBCHEM = "20_general_no_pubchem_datasets_middle.zip"
CHEMBL_ZIP_GENERAL_HIGH = "20_general_datasets_high.zip"
CHEMBL_ZIP_GENERAL_NO_PUBCHEM_HIGH = "20_general_no_pubchem_datasets_high.zip"
CHEMBL_CSV_GENERAL = "20_general_datasets.csv"
CHEMBL_CSV_GENERAL_NO_PUBCHEM = "20_general_no_pubchem_datasets.csv"

# Pathogens
PATHOGENS = [
    "abaumannii",
    "calbicans",
    "campylobacter",
    "ecoli",
    "efaecium",
    "enterobacter",
    "hpylori",
    "kpneumoniae",
    "mtuberculosis",
    "ngonorrhoeae",
    "paeruginosa",
    "pfalciparum",
    "saureus",
    "smansoni",
    "spneumoniae",
]

# Overall datasets
G_ORG_DR = ["IC50", "EC50", "IC90","MIC", "MIC50", "MIC80", "MIC90", "POTENCY"]
G_ORG_SP = ["INHIBITION", "ACTIVITY", "GI", "PERCENTEFFECT"]

# Model training
DESCRIPTORS = ["cddd", "chemeleon", "clamp", "morgan", "rdkit"]
N_FOLDS = 5
MIN_AUROC = 0.7  # minimum mean CV AUROC to retain a model in the pipeline

# Model quality weights (script 10a): each ramps linearly from 0 at its floor to 1 at 1.0
W_AUROC_FLOOR  = 0.7  # w2 — mean CV AUROC
W_SCREEN_FLOOR = 0.7  # w_screen — screening AUC (LazyQSAR >= 3.6): P(an out-of-fold active
                      # outranks a molecule of the fixed 50K drug-like reference library)

# The model-level quality weights, as columns of 10_reports.csv. 10a writes them and takes their
# mean as final_weight; 14 averages them (plus the per-compound cutoff ramp w7) into the consensus
# weight; 18b ships them. The shipped consensus.py keeps its own literal copy (it runs outside this
# repo); 18b is to assert that the two are equal.
QUALITY_WEIGHT_COLS = ["w1", "w2", "w3", "w4", "w5", "w6", "w_screen"]

# DrugBank filtering
MW_CAP = 1000.0

# Recapitulation thresholds
THRESHOLDS = [0.001, 0.01, 0.05]
THRESHOLD_SFXS = ["0.1pct", "1pct", "5pct"]

# Quality check flags (script 17)
FOLD_UNSTABLE_AUROC_STD = 0.05   # flag models with cross-fold auroc_std above this
LOW_WEIGHT_THRESHOLD    = 0.3    # flag models with final_weight below this

# Ersilia Hub refresh (scripts 18a, 18b, 19)
HUB_CLONES_DIR  = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chembl-models-tmp"))
                                                      # one clone per pathogen, {HUB_CLONES_DIR}/{eosXXXX}, next to this repo
HUB_URL         = "https://github.com/ersilia-os/{eos_id}.git"   # the remote the existing clones already use
HUB_BRANCH      = "main"                              # the branch a refresh is committed to and pushed to
HUB_RUNTIME_ENV = "cam-models-runtime"                # conda env in which 18b runs the model once (lazyqsar must match the pin in 18b)

ERSILIA_MODEL_IDS = {
    "abaumannii":"eos21dr",
    "calbicans":"eos8jx6",
    "campylobacter":"eos7iak",
    "ecoli":"eos5eya",
    "efaecium":"eos81zy",
    "enterobacter":"eos9bpi",
    "hpylori":"eos9eyo",
    "kpneumoniae":"eos6wb7",
    "mtuberculosis":"eos43d6",
    "ngonorrhoeae":"eos5qya",
    "paeruginosa":"eos2e3s",
    "pfalciparum":"eos4an7",
    "saureus":"eos8lcw",
    "smansoni":"eos8v1a",
    "spneumoniae":"eos5q52",
}