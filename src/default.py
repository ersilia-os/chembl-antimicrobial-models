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

# DrugBank filtering
MW_CAP = 1000.0

# Consensus scoring — tanh fit parameters (IQR-shrinking sigmoid)
TANH_A = 1.156
TANH_TAU = 6.47

# Recapitulation thresholds
THRESHOLDS = [0.001, 0.01, 0.05]
THRESHOLD_SFXS = ["0.1pct", "1pct", "5pct"]

# Quality check flags (script 17)
FOLD_UNSTABLE_AUROC_STD = 0.05   # flag models with cross-fold auroc_std above this
LOW_WEIGHT_THRESHOLD    = 0.3    # flag models with final_weight below this


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