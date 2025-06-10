from rdkit.Chem import rdFingerprintGenerator
from io import TextIOWrapper
from rdkit import Chem
import pandas as pd
import numpy as np
import joblib
import zipfile
from tqdm import tqdm
import os

# Suppress rdkit logs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Get all pathogens i.e. {pathogen}_{target}
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))[:1]

# Define some paths
PATH_TO_OUTPUT = os.path.join("..", "output", "06_features_ChEMBL")
os.makedirs(PATH_TO_OUTPUT, exist_ok=True)

# Read ChEMBL data
print("Reading ChEMBL data...")
zip_path = '../other/ChEMBL/ChEMBL_35_compounds_tsv.zip'
dfs = []
with zipfile.ZipFile(zip_path, 'r') as z:
    for idx,f in enumerate(z.namelist()):
        with z.open(f) as file:
            if idx == 0:
                df = pd.read_csv(TextIOWrapper(file, 'utf-8'), on_bad_lines='skip', sep='\t')
            else:
                df = pd.read_csv(TextIOWrapper(file, 'utf-8'), on_bad_lines='skip', header=None, names=dfs[0].columns, sep='\t')
            df['source_file'] = f
            dfs.append(df)
dfs = pd.concat(dfs, ignore_index=True)
print(f"{len(dfs)} processed compounds")

# Selecting small molecules
print("Selecting only small molecules...")
dfs = dfs[(dfs['Type'] == 'Small molecule') & (dfs['Smiles'].isna() == False)].reset_index(drop=True)
print(f"{len(dfs)} selected small molecules")
chembl_smiles = sorted(dfs['Smiles'])
N = 500000
np.random.seed(42)
indices = np.random.choice(len(chembl_smiles), size=N, replace=False)
print(f"Subsampling {len(indices)} small molecules...")
chembl_smiles = np.array(chembl_smiles)[indices]


# Calculate Morgan Fingerprints
print("Calculating Morgan Fingerprints")
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
X = []
for smiles in tqdm(chembl_smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mfp = mfpgen.GetCountFingerprint(mol)
        X.append(mfp.ToList())
    except:
        pass

# Convert to numpy array
X = np.array(X, dtype=np.int16)

# Save the fingerprints
np.save(os.path.join(PATH_TO_OUTPUT, "ChEMBL.npy"), X)