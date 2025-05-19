from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Get all pathogens i.e. {pathogen}_{target}
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))

# Define some paths
PATH_TO_OUTPUT = os.path.join("..", "output", "02_features")

for pathogen in PATHOGENS:

    print(f"Featurizing {pathogen}...")

    # Get all compounds: dict ik --> smiles
    ik_to_smiles = {}
    for task in sorted(os.listdir(os.path.join("..", "data", pathogen))):
        df = pd.read_csv(os.path.join("..", "data", pathogen, task))
        for ik, smi in zip(df['inchikey'], df['smiles']):
            if ik not in ik_to_smiles:
                ik_to_smiles[ik] = smi

    # Get all features
    IKS, X = [], []
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    for IK in tqdm(sorted(ik_to_smiles)):
        try:
            smi = ik_to_smiles[IK]
            mol = Chem.MolFromSmiles(smi)
            mfp = mfpgen.GetCountFingerprint(mol)
            X.append(mfp.ToList())
            IKS.append(IK)
        except:
            pass

    # Convert to numpy array
    X = np.array(X, dtype=np.int16)

    # Save results
    os.makedirs(os.path.join(PATH_TO_OUTPUT, pathogen), exist_ok=True)
    np.savez_compressed(os.path.join(PATH_TO_OUTPUT, pathogen, "X.npz"), X=X)
    with open(os.path.join(PATH_TO_OUTPUT, pathogen, "IKS.txt"), "w") as f:
        for ik in IKS:
            f.write(f"{ik}\n")

    # Assert that the number of IKs and MFPs are the same
    assert len(IKS) == X.shape[0], f"ERROR: Number of IKs ({len(IKS)}) does not match number of MFPs ({X.shape[0]})"