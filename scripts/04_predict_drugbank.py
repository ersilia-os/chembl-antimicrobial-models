from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
import pandas as pd
import numpy as np
import joblib
import os

# Suppress rdkit logs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# Get all pathogens i.e. {pathogen}_{target}
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))

# Define some paths
PATH_TO_OUTPUT = os.path.join("..", "output", "04_predictions_drugbank")
PATH_TO_MODELS = os.path.join("..", "output", "03_baseline_models")

# Get Drugbank Smiles
drugbank_smiles = pd.read_csv(os.path.join("../other/DrugBank/drugbank_smiles.csv"))['Smiles'].tolist()
drugbank_smiles = sorted(drugbank_smiles)

# Calculate Morgan Fingerprints
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
X = []
for smiles in drugbank_smiles:
    try:
        mol = Chem.MolFromSmiles(smiles)
        mfp = mfpgen.GetCountFingerprint(mol)
        X.append(mfp.ToList())
    except:
        pass

# Convert to numpy array
X = np.array(X, dtype=np.int16)

# Trained models
MODELS = ['NB', 'RF']#, 'FLAML']

for pathogen in PATHOGENS:

    print(f"Predicting DrugBank scores for PATHOGEN: {pathogen}")

    # Get list of tasks
    tasks = sorted(os.listdir(os.path.join("..", "data", pathogen)))

    # For each task
    for task in tasks:

        # For each model
        for model in MODELS:

            try:

                # Load model and predict DrugBank
                model_name = model
                model = joblib.load(os.path.join(PATH_TO_MODELS, pathogen, task.replace(".csv", ""), f"{model}.joblib"))
                predictions = model.predict_proba(X)[:,1]

                # Save predictions
                os.makedirs(os.path.join(PATH_TO_OUTPUT, pathogen, task.replace(".csv", "")), exist_ok=True)
                np.save(os.path.join(PATH_TO_OUTPUT, pathogen, task.replace(".csv", ""), f"{model_name}.npy"), predictions)

            except:

                pass