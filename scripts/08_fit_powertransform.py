from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os

# Load ChEMBL features
X = np.load("../output/06_features_ChEMBL/ChEMBL.npy")

# Set paths
PATH_TO_MODELS = "../other/models"

# Get all pathogens i.e. {pathogen}_{target}
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))
PATHOGENS = ["abaumannii_organism", 'kpneumoniae_organism']

# For each pathogen
for pathogen in PATHOGENS:

    print(f" --------------------------  Processing pathogen: {pathogen}  --------------------------- ")
    df = pd.DataFrame()

    # Read the run_columns file
    run_columns = pd.read_csv(os.path.join(PATH_TO_MODELS, f"model_{pathogen}", "framework", "columns", "run_columns.csv"))['name'].tolist()
    run_columns = [i for i in run_columns if i != "0_consensus_score"]  # Exclude consensus score

    # For each model
    for model in run_columns:

        print(f"Predicting model {model} ...")

        # Load the model
        model_joblib = joblib.load(os.path.join(PATH_TO_MODELS, f"model_{pathogen}", 'checkpoints', model + "_RF.joblib"))
        df[model] = model_joblib.predict_proba(X)[:, 1]

    print("Fitting PowerTransformer...")

    # Get probabilities
    pt = PowerTransformer()
    data = df.copy()

    # Fitting
    pt.fit(data)

    # Saving the PowerTransformer
    joblib.dump(pt, os.path.join(PATH_TO_MODELS, f"model_{pathogen}", 'checkpoints', "RF_PowerTransformer.joblib"))