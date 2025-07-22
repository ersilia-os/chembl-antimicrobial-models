from sklearn.model_selection import StratifiedKFold
import lazyqsar
import pandas as pd
import numpy as np
import collections
import joblib
import os

# Get all pathogens i.e. {pathogen}_{target}
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))

# Define some paths
PATH_TO_FEATURES = os.path.join("..", "output", "02_features")
PATH_TO_OUTPUT = os.path.join("..", "output", "03_baseline_models")


for pathogen in PATHOGENS:

    print(f"----------------------- PATHOGEN: {pathogen} ---------------------------")

    # Get list of tasks
    tasks = sorted(os.listdir(os.path.join("..", "data", pathogen)))

    # Get IK to MFP
    IKs = open(os.path.join(PATH_TO_FEATURES, pathogen, 'IKs_CheMeleon.txt')).read().splitlines()
    CheMeleons = np.load(os.path.join(PATH_TO_FEATURES, pathogen, "X_CheMeleon.npz"))['X']
    IK_TO_CHEMELEONS = {i: j for i, j in zip(IKs, CheMeleons)}

    # For each task
    for task in tasks:

        print(f"TASK: {task}")

        # Create output_dir
        output_dir = os.path.join(PATH_TO_OUTPUT, pathogen, task.replace(".csv", ""))
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        df = pd.read_csv(os.path.join("..", "data", pathogen, task))
        cols = df.columns.tolist()
        X, Y = [], []
        for ik, act in zip(df['inchikey'], df[cols[2]]):
            if ik in IK_TO_CHEMELEONS:
                X.append(IK_TO_CHEMELEONS[ik])
                Y.append(act)

        # To np.array
        X = np.array(X)
        Y = np.array(Y)

        print(X.shape)
        print(X)

        model = lazyqsar.LazyBinaryClassifier(model_type = "random_forest")
        model.fit(X[:5000], Y[:5000])
        preds = model.predict_proba(X)
        print(preds)

        break
    break
