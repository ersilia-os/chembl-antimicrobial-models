from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from flaml.automl import AutoML
from collections import Counter
import lazyqsar as lq
import pandas as pd
import numpy as np
import collections
import joblib
import os

# Get all pathogens i.e. {pathogen}_{target}
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))[:1]

# Define some paths
PATH_TO_FEATURES = os.path.join("..", "output", "02_features")
PATH_TO_OUTPUT = os.path.join("..", "output", "03_baseline_models")

for pathogen in PATHOGENS:

    print(f"----------------------- PATHOGEN: {pathogen} ---------------------------")

    # Get list of tasks
    tasks = sorted(os.listdir(os.path.join("..", "data", pathogen)))

    # # Get IK to MFP
    # IKs = open(os.path.join(PATH_TO_FEATURES, pathogen, 'IKS.txt')).read().splitlines()
    # MFPs = np.load(os.path.join(PATH_TO_FEATURES, pathogen, "X.npz"))['X']
    # IK_TO_MFP = {i: j for i, j in zip(IKs, MFPs)}

    # For each task
    for task in tasks:

        print(f"TASK: {task}")

        # Create output_dir
        output_dir = os.path.join(PATH_TO_OUTPUT, pathogen, task.replace(".csv", ""))
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        df = pd.read_csv(os.path.join("..", "data", pathogen, task))
        cols = df.columns.tolist()
        X_smiles, Y = [], []
        for smiles, act in zip(df['smiles'], df[cols[2]]):
            X_smiles.append(smiles)
            Y.append(act)

        # To np.array
        X_smiles = np.array(X_smiles)
        Y = np.array(Y)

        # Subsample
        indices = np.random.choice(len(X_smiles), size=500, replace=False)
        X_smiles = X_smiles[indices]
        Y = Y[indices]

        # Cross-validations
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aurocs = []
        for train_index, test_index in skf.split(X_smiles, Y):
            X_train, X_test = X_smiles[train_index], X_smiles[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            # Available descriptors: morgan, mordred, rdkit, classic, maccs
            # Available models: xgboost
            model_cv = lq.LazyBinaryQSAR(descriptor_type='morgan', model_type="xgboost")
            model_cv.fit(X=X_train, y=y_train)
            # model.save_model(model_dir="my_model")
            fpr, tpr, _ = roc_curve(y_test, model_cv.predict_proba(X_test))
            auroc = auc(fpr, tpr)
            aurocs.append(auroc)


        break
