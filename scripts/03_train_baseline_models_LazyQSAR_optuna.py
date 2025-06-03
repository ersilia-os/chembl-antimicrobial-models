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
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))

# Define some paths
PATH_TO_FEATURES = os.path.join("..", "output", "02_features")
PATH_TO_OUTPUT = os.path.join("..", "output", "03_baseline_models")

for pathogen in PATHOGENS:

    print(f"----------------------- PATHOGEN: {pathogen} ---------------------------")

    # Get list of tasks
    tasks = sorted(os.listdir(os.path.join("..", "data", pathogen)))

    # For each task
    for task in tasks:

        # if task is not done yet
        if os.path.exists(os.path.join(PATH_TO_OUTPUT, pathogen, task.replace(".csv", ""), "LQ_optuna_CV.csv")) == True:
            continue

        print(f"TASK: {task}")

        # Create output_dir
        output_dir = os.path.join(PATH_TO_OUTPUT, pathogen, task.replace(".csv", ""))
        os.makedirs(output_dir, exist_ok=True)

        # Load those IKs that have been processed in previous steps
        IKs = open(os.path.join(PATH_TO_FEATURES, pathogen, 'IKS.txt')).read().splitlines()

        # Load data
        df = pd.read_csv(os.path.join("..", "data", pathogen, task))
        cols = df.columns.tolist()
        X_smiles, Y = [], []
        for IK, smiles, act in zip(df['inchikey'], df['smiles'], df[cols[2]]):
            if IK in IKs:
                X_smiles.append(smiles)
                Y.append(act)

        # To np.array
        X_smiles = np.array(X_smiles)
        Y = np.array(Y)

        # # Subsample
        # indices = np.random.choice(len(X_smiles), size=100, replace=False)
        # X_smiles = X_smiles[indices]
        # Y = Y[indices]

        # Fit model with all data
        model_all = lq.LazyBinaryQSAR(descriptor_type='morgan', model_type="xgboost")
        model_all.fit(X_smiles, Y)
        model_all.save_model(model_dir=os.path.join(PATH_TO_OUTPUT, pathogen, task.replace(".csv", ""), "LQ_optuna"))

        # Cross-validations
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aurocs = []
        for train_index, test_index in skf.split(X_smiles, Y):
            X_train, X_test = X_smiles[train_index], X_smiles[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            model_cv = lq.LazyBinaryQSAR(descriptor_type='morgan', model_type="xgboost")
            model_cv.fit(X=X_train, y=y_train)
            fpr, tpr, _ = roc_curve(y_test, model_cv.predict_proba(X_test))
            auroc = auc(fpr, tpr)
            aurocs.append(auroc)


        # Save AUROC CVs
        with open(os.path.join(PATH_TO_OUTPUT, pathogen, task.replace(".csv", ""), "LQ_optuna_CV.csv"), "w") as f:
            f.write(",".join([str(round(i, 4)) for i in aurocs]))