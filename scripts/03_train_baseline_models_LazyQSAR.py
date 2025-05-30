from flaml.default import RandomForestClassifier as ZeroShotRandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from flaml.automl import AutoML
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
    IKs = open(os.path.join(PATH_TO_FEATURES, pathogen, 'IKS.txt')).read().splitlines()
    MFPs = np.load(os.path.join(PATH_TO_FEATURES, pathogen, "X.npz"))['X']
    IK_TO_MFP = {i: j for i, j in zip(IKs, MFPs)}

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
            if ik in IK_TO_MFP:
                X.append(IK_TO_MFP[ik])
                Y.append(act)

        # To np.array
        X = np.array(X)
        Y = np.array(Y)

        # print("Training NB...")

        # # Naive Bayes
        # NB, results_NB = NaiveBayesClassificationModel(X, Y)
        # joblib.dump(NB, os.path.join(output_dir, "NB.joblib"))
        # with open(os.path.join(output_dir, "NB_CV.csv"), "w") as f:
        #     f.write(",".join(results_NB))
