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


def NaiveBayesClassificationModel(X, Y, n_folds=5):

    # Fit model with all data
    model_all = MultinomialNB()
    model_all.fit(X, Y)

    # Cross-validations
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aurocs = []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        model_cv = MultinomialNB()
        model_cv.fit(X_train, y_train)
        fpr, tpr, _ = roc_curve(y_test, model_cv.predict_proba(X_test)[:, 1])
        auroc = auc(fpr, tpr)
        aurocs.append(auroc)

    return model_all, [str(round(i, 4)) for i in aurocs]


def RandomForestClassificationModel(X, Y, n_folds=5):

    # Fit model with all data
    zero_shot = ZeroShotRandomForestClassifier()
    hyperparams = zero_shot.suggest_hyperparams(X, Y)[0]
    hyperparams['n_jobs'] = 8
    # hyperparams['n_estimators'] = 10
    # print("Training full model...")
    model_all = RandomForestClassifier(**hyperparams)
    model_all.fit(X, Y)

    # Cross-validations
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aurocs = []
    for train_index, test_index in skf.split(X, Y):
        # print("Training CV model...")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        zero_shot_cv = ZeroShotRandomForestClassifier()
        hyperparams = zero_shot_cv.suggest_hyperparams(X_train, y_train)[0]
        hyperparams['n_jobs'] = 8
        # hyperparams['n_estimators'] = 10
        model_cv = RandomForestClassifier(**hyperparams)
        model_cv.fit(X_train, y_train)
        fpr, tpr, _ = roc_curve(y_test, model_cv.predict_proba(X_test)[:, 1])
        auroc = auc(fpr, tpr)
        aurocs.append(auroc)

    return model_all, [str(round(i, 4)) for i in aurocs]


# def FLAMLClassificationModel(X, Y, N_FOLDS=5, SPLITTING_ROUNDS=3):

#     # Constants for AutoML
#     N_FOLDS = 5 #5
#     SPLITTING_ROUNDS = 1 #3
#     FLAML_ESTIMATOR_LIST = ["rf"]
#     FLAML_COLD_MINIMUM_TIME_BUDGET_SECONDS = 250
#     FLAML_COLD_MAXIMUM_TIME_BUDGET_SECONDS = 500
#     FLAML_WARM_MINIMUM_TIME_BUDGET_SECONDS = 50
#     FLAML_WARM_MAXIMUM_TIME_BUDGET_SECONDS = 250
#     FLAML_COLD_MINIMUM_ITERATIONS = 50
#     FLAML_COLD_MAXIMUM_ITERATIONS = 250
#     FLAML_WARM_MINIMUM_ITERATIONS = 25
#     FLAML_WARM_MAXIMUM_ITERATIONS = 100

#     # Clean up log files function
#     def clean_log(log_file_name):
#         cwd = os.getcwd()
#         log_file = os.path.join(cwd, log_file_name)
#         if os.path.exists(log_file):
#             os.remove(log_file)

#     # Get AutoML settings based on data size
#     def get_automl_settings(time_budget, num_samples):
#         return {
#             "time_budget": int(
#                 np.clip(
#                     time_budget,
#                     FLAML_COLD_MINIMUM_TIME_BUDGET_SECONDS,
#                     FLAML_COLD_MAXIMUM_TIME_BUDGET_SECONDS,
#                 )
#             ),
#             "metric": "roc_auc",
#             "task": "classification",
#             "log_file_name": "flaml.log",
#             "log_training_metric": True,
#             "verbose": 0,
#             "n_jobs": 8,
#             "early_stop": True,
#             "max_iter": int(
#                 np.clip(
#                     num_samples / 3,
#                     FLAML_COLD_MINIMUM_ITERATIONS,
#                     FLAML_COLD_MAXIMUM_ITERATIONS,
#                 )
#             ),
#             "estimator_list": FLAML_ESTIMATOR_LIST
#         }
    
#     # Extract best configuration from a model
#     def get_starting_point(model):
#         best_models = model.best_config_per_estimator
#         best_estimator = model.best_estimator
#         return {best_estimator: best_models[best_estimator]}

#     # Initial "cold" fitting phase - broad search
#     def fit_cold(X, y):
#         # print("Starting cold fitting phase...")
#         automl_settings = get_automl_settings(
#             time_budget=(FLAML_COLD_MINIMUM_TIME_BUDGET_SECONDS + FLAML_COLD_MAXIMUM_TIME_BUDGET_SECONDS) / 2,
#             num_samples=len(y)
#         )
#         model = AutoML(n_jobs=8)
#         model.fit(
#             X_train=X, 
#             y_train=y, 
#             eval_method="auto",
#             split_type=None,
#             groups=None,
#             **automl_settings
#         )
#         # print(f"Cold fitting complete. Best estimator: {model.best_estimator}")
        
#         # Proceed to warm fitting
#         warm_model = fit_warm(X, y, model, automl_settings)
#         clean_log(automl_settings["log_file_name"])
#         return warm_model, automl_settings

#      # Refined "warm" fitting phase - focus on best model
    
#     # Refined "warm" fitting phase - focus on best model
#     def fit_warm(X, y, cold_model, automl_settings):
#         # print("Starting warm fitting phase...")
#         warm_settings = automl_settings.copy()
#         warm_settings["time_budget"] = int(
#             np.clip(
#                 warm_settings["time_budget"] * 0.5,
#                 FLAML_WARM_MINIMUM_TIME_BUDGET_SECONDS,
#                 FLAML_WARM_MAXIMUM_TIME_BUDGET_SECONDS,
#             )
#         )
#         warm_settings["log_file_name"] = "warm_flaml.log"
#         warm_settings["estimator_list"] = [cold_model.best_estimator]
#         warm_settings["max_iter"] = int(
#             np.clip(
#                 int(warm_settings["max_iter"] * 0.5) + 1,
#                 FLAML_WARM_MINIMUM_ITERATIONS,
#                 FLAML_WARM_MAXIMUM_ITERATIONS,
#             )
#         )
        
#         starting_point = get_starting_point(cold_model)
#         model = AutoML(n_jobs=8)
#         model.fit(
#             X_train=X, 
#             y_train=y, 
#             starting_points=starting_point,
#             eval_method="auto",
#             split_type=None, 
#             groups=None,
#             **warm_settings
#         )
#         clean_log(warm_settings["log_file_name"])
#         # print("Warm fitting complete.")
#         return model

#     # Cross-validation prediction to get out-of-sample estimates
#     def fit_predict_cv(X, y, model, automl_settings):
#         # print(f"Performing {SPLITTING_ROUNDS} rounds of {N_FOLDS}-fold cross-validation...")
#         cv_settings = automl_settings.copy()
#         cv_settings["time_budget"] = int(
#             np.clip(
#                 cv_settings["time_budget"] * 0.5,
#                 FLAML_WARM_MINIMUM_TIME_BUDGET_SECONDS,
#                 FLAML_WARM_MAXIMUM_TIME_BUDGET_SECONDS,
#             )
#         )
        
#         best_estimator = model.best_estimator
#         starting_point = get_starting_point(model)
        
#         # Create a dictionary to store results for each sample
#         results = collections.defaultdict(list)
#         aurocs = []
        
#         # Create stratified k-fold splitter
#         splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
        
#         # Loop through multiple splitting rounds
#         k = 0
#         for _ in range(SPLITTING_ROUNDS):
#             for train_index, test_index in splitter.split(X, y):
#                 # print(f"CV split {k+1}/{N_FOLDS*SPLITTING_ROUNDS}")
                
#                 # Extract train and test data
#                 X_train, X_test = X[train_index], X[test_index]
#                 y_train, y_test = y[train_index], y[test_index]
                
#                 # Configure settings for this fold
#                 fold_settings = cv_settings.copy()
#                 fold_settings["log_file_name"] = f"oos_{k}_automl.log"
#                 fold_settings["estimator_list"] = [best_estimator]
#                 fold_settings["max_iter"] = int(
#                     np.clip(
#                         int(fold_settings["max_iter"] * 0.25) + 1,
#                         FLAML_WARM_MINIMUM_ITERATIONS,
#                         FLAML_WARM_MAXIMUM_ITERATIONS,
#                     )
#                 )
#                 # Train a model on this fold
#                 fold_model = AutoML(n_jobs=8)
#                 fold_model.fit(
#                     X_train=X_train,
#                     y_train=y_train,
#                     starting_points=starting_point,
#                     eval_method="auto",
#                     split_type=None,
#                     groups=None,
#                     **fold_settings
#                 )
                
#                 # Get the base estimator
#                 base_model = fold_model.model.estimator
                
#                 # Try to calibrate the model for better probability estimates
#                 # print("Fitting a calibrated model for better probability estimates")
#                 try:
#                     calibrated_model = CalibratedClassifierCV(estimator=base_model, n_jobs=8)
#                     calibrated_model.fit(X_train, y_train, n_jobs=8)
#                     fold_estimator = calibrated_model
#                 except Exception as e:
#                     # print(f"Could not calibrate model: {e}. Continuing with uncalibrated model...")
#                     fold_estimator = base_model
                
#                 # Get probabilistic predictions for test data
#                 y_pred_proba = fold_estimator.predict_proba(X_test)[:, 1]
                
#                 # # Store predictions for each test sample
#                 # for i, idx in enumerate(test_index):
#                 #     results[idx].append(y_pred_proba[i])
#                 aurocs.append(roc_auc_score(y_test, y_pred_proba))
                
#                 # Clean up logs
#                 clean_log(fold_settings["log_file_name"])
#                 k += 1
        
#         # # Average predictions for each sample
#         # y_hat = []
#         # for i in range(len(y)):
#         #     y_hat.append(np.mean(results[i]))
        
#         # return {"y_hat": np.array(y_hat), "y": y}
#         # return results
#         return aurocs

#     # ---- Main Training Process ----
    
#     # 1. Train cold model (which gets refined with warm fitting)
#     model, automl_settings = fit_cold(X, Y)
    
#     # 2. Perform cross-validation to get out-of-sample predictions
#     cv_results = fit_predict_cv(X, Y, model, automl_settings)
    
#     # 3. Train final model on all data
#     # print("Training final calibrated model on all data...")
#     try:
#         final_model = CalibratedClassifierCV(model.model.estimator, n_jobs=8)
#         final_model.fit(X, Y, n_jobs=8)
#     except Exception as e:
#         # print(f"Could not calibrate final model: {e}. Using uncalibrated model...")
#         final_model = model.model.estimator

#     return final_model, [str(round(i, 4)) for i in cv_results]


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

        print("Training RF...")

        # Random Forest
        RF, results_RF = RandomForestClassificationModel(X, Y)
        joblib.dump(RF, os.path.join(output_dir, "RF.joblib"))
        with open(os.path.join(output_dir, "RF_CV.csv"), "w") as f:
            f.write(",".join(results_RF))

        # print("Training FLAML...")

        # # FLAML
        # FLAML, results_FLAML = FLAMLClassificationModel(X, Y)
        # joblib.dump(FLAML, os.path.join(output_dir, "FLAML.joblib"))
        # with open(os.path.join(output_dir, "FLAML_CV.csv"), "w") as f:
        #     f.write(",".join(results_FLAML))
