from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


# Get all pathogens i.e. {pathogen}_{target}
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))

# Define some paths
PATH_TO_OUTPUT = os.path.join("..", "output", "05_correlations.tsv")
PATH_TO_PREDICTIONS = os.path.join("..", "output", "04_predictions_drugbank")

# Trained models
MODELS = ['NB', 'RF', 'FLAML']

PREDICTIONS = {}
ALL_MODELS = []

for pathogen in PATHOGENS:

    print(f"Getting predictions for PATHOGEN: {pathogen}")
    
    # Get list of tasks
    tasks = sorted(os.listdir(os.path.join("..", "data", pathogen)))

    # For each task
    for task in tasks:
        
        task = task.replace(".csv", "")
        
        # For each model
        for model in MODELS:

            ALL_MODELS.append(tuple([pathogen, task, model]))
            
            try:
            
                # Load and save predictions
                preds = np.load(os.path.join(PATH_TO_PREDICTIONS, pathogen, task, f"{model}.npy"))
                PREDICTIONS[(pathogen, task, model)] = preds

            except:

                pass


CORRELATIONS = []
count = 0

for pat1, task1, model1 in tqdm(ALL_MODELS[:]):

    # Get predictions
    try:
        preds1 = np.array(PREDICTIONS[(pat1, task1, model1)])
    except:
        preds1 = [np.nan] * 11915
    
    for pat2, task2, model2 in ALL_MODELS[count:]:

        # Get same info
        same_pathogen = pat1 == pat2
        same_task = task1 == task2
        same_model = model1 == model2

        # Get predictions
        try:
            preds2 = np.array(PREDICTIONS[(pat2, task2, model2)])
        except:
            preds2 = [np.nan] * 11915

        # Calculate correlations
        spearman = spearmanr(preds1, preds2)
        pearson = pearsonr(preds1, preds2)
        kendall = kendalltau(preds1, preds2)

        CORRELATIONS.append([pat1, task1, model1, pat2, task2, model2, spearman.statistic, spearman.pvalue, pearson.statistic, pearson.pvalue, 
                             kendall.statistic, kendall.pvalue, same_pathogen, same_task, same_model])


    count += 1

CORRELATIONS = pd.DataFrame(CORRELATIONS, columns=['Pathogen1', 'Task1', 'Model1', 'Pathogen2', 'Task2', 'Model2', 'Spearman statistic', 'Spearman pvalue',
                                                   'Pearson statistic', 'Pearson pvalue', 'Kendall statistic', 'Kendall pvalue', 'Same pathogen', 'Same task', 'Same model'])

CORRELATIONS.to_csv(os.path.join(PATH_TO_OUTPUT), sep='\t', index=False)