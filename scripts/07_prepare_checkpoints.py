import pandas as pd
import shutil
import joblib
import tqdm
import os
import re

# Get all pathogens i.e. {pathogen}_{target}
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))

# Ersilia models columns' do not allow special characters apart from underscores
def clean_string_regex(input_string):
    return re.sub(r'[^a-zA-Z0-9_]', '', input_string).lower()

for pathogen in tqdm.tqdm(PATHOGENS):

    # Defin path to outputs
    PATH_TO_OUTPUT = os.path.join("..", "other", "models", f"model_{pathogen}")  # Modidy as needed
    os.makedirs(os.path.join(PATH_TO_OUTPUT, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_OUTPUT, 'framework/code'), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_OUTPUT, 'framework/columns'), exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_OUTPUT, 'framework/examples'), exist_ok=True)

    # Define path to data
    PATH_TO_DATA = os.path.join("..", "output", "03_baseline_models", f"{pathogen}")
    tasks = sorted(os.listdir(PATH_TO_DATA))

    # Load dataset report
    PATH_TO_CAMT = f"/home/acomajuncosa/Documents/chembl-antimicrobial-tasks/output/{pathogen}/018_selected_tasks_FINAL.csv"
    report = pd.read_csv(PATH_TO_CAMT)

    # Create run_columns.csv file
    run_columns = pd.DataFrame({
        "name": ["0_consensus_score"] + [clean_string_regex(i) for i in tasks],
        "type": ['float'] + ['float' for i in tasks],
        "direction": ['high'] + ['high' for i in tasks],
        "description": ['Consensus score among datasets'] + ["Predicted probability of being active according to task " + 
                                                             clean_string_regex(i) for i in tasks] # Caution with %s and special characters
    })
    run_columns.to_csv(os.path.join(PATH_TO_OUTPUT, "framework", "columns", "run_columns.csv"), index=False)

    # Greate run_input.csv file
    run_input = pd.DataFrame({
        "smiles": ["COc1ccc(\C=C\C(O)=O)cc1", "C[C@H](N[C@@H](CCc1ccccc1)C(O)=O)C(=O)N1CCC[C@H]1C(O)=O", "Cc1ccc(cc1C)N1CCN(Cc2nc3ccccc3[nH]2)CC1"]
    })
    run_input.to_csv(os.path.join(PATH_TO_OUTPUT, "framework", "examples", "run_input.csv"), index=False)

    # Copy the main.py file from template
    shutil.copyfile(os.path.join("..", "other", "templates", "main.py", ), 
                     os.path.join(PATH_TO_OUTPUT, "framework", "code", "main.py"))

    # Create the run.sh file
    with open(os.path.join(PATH_TO_OUTPUT, "framework", "run.sh"), 'w') as f:
        f.write("python $1/code/main.py $2 $3\n")

    # Load and filter correlations
    correlations = pd.read_csv(os.path.join("../output/05_correlations/05_correlations.tsv"), sep='\t')
    correlations_pathogen = correlations[(correlations['Pathogen1'] == pathogen) & 
                                         (correlations['Pathogen2'] == pathogen) &
                                         (correlations['Model1'] == "RF") & 
                                         (correlations['Model2'] == "RF") & 
                                         (correlations['Same task'] == False)].reset_index(drop=True)
    
    # Save correlations
    # correlations.to_csv(os.path.join(PATH_TO_OUTPUT, "checkpoints", f"correlations_{pathogen}.tsv"), sep='\t', index=False)

    task_to_auroc= {}
    task_to_priority= {}
    task_to_positives= {}
    task_to_max_correlation = {}

    # Get dataset metadata
    for task, auroc_avg_MOD, num_pos_samples, auroc_avg_DIS, priority, selected in zip(report["task"], 
                                                                                    report["auroc_avg_MOD"], 
                                                                                    report["num_pos_samples"], 
                                                                                    report["auroc_avg_DIS"], 
                                                                                    report["priority"], 
                                                                                    report["SELECTED"]):

        # Clean task name for Ersilia nomenclature
        task_label = task + "_" + str(selected)
        task = clean_string_regex(task) + "_" + str(selected)

        # Load results
        task_to_priority[task] = priority
        task_to_positives[task] = num_pos_samples
        if auroc_avg_MOD > 0.7:
            task_to_auroc[task] = auroc_avg_MOD
        else:
            task_to_auroc[task] = auroc_avg_DIS
        
        
        # Get task to max correlation
        corr = correlations_pathogen[(correlations_pathogen['Same task'] == False) & ((correlations_pathogen['Task1'] == task_label) | 
                                    (correlations_pathogen['Task2'] == task_label))].sort_values(by='Spearman statistic', ascending=False)
        task_to_max_correlation[task] = max(corr['Spearman statistic'])

    # Calculate weights for each task
    weights = {}

    # For each task
    for task in tasks:

        # Copy the zsRF model
        shutil.copyfile(os.path.join(PATH_TO_DATA, task, "RF.joblib"), 
                        os.path.join(PATH_TO_OUTPUT, "checkpoints", clean_string_regex(task) + "_RF.joblib"))
        
        task = clean_string_regex(task)
        
        # Weights 1 and 2
        w1 = max(0, (task_to_auroc[task] - 0.5) / 0.5)
        w2 = 1 - (task_to_priority[task]-1)/6

        # Weight 3
        if task_to_positives[task] > 1000:
            w3 = 1
        elif task_to_positives[task] > 500:
            w3 = 0.75
        elif task_to_positives[task] > 100:
            w3 = 0.5
        else:
            w3 = 0.25

        # Weight 4
        if task_to_max_correlation[task] > 0.5:
            w4 = 0.5
        else:
            w4 = 1

        # Calculate final weight
        w = (w1 + (w2+w3+w4)/3) / 2
        weights[task] = round(w, 4)

    # Save the weights
    joblib.dump(weights, os.path.join(PATH_TO_OUTPUT, "checkpoints", "weights_datasets.joblib"))