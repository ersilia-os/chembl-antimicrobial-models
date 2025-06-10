import pandas as pd
import shutil
import tqdm
import os
import re

# Get all pathogens i.e. {pathogen}_{target}
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))

# Ersilia models columns' do not allow special characters apart from underscores
def clean_string_regex(input_string):
    return re.sub(r'[^a-zA-Z0-9_]', '', input_string)

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

    # Copy Summary report
    PATH_TO_CAMT = f"/home/acomajuncosa/Documents/chembl-antimicrobial-tasks/output/{pathogen}/018_selected_tasks_FINAL.csv"
    shutil.copyfile(PATH_TO_CAMT, os.path.join(PATH_TO_OUTPUT, "checkpoints", "018_selected_tasks_FINAL.csv"))

    # Create run_columns.csv file
    run_columns = pd.DataFrame({
        "name": [clean_string_regex(i) for i in tasks],
        "type": ['float' for i in tasks],
        "direction": ['high' for i in tasks],
        "description": ["Predicted probability of being active according to task " + clean_string_regex(i) for i in tasks] # Caution with %s and special characters
    })
    run_columns.to_csv(os.path.join(PATH_TO_OUTPUT, "framework", "columns", "run_columns.csv"), index=False)

    # Greate run_input.csv file
    run_input = pd.DataFrame({
        "smiles": ["COc1ccc(\C=C\C(O)=O)cc1", "C[C@H](N[C@@H](CCc1ccccc1)C(O)=O)C(=O)N1CCC[C@H]1C(O)=O", "Cc1ccc(cc1C)N1CCN(Cc2nc3ccccc3[nH]2)CC1"]
    })
    run_input.to_csv(os.path.join(PATH_TO_OUTPUT, "framework", "examples", "run_input.csv"), index=False)

    # Create empty main.py file
    with open(os.path.join(PATH_TO_OUTPUT, "framework", "code", "main.py"), 'w') as f:
        f.write("# This is a placeholder for the main.py file.\n")

    # Create the run.sh file
    with open(os.path.join(PATH_TO_OUTPUT, "framework", "run.sh"), 'w') as f:
        f.write("python $1/code/main.py $2 $3\n")

    # FOr each task
    for task in tasks:

        # Copy the zsRF model
        shutil.copyfile(os.path.join(PATH_TO_DATA, task, "RF.joblib"), 
                        os.path.join(PATH_TO_OUTPUT, "checkpoints", clean_string_regex(task) + "_RF.joblib"))