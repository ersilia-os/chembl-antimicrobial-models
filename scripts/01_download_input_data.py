import os
import subprocess
import zipfile

def clone_repo_if_not_exists(repo_url, clone_dir):
    if not os.path.exists(clone_dir):
        subprocess.run(["git", "clone", repo_url, clone_dir])

URL_TO_REPOSITORY = "https://github.com/ersilia-os/chembl-antimicrobial-tasks"
PATH_TO_REPOSITORY = "/home/acomajuncosa/Documents/chembl-antimicrobial-tasks"
PATH_TO_DATA = "/home/acomajuncosa/Documents/chembl-antimicrobial-models/data"

# Clone the repository if it doesn't exist
clone_repo_if_not_exists(URL_TO_REPOSITORY, PATH_TO_REPOSITORY)

# Unzip and copy the data files
for output in sorted(os.listdir(os.path.join(PATH_TO_REPOSITORY, 'output'))):
    zippath = os.path.join(PATH_TO_REPOSITORY, 'output', output, f"{output}_tasks.zip")
    if os.path.isfile(zippath):
        with zipfile.ZipFile(zippath, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(PATH_TO_DATA, output))