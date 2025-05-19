import os

# Get all pathogens i.e. {pathogen}_{target}
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))

# Define some paths
FEATURES = sorted(os.listdir(os.path.join("..", "output", "02_features")))
PATH_TO_OUTPUT = os.path.join("..", "output", "03_baseline_models")