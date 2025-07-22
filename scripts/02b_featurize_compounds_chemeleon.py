# CheMeleon needs Python 3.11 & chemprop 2.2.0
# please run this code within a conda environment having chemprop 2.2.0 and Python 3.11

from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from rdkit.Chem import MolFromSmiles, Mol
from urllib.request import urlretrieve
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.nn import RegressionFFN
from chemprop.models import MPNN
from pathlib import Path
import torch


class CheMeleonFingerprint:
    def __init__(self, device: str | torch.device | None = None):
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        root = os.path.dirname(os.path.abspath(__file__))
        # ckpt_dir = Path().home() / ".chemprop"
        # ckpt_dir.mkdir(exist_ok=True)
        # mp_path = ckpt_dir / "chemeleon_mp.pt"
        mp_path = Path(root) / ".." / "other" / "CheMeleon" / "chemeleon_mp.pt"
        mp_path = mp_path.resolve()
        if not mp_path.exists():
            urlretrieve(
                r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                mp_path,
            )
        chemeleon_mp = torch.load(mp_path, weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
        mp.load_state_dict(chemeleon_mp['state_dict'])
        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),  # not actually used
        )
        self.model.eval()
        if device is not None:
            self.model.to(device=device)

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        bmg = BatchMolGraph([self.featurizer(MolFromSmiles(m) if isinstance(m, str) else m) for m in molecules])
        bmg.to(device=self.model.device)
        return self.model.fingerprint(bmg).numpy(force=True)


# Get all pathogens i.e. {pathogen}_{target}
PATHOGENS = sorted(os.listdir(os.path.join("..", "data")))

# Define some paths
PATH_TO_OUTPUT = os.path.join("..", "output", "02_features")

for pathogen in PATHOGENS[1:]:

    print(f"Featurizing {pathogen}...")

    # Get all compounds: dict ik --> smiles
    ik_to_smiles = {}
    for task in sorted(os.listdir(os.path.join("..", "data", pathogen))):
        df = pd.read_csv(os.path.join("..", "data", pathogen, task))
        for ik, smi in zip(df['inchikey'], df['smiles']):
            if ik not in ik_to_smiles:
                ik_to_smiles[ik] = smi

    # Instantiate CheMeleong embeddings
    chemeleon_fingerprint = CheMeleonFingerprint()

    print("Calculating CheMeleon embeddings...")
    BATCH_SIZE = 1

    # Get all features
    ALL_IKs = sorted(ik_to_smiles)
    X, IKs = [], []
    for batch in tqdm(range(0, len(ik_to_smiles), BATCH_SIZE)):
        try:
            ids = ALL_IKs[batch: batch + BATCH_SIZE]
            smiles = [ik_to_smiles[i] for i in ids]
            chemeleons = chemeleon_fingerprint(smiles)
            X.extend(chemeleons)
            IKs.extend(ids)
        except:
            pass

    # Convert to numpy array
    X = np.array(X, dtype="float16")

    # Assert that the number of IDs and MFPs are the same
    assert len(IKs) == X.shape[0], f"ERROR: Number of IKs ({len(IKs)}) does not match number of CheMeleon embeddings ({X.shape[0]})"

    # Save results
    np.savez_compressed(os.path.join(PATH_TO_OUTPUT, pathogen, "X_CheMeleon.npz"), X=X)
    with open(os.path.join(PATH_TO_OUTPUT, pathogen, "IKs_CheMeleon.txt"), "w") as f:
        f.write("\n".join(IKs))