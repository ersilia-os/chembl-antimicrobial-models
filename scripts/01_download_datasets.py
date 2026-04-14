"""
Step 01 — Download representative datasets.

Copies the selected binary datasets produced by the chembl-antimicrobial-tasks
pipeline into data/raw/<pathogen>/. The primary source is
<tasks_output_dir>/<pathogen>/19_final_datasets.zip, which contains one CSV per
assay with two columns: smiles and bin (binary activity label).

Usage:
    python scripts/01_download_datasets.py --pathogen ecoli --source ../chembl-antimicrobial-tasks/output
    python scripts/01_download_datasets.py --all --source ../chembl-antimicrobial-tasks/output
"""

import argparse

# TODO: import shutil, zipfile, pathlib as needed
# TODO: define PATHOGENS list (the 15 supported codes)
# TODO: implement download/copy logic per pathogen:
#       - locate <source>/<pathogen>/19_final_datasets.zip
#       - extract CSVs into data/raw/<pathogen>/
#       - optionally also copy 19_final_datasets_metadata.csv


def download_pathogen(pathogen: str, source: str) -> None:
    # TODO: implement
    print(f"[TODO] Downloading datasets for {pathogen} from {source}")


def main(args: argparse.Namespace) -> None:
    # TODO: implement
    print("Not yet implemented.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download representative binary datasets from chembl-antimicrobial-tasks outputs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pathogen",
        type=str,
        help="Pathogen code to download (e.g. ecoli, mtuberculosis).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download datasets for all supported pathogens.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the chembl-antimicrobial-tasks output directory.",
    )
    args = parser.parse_args()
    main(args)
