import argparse
import glob
import os
import random
from matplotlib.style import available
import shutil

import numpy as np

from utils import get_module_logger
from pathlib import Path
import random


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    train_path = f"{destination}/train"
    test_path = f"{destination}/test"
    val_path = f"{destination}/val"
    # clear the folders if they exist
    shutil.rmtree(train_path, ignore_errors=True)
    shutil.rmtree(test_path, ignore_errors=True)
    shutil.rmtree(val_path, ignore_errors=True)
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(test_path).mkdir(parents=True, exist_ok=True)
    Path(val_path).mkdir(parents=True, exist_ok=True)
    available_files = os.listdir(source)
    random.shuffle(available_files)
    train_samples = int(len(available_files)*0.7)
    val_samples = int(len(available_files)*0.2)
    test_samples = len(available_files) - train_samples - val_samples
    for idx, file in enumerate(available_files):
        if idx < train_samples:
            shutil.copy(f"{source}/{file}", f"{train_path}/{file}")
        elif train_samples <= idx < train_samples+val_samples:
            shutil.copy(f"{source}/{file}", f"{val_path}/{file}")
        else:
            shutil.copy(f"{source}/{file}", f"{test_path}/{file}")


    # TODO: Implement function


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)