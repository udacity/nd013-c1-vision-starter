import argparse
import glob
import os
import random
from shutil import copy

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    with open('dark_files.txt') as file:
        list_of_dark_files = file.readlines()
        list_of_dark_files = [line.rstrip() for line in list_of_dark_files]
    
    with open('rainy_files.txt') as file:
        list_of_rainy_files = file.readlines()
        list_of_rainy_files = [line.rstrip() for line in list_of_rainy_files]
    
    combined_list_of_special_files = list_of_dark_files + list_of_rainy_files

    list_of_all_files = [os.path.basename(x) for x in glob.glob(source + '/*.tfrecord')]
    filtered_files = [file for file in list_of_all_files if file not in combined_list_of_special_files]

    training_files = []
    validation_files = []
    testing_files = []

    # 75% for training, 15% for validation, 10% testing
    
    # filtered files
    idx_of_75_perct = int(len(filtered_files) * 0.75)
    idx_of_90_perct = int(len(filtered_files) * 0.9)
    training_files += filtered_files[0:idx_of_75_perct]
    validation_files += filtered_files[idx_of_75_perct:idx_of_90_perct]
    testing_files += filtered_files[idx_of_90_perct:]

    # rainy files
    idx_of_75_perct = int(len(list_of_rainy_files) * 0.75)
    idx_of_90_perct = int(len(list_of_rainy_files) * 0.9)
    training_files += list_of_rainy_files[0:idx_of_75_perct]
    validation_files += list_of_rainy_files[idx_of_75_perct:idx_of_90_perct]
    testing_files += list_of_rainy_files[idx_of_90_perct:]

    # dark files
    idx_of_75_perct = int(len(list_of_dark_files) * 0.75)
    idx_of_90_perct = int(len(list_of_dark_files) * 0.9)
    training_files += list_of_dark_files[0:idx_of_75_perct]
    validation_files += list_of_dark_files[idx_of_75_perct:idx_of_90_perct]
    testing_files += list_of_dark_files[idx_of_90_perct:]

    for file in training_files:
        copy(source + '/' + file, destination + '/train/')

    for file in validation_files:
        copy(source + '/' + file, destination + '/val/')

    for file in testing_files:
        copy(source + '/' + file, destination + '/test/')


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