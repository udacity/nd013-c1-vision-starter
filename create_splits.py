import argparse
import glob
import os
import random

import numpy as np

import shutil

from utils import get_module_logger


"""
NOTE: as described in the project README:
The training_and_validation folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The testing folder contains frames from the 10 fps video without downsampling.
You will split this training_and_validation data into train, and val sets by completing and executing the create_splits.py file.
So we will not change the "test" folder in the code
"""

def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.
    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    
    # TODO: Implement function
    files = np.array(glob.glob(data_dir + '/training_and_validation/*.tfrecord'))
    
    np.random.shuffle(files)
    # 80-90% is best for training
    train_ratio = 0.85
    idx = int(train_ratio * files.shape[0])

    train_files = files[0: idx]
    val_files = files[idx:]
    
    for file in train_files:
#         print(file)
        shutil.move(file, data_dir + '/train/')
    
    for file in val_files:
#         print(file)
        shutil.move(file, data_dir + '/val/')
    

    
# python create_splits.py --data_dir ./data/waymo
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir) 
