## Randomly select VATs(JSONs)
## Create Training & Testing JSON files using all-to-all method, or all-to-10% method

import numpy as np
import pandas as pd
from pandas.io.common import dataclasses
import os
import json
import random

# Define the paths
######### Caution the time setting
pos_path = '/workspaces/VAT-Processing/Traces/Pos/80'
neg_path = '/workspaces/VAT-Processing/Traces/Neg/80'
output_path = '/workspaces/VAT-Processing/V3TrnSet/All2All/80'
# output_path = '/workspaces/VAT-Processing/5Sec/All-1'


# # Get all the JSON files in the directories
# pos_files = [os.path.join(pos_path, f) for f in os.listdir(pos_path) if f.endswith('.json')]
# neg_files = [os.path.join(neg_path, f) for f in os.listdir(neg_path) if f.endswith('.json')]

# No name suffix, no check
pos_files = [os.path.join(pos_path, f) for f in os.listdir(pos_path)]
neg_files = [os.path.join(neg_path, f) for f in os.listdir(neg_path)]

# Number of files to select, folder with fewer files, minus 1
Sample_Num = min(len(pos_files), len(neg_files)) - 1
# Select 9/10 data as training, following 10-fold method
trn_Num = int(Sample_Num * 0.9)


# Function to select files and create samples
def create_samples(files, key):
    # Randomly select files
    selected_files = random.sample(files, k = Sample_Num)  # Use choices instead of sample for replacement
    # Split into training and test files


    # All2All data samples
    
    training_files = selected_files
    test_files = selected_files

    # # All2One data samples, Use all to train, select 10% to test
    # training_files = selected_files
    # test_files = selected_files[trn_Num:]


    # Print selected file names
    # print(f'Selected files for {key} training: {training_files}')
    # print(f'Selected files for {key} testing: {test_files}')
    # Create training and test samples
    training_sample = {key: [json.load(open(f))[key] for f in training_files]}
    test_sample = {key: [json.load(open(f))[key] for f in test_files]}
    return training_sample, test_sample


# Define vocab
vocab = ["alt", "att", "spd", "vsp", "otw", "nos", "oth", "a", "b", "c", "d","e"]

# Loop 10 times
for i in range(10):
    print(f'Loop {i+1}')
    # Create positive samples
    pos_training_sample, pos_test_sample = create_samples(pos_files, 'traces_pos')
    # Create negative samples
    neg_training_sample, neg_test_sample = create_samples(neg_files, 'traces_neg')
    # Merge training samples and save as JSON
    training_sample = {'vocab': vocab, **pos_training_sample, **neg_training_sample}
    with open(os.path.join(output_path, f'training_{i+1}.json'), 'w') as f:
        json.dump(training_sample, f)
    # Merge test samples and save as JSON
    test_sample = {'vocab': vocab, **pos_test_sample, **neg_test_sample}
    with open(os.path.join(output_path, f'test_{i+1}.json'), 'w') as f:
        json.dump(test_sample, f)
