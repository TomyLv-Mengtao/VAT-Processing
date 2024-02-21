## Randomly select VATs(JSONs)
## Create Training & Testing JSON files using 10-fold method

import numpy as np
import pandas as pd
from pandas.io.common import dataclasses
import os
import json
import random

# Define the paths
# # 20 Sec
# pos_path = '/workspaces/VAT-Processing/Traces/Pos/20'
# neg_path = '/workspaces/VAT-Processing/Traces/Neg/20'
# output_path = '/workspaces/VAT-Processing/Training_Set/20'

# # 15 Sec
# pos_path = '/workspaces/VAT-Processing/Traces/Pos/15'
# neg_path = '/workspaces/VAT-Processing/Traces/Neg/15'
# output_path = '/workspaces/VAT-Processing/Training_Set/15'

# # 10 Sec
# pos_path = '/workspaces/VAT-Processing/Traces/Pos/10'
# neg_path = '/workspaces/VAT-Processing/Traces/Neg/10'
# output_path = '/workspaces/VAT-Processing/Training_Set/10'

# # 5 Sec
# pos_path = '/workspaces/VAT-Processing/Traces/Pos/5'
# neg_path = '/workspaces/VAT-Processing/Traces/Neg/5'
# output_path = '/workspaces/VAT-Processing/5Sec/9-1'

# 35 Sec
pos_path = '/workspaces/VAT-Processing/Traces/Pos/35'
neg_path = '/workspaces/VAT-Processing/Traces/Neg/35'
output_path = '/workspaces/VAT-Processing/Training_Set/35'

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
    training_files = selected_files[:trn_Num]
    test_files = selected_files[trn_Num:]
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
