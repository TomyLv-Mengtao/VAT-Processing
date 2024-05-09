# This is the 2nd step: to create VATs (JSON files) based on the slices
# The dwell times are binned in this step

import numpy as np
import pandas as pd
from pandas.io.common import dataclasses
import os
import json
import random


# Step 1: Load the files
# folder_path = "/workspaces/VAT-Processing/Original Sample (40)"

############## Attetnion!
# notice the length of the dwell slices
############## Attetnion!

# # 15 Sec
# pos_path = "/workspaces/VAT-Processing/Duration_Slices/15/Pos"
# neg_path = "/workspaces/VAT-Processing/Duration_Slices/15/Neg"

# pos_save_path = "/workspaces/VAT-Processing/Traces/Pos/15"
# neg_save_path = "/workspaces/VAT-Processing/Traces/Neg/15"

# 5 Sec
pos_path = "/workspaces/VAT-Processing/Duration_Slices/35/Pos"
neg_path = "/workspaces/VAT-Processing/Duration_Slices/35/Neg"

pos_save_path = "/workspaces/VAT-Processing/Ln/Pos/35"
neg_save_path = "/workspaces/VAT-Processing/Ln/Neg/35"


# Get a list of all files in the directory
pos_files = os.listdir(pos_path)
neg_files = os.listdir(neg_path)

# Define a function to process each file
def process_file(file, path, save_path, key):
    # Print the name of the current file
    print(f'Processing file: {file}')

    # Load the file into a pandas DataFrame
    df = pd.read_excel(os.path.join(path, file))

    # Initialize an empty list to store the results
    results = []

    # Iterate over each row in the DataFrame
    # Bin the dwell times
    
    ## By 25iles: 100; 241; 491; 1052
    # # 25iles
    # for _, row in df.iterrows():
    #     # Determine the second value based on the duration
    #     if row['duration'] < 100:
    #         second_value = 'a'
    #     elif row['duration'] < 241:
    #         second_value = 'b'
    #     elif row['duration'] < 491:
    #         second_value = 'c'
    #     elif row['duration'] < 1052:
    #         second_value = 'd'
    #     else:
    #         second_value = 'e'
    
    # By Ln: Ln4.6 = 99.5; Ln 5.6 = 270.4; Ln 6.6 = 735.1; Ln7.6 = 1998.2
    # Ln
    for _, row in df.iterrows():
        # Determine the second value based on the duration
        if row['duration'] < 100:
            second_value = 'a'
        elif row['duration'] < 270.4:
            second_value = 'b'
        elif row['duration'] < 735.1:
            second_value = 'c'
        elif row['duration'] < 1998.2:
            second_value = 'd'
        else:
            second_value = 'e'
    
    # # By GMM (Gaussian mixture model): 100, 525.6; 1728.5; 5832.5   
    # # GMM
    # for _, row in df.iterrows():
    #     # Determine the second value based on the duration
    #     if row['duration'] < 100:
    #         second_value = 'a'
    #     elif row['duration'] < 525.6:
    #         second_value = 'b'
    #     elif row['duration'] < 1728.5:
    #         second_value = 'c'
    #     elif row['duration'] < 5832.5:
    #         second_value = 'd'
    #     else:
    #         second_value = 'e'
            
    # # By K-Means: 100, 1734, 6011, 17531     
    # # K-Means
    # for _, row in df.iterrows():
    #     # Determine the second value based on the duration
    #     if row['duration'] < 100:
    #         second_value = 'a'
    #     elif row['duration'] < 1735:
    #         second_value = 'b'
    #     elif row['duration'] < 6012:
    #         second_value = 'c'
    #     elif row['duration'] < 17532:
    #         second_value = 'd'
    #     else:
    #         second_value = 'e'

        
        # Add the dwell and its second value to the results list
        results.append([row['dwell'], second_value])

    # Save the results as a JSON file
    with open(os.path.join(save_path, f'{os.path.splitext(file)[0]}.json'), 'w') as f:
        json.dump({key: results}, f)

# Process each file in the directory
for file in pos_files:
    process_file(file, pos_path, pos_save_path, "traces_pos")

for file in neg_files:
    process_file(file, neg_path, neg_save_path, "traces_neg")
