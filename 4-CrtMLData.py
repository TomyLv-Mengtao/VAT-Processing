# Step 4, process data format for ML methods processing
# 2024-02-06

import numpy as np
import pandas as pd
from pandas.io.common import dataclasses
import os
import json

# Function
def process_files(input_folder, output_folder):
    # Get all the file names in the input folder
    filenames = os.listdir(input_folder)

    # Define all possible categories
    categories = ["alt", "att", "spd", "vsp", "otw", "nos", "oth"]

    for filename in filenames:
        # Read the file
        df = pd.read_excel(os.path.join(input_folder, filename))

        # Perform one-hot encoding on the 'dwell' column
        df_encoded = pd.get_dummies(df, columns=['dwell'])

        # Ensure all categories are present
        for category in categories:
            if f'dwell_{category}' not in df_encoded.columns:
                df_encoded[f'dwell_{category}'] = 0

        # Convert boolean values to integers. Otherwise, it will be "TURE" and "FALSE", instead of "1" and "0"
        for category in categories:
            df_encoded[f'dwell_{category}'] = df_encoded[f'dwell_{category}'].astype(int)

        # Reorder the columns to ensure they are always in the same sequence
        df_encoded = df_encoded[['duration'] + [f'dwell_{category}' for category in categories]]

        # Remove the extension from the filename
        filename_without_extension = os.path.splitext(filename)[0]

        # Save the processed DataFrame to the output folder with the same filename
        df_encoded.to_csv(os.path.join(output_folder, filename_without_extension + ".csv"), index=False)

# Process the positive and negative examples
# Pos path
process_files("/workspaces/VAT-Processing/Duration_Slices/30/Pos", "/workspaces/VAT-Processing/ML/Pos/30")
# Neg path
process_files("/workspaces/VAT-Processing/Duration_Slices/30/Neg", "/workspaces/VAT-Processing/ML/Neg/30")

