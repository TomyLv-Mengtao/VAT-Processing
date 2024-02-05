# This is the 1st step: to split the origincal sample files (pre-processed and label) into slices
# Slices have different length/ durations
# The dutation of the slices have not been binned yet

import numpy as np
import pandas as pd
from pandas.io.common import dataclasses
import os
import json
import random


# Step 1: Load the files
folder_path = "/workspaces/VAT-Processing/Original Sample (40)"

## Attetnion!
# notice the length of the dwell slices
## Attetnion!
# 20 Sec
output_path = "/workspaces/VAT-Processing/Duration_Slices/20"
LENGTH = 20

# 15 Sec
# output_path = "/workspaces/VAT-Processing/Duration_Slices/15"
# LENGTH = 15

# 10 Sec
# output_path = "/workspaces/VAT-Processing/Duration_Slices/10"
# LENGTH = 10

# 5 Sec
# output_path = "/workspaces/VAT-Processing/Duration_Slices/5"
# LENGTH = 5

# Slice duartion calculation
DUR = LENGTH * 1000 # MSec
SUB_DUR =  DUR * 0.6 # Sub duration for calculating the shortest 


files = os.listdir(folder_path)

for file in files:
    df = pd.read_excel(os.path.join(folder_path, file))

    # Remove the extension from the filename
    file_name, _ = os.path.splitext(file)
    print(file_name)

    # Step 2: Check the "Mode" column
    if df['Mode'][0] == 'SAP':
        r_mid = df[df['Event'] == 'Manual>AP 1'].index[0] + 1
        parts = [df.iloc[:r_mid], df.iloc[r_mid:]]
    elif df['Mode'][0] == 'AP':
        parts = [df]
    else:
        print(f'Error in file {file}: Mode is not SAP or AP')
        continue

    for k, part in enumerate(parts):
        # Step 3: Delete rows where "Eye movement type" is "Saccade"
        part = part[part['Eye movement type'] != 'Saccade']

        # Step 4: Define dwells
        dwell_dict = {
            'AOI hit [Screenshot 2 - ALT]': 'alt',
            'AOI hit [Screenshot 2 - ATT]': 'att',
            'AOI hit [Screenshot 2 - NOSE]': 'nos',
            'AOI hit [Screenshot 2 - OTW]': 'otw',
            'AOI hit [Screenshot 2 - SPD]': 'spd',
            'AOI hit [Screenshot 2 - VSPD]': 'vsp'
        }

        dwell_list = []
        dwell_start_time_list = []
        dwell_end_time_list = []
        for _, row in part.iterrows():
            dwell_type = None

            for col, dwell in dwell_dict.items():
                if row[col] == 1:
                    dwell_type = dwell
                    break
            if dwell_type is None:
                dwell_type = 'oth'

            if len(dwell_list) > 0 and dwell_list[-1] == dwell_type:
                dwell_end_time_list[-1] = row['Recording timestamp [ms]']
                # print("dwell type", dwell_type)
                # print("end_time",row['Recording timestamp [ms]'])
                continue

            dwell_list.append(dwell_type)
            dwell_start_time_list.append(row['Recording timestamp [ms]'])
            dwell_end_time_list.append(row['Recording timestamp [ms]'])

        # Calculate durations of dwells
        dwell_durations = np.array(dwell_end_time_list) - np.array(dwell_start_time_list)
        # print("dwell_list",dwell_list)
        # print("dwell_start_time_list",dwell_start_time_list)
        # print("dwell_end_time_list",dwell_end_time_list)
        # print("dwell duration", dwell_durations)

        # Step 5: Separate dwells into DUR slices
        slices = []
        slice_start_time = dwell_start_time_list[0]
        slice_dwell_list = []
        slice_durations_list = []
        for i in range(len(dwell_start_time_list)):
            if (slice_start_time + DUR) <= dwell_end_time_list[i]:
                if (slice_start_time + DUR) - dwell_start_time_list[i] >= dwell_end_time_list[i] - (slice_start_time + DUR):
                    slice_dwell_list.append(dwell_list[i])
                    slice_durations_list.append(dwell_end_time_list[i] - dwell_start_time_list[i])
                    slices.append((slice_dwell_list, slice_durations_list))
                    slice_dwell_list = []
                    slice_durations_list = []
                    if i < len(dwell_start_time_list) - 1:
                        slice_start_time = dwell_start_time_list[i+1]
                else:
                    slices.append((slice_dwell_list, slice_durations_list))
                    slice_dwell_list = [dwell_list[i]]
                    slice_durations_list = [dwell_end_time_list[i] - dwell_start_time_list[i]]
                    if i < len(dwell_start_time_list) - 1:
                        slice_start_time = dwell_start_time_list[i]
            else:
                slice_dwell_list.append(dwell_list[i])
                slice_durations_list.append(dwell_durations[i])



        # Check the last slice duration
        if dwell_end_time_list[i] - slice_start_time < SUB_DUR:
            slices.pop()

        # Step 6: Save each slice as a file
        for j, (slice_dwells, slice_durations) in enumerate(slices):
            df_slice = pd.DataFrame({'dwell': slice_dwells, 'duration': slice_durations})

            if df['Mode'][0] == 'SAP':
                filename = f'pos_{file_name}_{k}_{j}.xlsx'
            else:
                filename = f'neg_{file_name}_{k}_{j}.xlsx'

            df_slice.to_excel(os.path.join(output_path, filename), index=False)