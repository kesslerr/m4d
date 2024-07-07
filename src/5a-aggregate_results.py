
# aggregate the results of eegnet and sliding 


import os, sys
import pandas as pd
import numpy as np
from natsort import natsorted
# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#ptmp_dir = "/ptmp/kroma/m4d/"
os.chdir(base_dir)
sys.path.append(base_dir)
from src.config import experiments, subjects # ERPCORE 


# Define the directory containing the subfolders
model_folder = "/ptmp/kroma/m4d/models/eegnet"

experiments = ["ERN", "LRP", "MMN", "N170", "N2pc", "N400", "P3"]

subjects = [subdir for subdir in os.listdir(os.path.join(model_folder, experiments[0])) if os.path.isdir(os.path.join(model_folder, experiments[0], subdir))]

new_column_names = ['emc', 'mac', 'lpf', 'hpf', 'ref', 'det', 'base', 'ar']

# Initialize an empty DataFrame to store the concatenated data
concatenated_data = pd.DataFrame()

# DEBUG
#experiment = experiments[0]
#subject = subjects[0]

for experiment in experiments:
    print(f"Processing Experiment {experiment}!")
    for subject in subjects:
        # List all CSV files in the subdirectory
        files = [os.path.join(model_folder, experiment, subject, f) for f in os.listdir(os.path.join(model_folder, experiment, subject)) if f.endswith('.csv')]

        data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
        
        # Split the column at each "_" and name the new columns
        data[new_column_names] = data['forking_path'].str.split('_', expand=True)
        
        # Add subdirectory names as separate columns
        data['experiment'] = experiment
        data['subject'] = subject
        
        # Delete the "forking_path" column
        data.drop(columns=['forking_path'], inplace=True)
        
        concatenated_data = pd.concat([concatenated_data, data], ignore_index=True)

# recode variables
concatenated_data['emc'] = pd.Categorical(concatenated_data['emc'], categories=["None", "ica"], ordered=True)
concatenated_data['mac'] = pd.Categorical(concatenated_data['mac'], categories=["None", "ica"], ordered=True)
concatenated_data['lpf'] = pd.Categorical(concatenated_data['lpf'], categories=["None", "6", "20", "45"], ordered=True)
concatenated_data['hpf'] = pd.Categorical(concatenated_data['hpf'], categories=["None", "0.1", "0.5"], ordered=True)
concatenated_data['ref'] = pd.Categorical(concatenated_data['ref'], categories=["average", "Cz", "P9P10"], ordered=True)
concatenated_data['det'] = pd.Categorical(concatenated_data['det'], categories=["None", "linear"], ordered=True)
concatenated_data['base'] = pd.Categorical(concatenated_data['base'], categories=["None", "200ms", "400ms"], ordered=True)
concatenated_data['ar'] = pd.Categorical(concatenated_data['ar'], categories=["False", "int", "intrej"], ordered=True)
concatenated_data['experiment'] = pd.Categorical(concatenated_data['experiment'])

concatenated_data.to_csv(os.path.join("/u/kroma/m4d/models/eegnet_extended.csv"), index=False)

# Output the combined data
print(concatenated_data)

