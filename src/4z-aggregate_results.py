
# aggregate the results of eegnet and sliding 


import os
import pandas as pd
import numpy as np
from natsort import natsorted

## ERPCORE -- EEGNET

# Define the directory containing the subfolders
model_folder = "/u/kroma/m4d/models/eegnet/"

# Get subdirectory names
experiments = [subdir for subdir in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, subdir))]
# delete MIPDB
experiments = [x for x in experiments if x != "MIPDB"]

subjects = [subdir for subdir in os.listdir(os.path.join(model_folder, experiments[0])) if os.path.isdir(os.path.join(model_folder, experiments[0], subdir))]

new_column_names = ['ref', 'hpf', 'lpf', 'emc', 'mac', 'base', 'det', 'ar']

# Initialize an empty DataFrame to store the concatenated data
concatenated_data = pd.DataFrame()



for experiment in experiments:
    for subject in subjects:
        # List all CSV files in the subdirectory
        files = [os.path.join(model_folder, experiment, subject, f) for f in os.listdir(os.path.join(model_folder, experiment, subject)) if f.endswith('.csv')]
        
        # Read and vertically concatenate the CSV files
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
concatenated_data['hpf'] = pd.Categorical(concatenated_data['hpf'], categories=["None", "0.1", "0.5"], ordered=True)
concatenated_data['lpf'] = pd.Categorical(concatenated_data['lpf'], categories=["None", "6", "20", "45"], ordered=True)
concatenated_data['ref'] = pd.Categorical(concatenated_data['ref'], categories=["average", "Cz", "P9P10"], ordered=True)
concatenated_data['emc'] = pd.Categorical(concatenated_data['emc'], categories=["None", "ica"], ordered=True)
concatenated_data['mac'] = pd.Categorical(concatenated_data['mac'], categories=["None", "ica"], ordered=True)
concatenated_data['base'] = pd.Categorical(concatenated_data['base'], categories=["200ms", "400ms"], ordered=True)
concatenated_data['det'] = pd.Categorical(concatenated_data['det'], categories=["offset", "linear"], ordered=True)
concatenated_data['ar'] = pd.Categorical(concatenated_data['ar'], categories=["False", "True"], ordered=True)
concatenated_data['experiment'] = pd.Categorical(concatenated_data['experiment'])

concatenated_data["dataset"] = "ERPCORE"
concatenated_data['dataset'] = pd.Categorical(concatenated_data['dataset'])


# Output the combined data
print(concatenated_data)

#data_folder = "/u/kroma/m4d/model"

concatenated_data_1 = concatenated_data.copy()
#concatenated_data.to_csv(os.path.join(model_folder, "ERPCORE_eegnet.csv"), index=False)




## MIPDB -- EEGNET

model_folder = "/u/kroma/m4d/models/eegnet/MIPDB/"

subjects = [subdir for subdir in os.listdir(os.path.join(model_folder)) if os.path.isdir(os.path.join(model_folder, subdir))]


# during aggregation, define the age_group by the age of the subject, and call it LRP_8-10, LRP_10-12, LRP_12-14, LRP_14-16, LRP_16-18
dfage = pd.read_csv("/u/kroma/m4d/data/mipdb/participants.csv")

bins = [6, 9, 11, 13, 17, float('inf')]  # The last bin is for 18+
labels = ['6-9', '10-11', '12-13', '14-17', '18+']
# Bin the 'Age' column
dfage['Age_Group'] = pd.cut(dfage['Age'], bins=bins, labels=labels)

age_group_counts = dfage['Age_Group'].value_counts()
print(age_group_counts)

concatenated_data = pd.DataFrame()
for subject in subjects:
    
    try:
        # List all CSV files in the subdirectory
        files = [os.path.join(model_folder, subject, f) for f in os.listdir(os.path.join(model_folder, subject)) if f.endswith('.csv')]
        
        # Read and vertically concatenate the CSV files
        data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
        
        # Split the column at each "_" and name the new columns
        data[new_column_names] = data['forking_path'].str.split('_', expand=True)
        
        # Add subdirectory names as separate columns
        #data['experiment'] = experiment
        data['subject'] = subject
        
        # Delete the "forking_path" column
        data.drop(columns=['forking_path'], inplace=True)
        
        concatenated_data = pd.concat([concatenated_data, data], ignore_index=True)

    except:
        print(f"Error with {subject}")

# merge the age on the experiment variable

concatenated_data["age_group"] = concatenated_data["subject"].map(dfage.set_index("ID")["Age_Group"])
concatenated_data["experiment"] = "LRP_" + concatenated_data["age_group"].astype(str)
concatenated_data.drop(columns=["age_group"], inplace=True)

# merge the dataset on the dataset variable
concatenated_data["dataset"] = "MIPDB"

# recode variables
concatenated_data['hpf'] = pd.Categorical(concatenated_data['hpf'], categories=["None", "0.1", "0.5"], ordered=True)
concatenated_data['lpf'] = pd.Categorical(concatenated_data['lpf'], categories=["None", "6", "20", "45"], ordered=True)
concatenated_data['ref'] = pd.Categorical(concatenated_data['ref'], categories=["average", "Cz", "P9P10"], ordered=True)
concatenated_data['emc'] = pd.Categorical(concatenated_data['emc'], categories=["None", "ica"], ordered=True)
concatenated_data['mac'] = pd.Categorical(concatenated_data['mac'], categories=["None", "ica"], ordered=True)
concatenated_data['base'] = pd.Categorical(concatenated_data['base'], categories=["200ms", "400ms"], ordered=True)
concatenated_data['det'] = pd.Categorical(concatenated_data['det'], categories=["offset", "linear"], ordered=True)
concatenated_data['ar'] = pd.Categorical(concatenated_data['ar'], categories=["False", "True"], ordered=True)

age_group_sorted = natsorted(np.unique(concatenated_data['experiment']))
concatenated_data['experiment'] = pd.Categorical(concatenated_data['experiment'], categories=age_group_sorted)
concatenated_data['dataset'] = pd.Categorical(concatenated_data['dataset'])


# concatenate and save the data
concatenated_data_2 = concatenated_data.copy()

concatenated_data_all = pd.concat([concatenated_data_1, concatenated_data_2], ignore_index=True)
concatenated_data_all.to_csv(os.path.join("/u/kroma/m4d/models/", "eegnet.csv"), index=False)
