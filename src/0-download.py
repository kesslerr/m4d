
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pipeline.datasets import get_erpcore
import os, sys
import tarfile
from glob import glob
import shutil
from joblib import Parallel, delayed, dump
from tqdm import tqdm 

os.chdir('./..')
os.chdir('/u/kroma/m4d')
sys.path.append('.')
from src.utils import download_mipdb

## ERPCORE

# define experiment and number of participants
experiments = ['ERN', 'LRP', 'MMN', 'N170', 'N2pc', 'N400', 'P3']
n_participants = 40

# download 
file_paths = {}
for experiment in experiments:
    file_paths[experiment] = get_erpcore(experiment, 
                                        participants=n_participants, 
                                        path='data')


## CHILD MIND INSTITUTE

# check which participants have EEG available
df = pd.read_csv('data/mipdb/MIPDB_PublicFile.csv')

# only participants with all 3 contrast-change EEG blocks available
df = df[(df.EEG_Contrast_Change_Block1 == 1) & (df.EEG_Contrast_Change_Block2 == 1) & (df.EEG_Contrast_Change_Block3 == 1) ] 

# only participants until 24 years old
df = df[df.Age <= 24]

# recode sex (male=1, female=2)
df["Sex"].replace({1: "male", 2: "female"}, inplace=True)

# keep only necessary info and save
df = df[['ID', 'Age', 'Sex']]
df.to_csv('data/mipdb/participants.csv', index=False)

# histogram of age, colored by sex
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='Age', hue='Sex', palette='Greys')
plt.title('Sociodemographic distribution of participants in the MIPDB dataset')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Sex', loc='upper right')
os.makedirs('plots/mipdb', exist_ok=True)
plt.savefig("plots/mipdb/sociodemographic_distribution.png", dpi=200)
plt.show()


subjects = df['ID'].tolist()

def parallel_download(subject):
    # download sub
    destination = f'data/mipdb/{subject}.tar.gz'
    download_mipdb(subject, destination)
    
Parallel(n_jobs=-1)(delayed(parallel_download)(subject) for subject in subjects[41:]) # TODO increase to all  

#subject = included_ids[0] #'A00053375'
# for subject in subjects[41:]: # TODO increase to all
    
#     # download sub
#     destination = f'data/mipdb/{subject}.tar.gz'
#     download_mipdb(subject, destination)


#for subject in tqdm(subjects[41:]): # TODO increase to all

def parallel_extract(subject):    
    # extract only the csv_task files, to check, which are the contrast-change tasks
    extract_path = f'./data/mipdb/tmp_{subject}'
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    file_path_within_archive = f'{subject}/EEG/raw/csv_format'
    # print current path
    print(os.getcwd())
    # Open the .tar.gz file for reading
    print(os.path.exists(f'./data/mipdb/{subject}.tar.gz'))

    with tarfile.open(f'./data/mipdb/{subject}.tar.gz', 'r:gz') as tar:
        # Iterate over all members (files and folders) in the archive
        for member in tar.getmembers():
            # Check if the member is inside the specified subfolder
            if member.name.startswith(file_path_within_archive):
                # Extract the member to the destination folder
                member.name = os.path.relpath(member.name, file_path_within_archive)  # Strip the subfolder path
                tar.extract(member, extract_path)
                print(f"Extracted {member.name} successfully.")

    # in the csv files, open each _events.csv, and check, if "94  " or "95  " or "96  " is in the file
    task_files = sorted(glob(f'{extract_path}/*_events.csv'))
    contrast_task_endings = []
    length_of_files = []
    for file in task_files:
        with open(file, 'r') as f:
            content = f.read()
            if "94  " in content or "95  " in content or "96  " in content:
                print(file)
                contrast_task_endings.append(file)
                length_of_files.append(len(content))

    # extract the 3 digits before "_events.csv"
    #contrast_task_endings = [file.split('/')[-1].split('_')[0][-3:] for file in contrast_task_endings]
    #assert len(contrast_task_endings) >= 3, "Not at least 3 contrast-change tasks found!"
    # because of name variability: do it by using everything between subject and _events.csv
    contrast_task_endings = [file.split('/')[-1].replace("_events.csv", ".raw") for file in contrast_task_endings]
    assert len(contrast_task_endings) >= 3, "Not at least 3 contrast-change tasks found!"
    
    # delete the content of './data/mipdb/tmp' folder
    shutil.rmtree(f'./data/mipdb/tmp_{subject}')

    # Specify the directory where you want to extract the contents
    extract_path = './data/mipdb'

    for contrast_task_ending in contrast_task_endings:

        # Specify the path of the file you want to extract within the archive
        file_path_within_archive = f'{subject}/EEG/raw/raw_format/{contrast_task_ending}' # new because of fails
        #file_path_within_archive = f'{subject}/EEG/raw/raw_format/{subject}{contrast_task_ending}.raw'

        # Open the .tar.gz file for reading
        with tarfile.open(f'data/mipdb/{subject}.tar.gz', 'r:gz') as tar:
            # Check if the file exists in the archive
            if file_path_within_archive in tar.getnames():
                # Extract the file to the specified directory
                member = tar.getmember(file_path_within_archive)
                member.name = os.path.basename(member.name)  # Extract only the filename without the path
                tar.extract(member, path=extract_path)
                print(f"File '{os.path.basename(file_path_within_archive)}' extracted successfully.")
            else:
                print(f"File '{file_path_within_archive}' not found in the archive.")


Parallel(n_jobs=-1)(delayed(parallel_extract)(subject) for subject in subjects[41:]) # TODO increase to all  


#error_subjects = ['A00053398', 'A00053440', 'A00054866', 'A00062919', 'A00063051', 'A00063117']
#Parallel(n_jobs=-1)(delayed(parallel_extract)(subject) for subject in error_subjects) # TODO increase to all  


# remove the downloaded tar.gz file

#wget -P /path/to/directory http://example.com/file.txt


# https://fcp-indi.s3.amazonaws.com/data/Projects/EEG_Eyetracking_CMI_data/compressed/A00053375.tar.gz

## CICHY

"""
to download:
adults (30 GB): wget -nc -O adults.zip https://files.de-1.osf.io/v1/resources/ruxfg/providers/osfstorage/6361374bb2331213c9431246/?zip=
infants (1.5 GB): wget -nc -O infants.zip https://files.de-1.osf.io/v1/resources/ruxfg/providers/osfstorage/63612a03b233121399431479/?zip=
"""
