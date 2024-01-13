
import json
import seaborn as sns
from pipeline.datasets import get_erpcore
import os, sys
os.chdir('./..')

# define experiment and number of participants
experiments = ['ERN', 'LRP', 'MMN', 'N170', 'N2pc', 'N400', 'P3']
n_participants = 40

# download 
file_paths = {}
for experiment in experiments:
    file_paths[experiment] = get_erpcore(experiment, 
                                        participants=n_participants, 
                                        path='data')

# move from data/erpcore to data, and delete the "eeg" subfolder
# TODO: this does not yet move all files, some are not listed in the dict
# for experiment in experiments:
#     for file_type in file_paths[experiment].keys():
#         for file in file_paths[compexperimentonent][file_type]:
#             new_file = file.replace('/erpcore/','/')
#             new_file = new_file.replace('/eeg/','/')
#             os.renames(file,  
#                        new_file)
#             # os.renames creates directory (os.rename (without s) 
#             # would throw error if directory doesn't exist)