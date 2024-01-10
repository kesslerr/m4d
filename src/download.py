
import json
import seaborn as sns
from pipeline.datasets import get_erpcore
import os, sys
os.chdir('./..')

# define components and number of participants
components = ['ERN', 'LRP', 'MMN', 'N170', 'N2pc', 'N400', 'P3']
n_participants = 5

# download 
file_paths = {}
for component in components:
    file_paths[component] = get_erpcore(component, 
                                        participants=n_participants, 
                                        path='data')

# move from data/erpcore to data, and delete the "eeg" subfolder
# TODO: this does not yet move all files, some are not listed in the dict
# for component in components:
#     for file_type in file_paths[component].keys():
#         for file in file_paths[component][file_type]:
#             new_file = file.replace('/erpcore/','/')
#             new_file = new_file.replace('/eeg/','/')
#             os.renames(file,  
#                        new_file)
#             # os.renames creates directory (os.rename (without s) 
#             # would throw error if directory doesn't exist)