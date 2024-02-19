
import json
import seaborn as sns
from pipeline.datasets import get_erpcore
import os, sys
os.chdir('./..')

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


## CICHY

"""
to download:
adults (30 GB): wget -nc -O adults.zip https://files.de-1.osf.io/v1/resources/ruxfg/providers/osfstorage/6361374bb2331213c9431246/?zip=
infants (1.5 GB): wget -nc -O infants.zip https://files.de-1.osf.io/v1/resources/ruxfg/providers/osfstorage/63612a03b233121399431479/?zip=



"""