
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


