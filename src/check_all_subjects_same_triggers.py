import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
import fnmatch
import re
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt

# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.append(base_dir)
alt_dir = "/ptmp/kroma/m4d/"

#from src.utils import CharacteristicsManager, ica_eog_emg, autorej, summarize_artifact_interpolation
from src.config import subjects, experiments, channels_of_interest, luck_forking_paths, contrasts
from src.utils import get_forking_paths

d = {'experiment': [], 'subject': [], 'keys': []}

for experiment in experiments:
    for subject in subjects:
        forking_path = luck_forking_paths[experiment] 
        #forking_path = forking_paths[0] # get the first forking path for simplicity, I assume an error would occur in all of them equally
        
        # load epochs
        file = os.path.join(alt_dir, "data", "processed", experiment, subject, f"{forking_path}-epo.fif")
        epochs = mne.read_epochs(file, preload=True, verbose=None)
        
        d['experiment'].append(experiment)
        d['subject'].append(subject)
        d['keys'].append(list(epochs.event_id.keys()))


df = pd.DataFrame(d)
df.head()