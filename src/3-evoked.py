import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
import numpy as np
import itertools
import pandas as pd


# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.append(base_dir)

#from src.utils import CharacteristicsManager, ica_eog_emg, autorej, summarize_artifact_interpolation
from src.config import subjects, experiments, channels_of_interest
from src.utils import get_forking_paths

""" HEADER END """

experiment = "N170"
subject = "sub-001"



""" SPECIFICATIONS END """

forking_paths, files, _ = get_forking_paths(base_dir=base_dir, experiment=experiment, subject=subject, sample=5)


for forking_path, file in zip(forking_paths, files):
    
    # read epochs
    epochs = mne.read_epochs(file, preload=True, verbose=None)
    
    # make evoked per condition
    evokeds = {}
    for condition in epochs.event_id.keys():
        evokeds[condition] = epochs[condition].average()
    
    channels = channels_of_interest[experiment]
    for channel in channels:
        # plot evoked
        fig = mne.viz.plot_compare_evokeds(evokeds, picks=channel, show=False)
        
    