import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt

# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.append(base_dir)

#from src.utils import CharacteristicsManager, ica_eog_emg, autorej, summarize_artifact_interpolation
from src.config import subjects, experiments, channels_of_interest, luck_forking_paths
from src.utils import get_forking_paths

plot_dir = os.path.join(base_dir, "plots", "evoked")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

""" HEADER END """

experiment = "N170"
subject = "sub-001" # TODO: mean across ALL subjects!!



""" SPECIFICATIONS END """

#forking_paths, files, _ = get_forking_paths(base_dir=base_dir, experiment=experiment, subject=subject, sample=5)


#for forking_path, file in zip(forking_paths, files):
forking_path = luck_forking_paths[experiment] 
file = glob(os.path.join(base_dir, "data", "processed", experiment, subject, f"{forking_path}-epo.fif"))[0]    

# read epochs
epochs = mne.read_epochs(file, preload=True, verbose=None)

# make evoked per condition
evokeds = {}
for condition in epochs.event_id.keys():
    evokeds[condition] = epochs[condition].average()
    
# make difference waves
evokeds_diff = {}
if experiment == 'N170':
    evokeds_diff['faces-cars'] = evokeds['faces'].copy()
    evokeds_diff['faces-cars'].data = evokeds['faces'].data - evokeds['cars'].data

evokeds_diff = {}


channels = channels_of_interest[experiment]

for channel in channels:
    # plot evoked
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True, sharey=True)
    
    mne.viz.plot_compare_evokeds(evokeds,
                                picks=channel,
                                title=None, #f"{experiment} {subject} {channel} {forking_path}",
                                axes=ax[0],
                                show=False)[0]

    # plot difference wave
    #if experiment == 'N170':
    mne.viz.plot_compare_evokeds(evokeds_diff,
                                picks=channel,
                                title=None, #f"{experiment} {subject} {channel} {forking_path}",
                                axes=ax[1],
                                colors=['k'],
                                show=False)[0]
    plt.suptitle(f"{experiment} {subject} {channel} {forking_path}")
    plt.tight_layout()
    
    # save plot
    fig.savefig(os.path.join(plot_dir, f"{experiment}_{subject}_{channel}_{forking_path}.png"))
