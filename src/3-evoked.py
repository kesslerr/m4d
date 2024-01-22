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

plot_dir = os.path.join(base_dir, "plots", "evoked")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

""" HEADER END """

# DEBUG
#experiment = "N170"
#subject = "sub-001" # TODO: mean across ALL subjects!!

""" SPECIFICATIONS END """


""" Individual evoked plots """

for experiment in experiments:

    forking_path = luck_forking_paths[experiment] 
    contrast = contrasts[experiment]
    channels = channels_of_interest[experiment]
    assert len(contrast) == len(channels), "Number of contrasts and channels must be the same!"
    # contrasts and channels are ordered, so that they can be processed with zip


    group_evoked = {}
    group_evoked_diff = {}
    for condition in contrast[next(iter(contrast))]:
        group_evoked[condition] = []
        #print(condition)
    for this_contrast in contrast.keys():
        group_evoked_diff[this_contrast] = []
        #print(this_contrast)

    for subject in subjects:
        plot_dir_exp_sub = os.path.join(plot_dir, experiment, subject)
        if not os.path.exists(plot_dir_exp_sub):
            os.makedirs(plot_dir_exp_sub)

        file = os.path.join(alt_dir, "data", "processed", experiment, subject, f"{forking_path}-epo.fif")
        
        # read epochs
        epochs = mne.read_epochs(file, preload=True, verbose=None)

        # make evoked per condition
        evokeds = {}
        for condition in epochs.event_id.keys():
            evokeds[condition] = epochs[condition].average()
            group_evoked[condition].append(evokeds[condition].copy())
            
        # make difference waves
        #if experiment == 'N170':
        #    evokeds_diff = mne.combine_evoked([evokeds['faces'], evokeds['cars']], weights=[1, -1])
        
        evokeds_diff = {}
        for this_contrast in contrast.keys():
            evokeds_diff[this_contrast] = mne.combine_evoked([evokeds[contrast[this_contrast][0]], evokeds[contrast[this_contrast][1]]], weights=[1, -1])
            group_evoked_diff[this_contrast].append(evokeds_diff[this_contrast].copy())
            #evokeds_diff.comment = this_contrast # might be the default comment

        for channel, this_contrast in zip(channels, contrast.keys()):
            # plot evoked
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True, sharey=True)
            
            mne.viz.plot_compare_evokeds(evokeds,
                                        picks=channel,
                                        title=None, #f"{experiment} {subject} {channel} {forking_path}",
                                        axes=ax[0],
                                        show=False)[0]

            # plot difference wave
            #if experiment == 'N170':
            mne.viz.plot_compare_evokeds(evokeds_diff[this_contrast],
                                        picks=channel,
                                        title=None, #f"{experiment} {subject} {channel} {forking_path}",
                                        axes=ax[1],
                                        colors=['k'],
                                        show=False)[0]
            plt.suptitle(f"{experiment}, {subject}, channel: {channel}, forking path: {forking_path}")
            plt.tight_layout()
            
            # save plot
            fig.savefig(os.path.join(plot_dir_exp_sub, f"{channel}_{forking_path}.png"))
            plt.close(fig)

    """ grand average evoked plots """

    grand_average_evoked = {}
    for condition in group_evoked.keys():
        grand_average_evoked[condition] = mne.grand_average(group_evoked[condition])

    grand_average_evoked_diff = {}
    for this_contrast in contrast.keys():
        grand_average_evoked_diff[this_contrast] = mne.grand_average(group_evoked_diff[this_contrast])

    for channel, this_contrast in zip(channels, contrast.keys()):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True, sharey=True)
        mne.viz.plot_compare_evokeds(grand_average_evoked,
                                    picks=channel,
                                    legend='upper left', 
                                    show_sensors='upper right',
                                    title=None, #f"{experiment} {subject} {channel} {forking_path}",
                                    axes=ax[0],
                                    truncate_xaxis=True,
                                    truncate_yaxis=True,
                                    show=False)[0]
        mne.viz.plot_compare_evokeds(grand_average_evoked_diff[this_contrast],
                                    picks=channel,
                                    legend='upper left', 
                                    show_sensors='upper right',
                                    title=None, #f"{experiment} {subject} {channel} {forking_path}",
                                    axes=ax[1],
                                    colors=['k'],
                                    truncate_xaxis=True,
                                    truncate_yaxis=True,
                                    show=False)[0]
        plt.suptitle(f"{experiment}, channel: {channel}, forking path: {forking_path}")
        plt.tight_layout()

        # save plot
        fig.savefig(os.path.join(plot_dir, f"{experiment}_{channel}_{forking_path}.png"))
        #plt.close(fig)


    # TODO: there is huge whitespace above the first subplot
    # TODO: y axis is not showing numbers