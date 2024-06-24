import mne
from scipy.signal import detrend
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import copy # deep copy of dicts
import pickle

# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.append(base_dir)
alt_dir = "/ptmp/kroma/m4d/"

#from src.utils import CharacteristicsManager, ica_eog_emg, autorej, summarize_artifact_interpolation
from src.config import subjects, experiments, channels_of_interest, luck_forking_paths, contrasts, contrasts_combined

plot_dir = os.path.join(base_dir, "plots")
model_dir = os.path.join(base_dir, "models")

""" HEADER END """

# DEBUG
#experiment = "N170"
#subject = "sub-001" # TODO: mean across ALL subjects!!

""" SPECIFICATIONS END """


""" Individual evoked plots """
group_results = {}

for experiment in experiments:
    group_results[experiment] = {}

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


    """ grand average evoked plots """

    grand_average_evoked = {}
    for condition in group_evoked.keys():
        grand_average_evoked[condition] = mne.grand_average(group_evoked[condition])

    grand_average_evoked_diff = {}
    for this_contrast in contrast.keys():
        grand_average_evoked_diff[this_contrast] = mne.grand_average(group_evoked_diff[this_contrast])
        grand_average_evoked_diff[this_contrast].comment = this_contrast # TODO: test if this is right

    for channel, this_contrast in zip(channels, contrast.keys()):        
        # merge grand_average_evoked and grand_average_evoked_diff
        grand_average_both = grand_average_evoked.copy()
        grand_average_both[this_contrast] = grand_average_evoked_diff[this_contrast]
        
        group_results[experiment][channel] = grand_average_both





""" combine left and right (N2pc and LRP) """

#group_results_combined = group_results.copy() # copy it, and change channel values manually
group_results_combined = copy.deepcopy(group_results)

# LRP

C3_right = group_results['LRP']["C3"]["response_right"].copy().pick("C3").get_data() # contralateral
C3_left  = group_results['LRP']["C3"]["response_left"].copy().pick("C3").get_data() # ipsilateral
C4_right = group_results['LRP']["C4"]["response_right"].copy().pick("C4").get_data() # ipsilateral
C4_left  = group_results['LRP']["C4"]["response_left"].copy().pick("C4").get_data() # contralateral
contralateral = np.mean([C3_right, C4_left], axis=0)
ipsilateral = np.mean([C3_left, C4_right], axis=0)
contra_minus_ipsi = contralateral - ipsilateral

contralateral = np.vstack((contralateral, contralateral))
ipsilateral = np.vstack((ipsilateral, ipsilateral))
contra_minus_ipsi = np.vstack((contra_minus_ipsi, contra_minus_ipsi))

del group_results_combined['LRP']["C3"]
del group_results_combined['LRP']["C4"]

group_results_combined['LRP']["C3/C4"] = {}
group_results_combined['LRP']["C3/C4"]["contralateral"] = mne.EvokedArray(contralateral,
                                                                        group_results['LRP']["C3"]["response_right"].copy().pick(["C3", "C4"]).info, 
                                                                        tmin=group_results['LRP']["C3"]["response_right"].tmin)
group_results_combined['LRP']["C3/C4"]["ipsilateral"] = mne.EvokedArray(ipsilateral,
                                                                        group_results['LRP']["C3"]["response_right"].copy().pick(["C3", "C4"]).info, 
                                                                        tmin=group_results['LRP']["C3"]["response_right"].tmin)
group_results_combined['LRP']["C3/C4"]["contralateral - ipsilateral"] = mne.EvokedArray(contra_minus_ipsi,
                                                                        group_results['LRP']["C3"]["response_right"].copy().pick(["C3", "C4"]).info, 
                                                                        tmin=group_results['LRP']["C3"]["response_right"].tmin)

# experiment = "LRP"
# mne.viz.plot_compare_evokeds(group_results_combined[experiment]["C3/C4"],
#                             picks=["C3", "C4"],
#                             combine='mean',
#                             legend='upper left', 
#                             show_sensors='upper right',
#                             title=experiment, #f"{experiment} {subject} {channel} {forking_path}",
#                             #axes=ax[ax_counter],
#                             colors=colors,
#                             linestyles=linestyles,
#                             #ci=0.95, # this doesnt work with dashed?
#                             truncate_xaxis=False,
#                             truncate_yaxis=False,
#                             show=False)[0]

# N2pc

PO7_right = group_results['N2pc']["PO7"]["target_right"].copy().pick("PO7").get_data() # contralateral
PO7_left  = group_results['N2pc']["PO7"]["target_left"].copy().pick("PO7").get_data() # ipsilateral
PO8_right = group_results['N2pc']["PO8"]["target_right"].copy().pick("PO8").get_data() # ipsilateral
PO8_left  = group_results['N2pc']["PO8"]["target_left"].copy().pick("PO8").get_data() # contralateral
contralateral = np.mean([PO7_right, PO8_left], axis=0)
ipsilateral = np.mean([PO7_left, PO8_right], axis=0)
contra_minus_ipsi = contralateral - ipsilateral
# TODO: double check that no side effects with experiment LRP

contralateral = np.vstack((contralateral, contralateral))
ipsilateral = np.vstack((ipsilateral, ipsilateral))
contra_minus_ipsi = np.vstack((contra_minus_ipsi, contra_minus_ipsi))

del group_results_combined['N2pc']["PO7"]
del group_results_combined['N2pc']["PO8"]

group_results_combined['N2pc']["PO7/PO8"] = {}
group_results_combined['N2pc']["PO7/PO8"]["contralateral"] = mne.EvokedArray(contralateral,
                                                                        group_results['N2pc']["PO7"]["target_right"].copy().pick(["PO7", "PO8"]).info, 
                                                                        tmin=group_results['N2pc']["PO7"]["target_right"].tmin)
group_results_combined['N2pc']["PO7/PO8"]["ipsilateral"] = mne.EvokedArray(ipsilateral,
                                                                        group_results['N2pc']["PO7"]["target_right"].copy().pick(["PO7", "PO8"]).info, 
                                                                        tmin=group_results['N2pc']["PO7"]["target_right"].tmin)
group_results_combined['N2pc']["PO7/PO8"]["contralateral - ipsilateral"] = mne.EvokedArray(contra_minus_ipsi,
                                                                        group_results['N2pc']["PO7"]["target_right"].copy().pick(["PO7", "PO8"]).info, 
                                                                        tmin=group_results['N2pc']["PO7"]["target_right"].tmin)


# experiment = "N2pc"
# mne.viz.plot_compare_evokeds(group_results_combined[experiment]["PO7/PO8"],
#                             picks=["PO7", "PO8"],
#                             combine='mean',
#                             legend='upper left', 
#                             show_sensors='upper right',
#                             title=experiment, #f"{experiment} {subject} {channel} {forking_path}",
#                             #axes=ax[ax_counter],
#                             colors=colors,
#                             linestyles=linestyles,
#                             #ci=0.95, # this doesnt work with dashed?
#                             truncate_xaxis=False,
#                             truncate_yaxis=False,
#                             show=False)[0]

""" save results """
with open(f'{model_dir}/evoked_combined.pck', 'wb') as handle:
    pickle.dump(group_results_combined, handle)
with open(f'{model_dir}/evoked.pck', 'wb') as handle:
    pickle.dump(group_results, handle)

""" load results """
with open(f'{model_dir}/evoked_combined.pck', 'rb') as handle:
    group_results_combined = pickle.load(handle)
with open(f'{model_dir}/evoked.pck', 'rb') as handle:
    group_results = pickle.load(handle)


""" plotting """

colors = ['#878787', '#878787', 'k'] # ['#a6cee3', '#1f78b4', '#b2df8a']
linestyles = ['dashed', 'dotted', 'solid']
n_subfigures = len(experiments)

fig, ax = plt.subplots(nrows=n_subfigures, ncols=1, figsize=(8, 10), sharex=False, sharey=False)

for ax_counter, experiment in enumerate(experiments):

    contrast = contrasts_combined[experiment]
    channels = channels_of_interest[experiment]
    if experiment not in ['N2pc', 'LRP']:
        channels = channels[0] # only one channel for all other experiments
        channels_key = channels
    else:
        channels_key = f"{channels[0]}/{channels[1]}"
    
    mne.viz.plot_compare_evokeds(group_results_combined[experiment][channels_key],
                                picks=channels,
                                combine='mean',
                                legend='upper left', 
                                show_sensors='upper right',
                                title=experiment, #f"{experiment} {subject} {channel} {forking_path}",
                                axes=ax[ax_counter],
                                colors=colors,
                                linestyles=linestyles,
                                #ci=0.95, # this doesnt work with dashed?
                                truncate_xaxis=False,
                                truncate_yaxis=False,
                                show=False)[0]
    # turn x axis label description off in all but last axis
    if ax_counter < (n_subfigures - 1):
        ax[ax_counter].set_xlabel("")
    else:
        ax[ax_counter].set_xlabel("Time [s]")
    # set title to experiment name
    ax[ax_counter].set_title(experiment)
    # make individual titles larger
    ax[ax_counter].title.set_fontsize(14)
    # give titles grey background
    ax[ax_counter].title.set_backgroundcolor('lightgrey')
    # make legend white background
    #leg = ax[ax_counter].legend()
    #frame = leg.get_frame()
    #frame.set_facecolor('lightgrey')
    
    #ax[ax_counter].legend(facecolor='white', framealpha=1, loc='upper left') # BUG framealpha does never work
    ax[ax_counter].legend(facecolor='white', framealpha=1, loc='center left', bbox_to_anchor=(1, 0.5)) # BUG framealpha does never work
    
    
    
    #ax[ax_counter].legend().get_frame().set_alpha(1.0)
    # put legend on upper left
    #ax[ax_counter].legend(loc='upper left')

plt.tight_layout()

# save plot
fig.savefig(os.path.join(plot_dir, f"evokeds.png"), dpi=300)
plt.show()
#plt.close(fig)

