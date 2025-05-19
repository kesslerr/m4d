import os, sys
import numpy as np
import pandas as pd
import json
import mne
from glob import glob
import re
from mne.preprocessing import ICA
from autoreject import AutoReject
import requests

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.append(base_dir)

from src.config import epoch_windows

""" ------------------- pre-multiverse ------------------- """

# download one participant MPIDB
def download_mipdb(subject, destination):
    url = f'https://fcp-indi.s3.amazonaws.com/data/Projects/EEG_Eyetracking_CMI_data/compressed/{subject}.tar.gz'

    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)


# delete all unnecessary triggers
def discard_triggers(raw, delete_triggers):
    """Delete all unnecessary triggers from the data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    delete_triggers : list of str
        List of triggers to delete.

    Returns
    -------
    raw : mne.io.Raw
        Raw data with deleted triggers.
    """
    raw.annotations.delete([i for i, x in enumerate(raw.annotations.description) if x in delete_triggers])
    return raw

# delete all unnecessary triggers if defined by stimulus channel and not annotations
def discard_triggers_mipdb(raw, events, delete_triggers):
    """Delete all unnecessary triggers from the data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    delete_triggers : list of str
        List of triggers to delete.

    Returns
    -------
    raw : mne.io.Raw
        Raw data with deleted triggers.
    """
    
    
    raw.annotations.delete([i for i, x in enumerate(raw.annotations.description) if x in delete_triggers])
    return raw

# collate all triggers of the same conditions into the key of the dictionary
def rename_annotations(raw, conditions_triggers):
    """Collate all triggers of the same conditions into the key of the dictionary.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data.
    conditions_triggers : dict
        Dictionary with conditions as keys and triggers as values.

    Returns
    -------
    raw : mne.io.Raw
        Raw data with collated triggers.
    """
    for key, value in conditions_triggers.items():
        raw.annotations.description = np.array(
            [
                key if i in value else i
                for i in raw.annotations.description
            ]
        )
    return raw

# make artificial EOG channels by combining existing channels --> ERPCORE
def recalculate_eog_signal(raw, sfreq=256, has_EOG=True):
    """
    Recalculates the EOG (Electrooculogram) signal by creating HEOG (Horizontal EOG) and VEOG (Vertical EOG) channels.

    Args:
        raw (mne.io.Raw): The raw data containing the original EOG channels.
        sfreq (int): The sampling frequency of the data.
        has_EOG (bool): Whether the data has EOG channels or not.

    Returns:
        mne.io.Raw: The raw data with the recalculated EOG channels.

    """
    #Create HEOG and VEOG channel
    heog_info = mne.create_info(['HEOG'], sfreq, "eog")
    veog_info = mne.create_info(['VEOG'], sfreq, "eog")
    if has_EOG:    
        heog_data = raw['HEOG_left'][0]-raw['HEOG_right'][0]
        veog_data = raw['VEOG_lower'][0]-raw['FP2'][0]
    else:
        heog_data = raw['F9'][0]-raw['F10'][0]
        veog_data = ( raw['Fp1'][0] + raw['Fp2'][0] ) / 2

    heog_raw = mne.io.RawArray(heog_data, heog_info)
    veog_raw = mne.io.RawArray(veog_data, veog_info)
    #Append them to the data
    raw.add_channels([heog_raw, veog_raw],True)
    # delete original EOG channels
    if has_EOG:
        raw.drop_channels([ 'HEOG_left', 'HEOG_right', 'VEOG_lower'])
    
    return raw

# make artificial EOG channels by combining existing channels
def recalculate_eog_signal_128egi(raw):
    """
    Recalculates the EOG (Electrooculogram) signal by creating HEOG (Horizontal EOG) and VEOG (Vertical EOG) channels.
    Study who uses this calculations is: https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-15-68
    Based on 129 channel EGI system. No particular EOG channels have been defined a priori.

    Args:
        raw (mne.io.Raw): The raw data containing the original EOG channels.

    Returns:
        mne.io.Raw: The raw data with the recalculated EOG channels.

    """
    #Create HEOG channel...
    heog_info = mne.create_info(['HEOG'], 250, "eog") # 250 Hz in this study, not 256
    heog_data = raw['E128'][0]-raw['E125'][0]
    heog_raw = mne.io.RawArray(heog_data, heog_info)
    #...and VOEG
    veog_info = mne.create_info(['VEOG'], 250, "eog")
    veog_data = ( (raw['E8'][0]-raw['E126'][0]) + (raw['E25'][0]-raw['E127'][0]) ) / 2
    veog_raw = mne.io.RawArray(heog_data, veog_info)
    #Append them to the data
    raw.add_channels([heog_raw, veog_raw],True)
    # delete original EOG channels
    #raw.drop_channels([ 'HEOG_left', 'HEOG_right', 'VEOG_lower'])
    return raw

# set the Luck et al montage
def set_montage(raw, experiment=None):
    """
    Sets the montage for the Luck et al. raw data.

    Parameters:
    raw (mne.io.Raw): The raw data to set the montage for.

    Returns:
    mne.io.Raw: The raw data with the montage set.
    """
    if experiment == None:
        # rename channels so they match with templates
        raw.rename_channels(dict(FP1 = 'Fp1', FP2 = 'Fp2'))
    
    mont1005 = mne.channels.make_standard_montage('standard_1005')
    # Choose what channels you want to keep
    kept_channels = raw.ch_names

    ind = [i for (i, channel) in enumerate(mont1005.ch_names) if (channel in kept_channels)] # or (channel in add_channels)
    mont = mont1005.copy()

    # Keep only the desired channels
    mont.ch_names = [mont1005.ch_names[x]for x in ind]
    kept_channel_info = [mont1005.dig[x + 3] for x in ind]

    # Keep the first three rows as they are the fiducial points information
    mont.dig = mont1005.dig[:3] + kept_channel_info

    # plot
    #mont.plot()

    # Apply the montage
    raw.set_montage(mont)
    
    return raw



""" ------------------- multiverse ------------------- """

# this to save some information about e.g. dropouts in some pipelines, n interpolated, and so on
class CharacteristicsManager:
    def __init__(self, file_path, force_new=False):
        self.file_path = file_path
        self.characteristics = {}
        if not force_new:
            self._load_characteristics()
        self.save_characteristics()  # Save the file upon initialization

    def _load_characteristics(self):
        if os.path.isfile(self.file_path):
            with open(self.file_path, 'r') as file:
                self.characteristics = json.load(file)

    def save_characteristics(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.characteristics, file, indent=4)

    def update_characteristic(self, key, value):
        # if only 1 level is added
        self.characteristics[key] = value
        self.save_characteristics()

    def get_characteristic(self, key):
        return self.characteristics.get(key, None)

    def update_subfield(self, key, subfield, subfield_value):
        # if more than 1 level is added
        if key not in self.characteristics:
            self.characteristics[key] = {}
        self.characteristics[key][subfield] = subfield_value
        self.save_characteristics()

    def get_subfield(self, key, subfield):
        return self.characteristics.get(key, {}).get(subfield, None)

    def update_subsubfield(self, key, subfield, subsubfield, subsubfield_value):
        # if more than 2 level is added
        if key not in self.characteristics:
            self.characteristics[key] = {}
        if subfield not in self.characteristics[key]:
            self.characteristics[key][subfield] = {}
        self.characteristics[key][subfield][subsubfield] = subsubfield_value
        self.save_characteristics()

    def get_subsubfield(self, key, subfield, subsubfield):
        return self.characteristics.get(key, {}).get(subfield, {}).get(subsubfield, None)


def ica_eog_emg(raw, method='eog'):
    """ 
    ICA to find EOG or EMG artifacts and remove them
    For EOG, EOG channels need to be defined in raw.
    Args:   
        raw (mne.raw): raw data object
        method (str): automated detection of either eye (eog) or muscle (emg) artifacts
        save_ica (bool): save ica object
        save_plot (bool): save ica plots
        save_str (str): string to add to save name
    Returns:
        raw_new (mne.raw): raw data with artifacts regressed out
        n_corr_components (int): number of found components that correlate with eog/emg
    """
    raw_new = raw.copy()

    # HPF (necessary for ICA), l_freq is HPF cutoff, and h_freq is LPF cutoff
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None, n_jobs=-1)

    # ica
    ica = ICA(n_components=20, max_iter="auto", method='picard', random_state=97)
    ica.fit(filt_raw) # bads seem to be ignored by default

    # automatic detection of EOG/EMG components
    ica.exclude = []

    if method == 'eog':
        # find which ICs match the EOG pattern
        indices, scores = ica.find_bads_eog(raw_new)
    elif method == 'emg':
        # find which ICs match the muscle pattern
        indices, scores = ica.find_bads_muscle(raw_new)
    
    # new save expl var    
    if len(indices) > 0:
        explained_var_ratio = ica.get_explained_variance_ratio(filt_raw, components=indices, ch_type="eeg")["eeg"]
    else:
        explained_var_ratio = 0
    
    print(f'Found {len(indices)} independent components correlating with {method.upper()}.')
    print(f'{method.upper()} components explain {int(np.round(explained_var_ratio*100))}% of variance.')
    
    ica.exclude.extend(indices) 

    # because it is an "additive" process, the ica component removel on filtered data 
    # can be used on the unfiltered raw data (source: tutorial)
    # ica.apply() changes the Raw object in-place, so do it on the copy
    ica.apply(raw_new)

    return raw_new, len(indices), explained_var_ratio



def prestim_baseline_correction_ERN(raw, events, event_dict, detrend=None, baseline=200):
    baselines = []
    if baseline == 200: 
        tps = 51 # pre-stimulus timepoints to include
    elif baseline == 400:
        tps = 102
    else:
        raise ValueError("Baseline not implemented.")

    data = raw.get_data()
    for i in range(events.shape[0]):
        stop = events[i, 0]
        start = stop - tps
        this_base = data[:,start:stop].mean(axis=1) # 1 mean value per electrode
        baselines.append(this_base)
    
    # epochieren aller events (ohne baseline correction)
    epochs = mne.Epochs(raw.copy(), 
                        events=events, 
                        event_id=event_dict,
                        tmin=epoch_windows["ERN"][0], 
                        tmax=epoch_windows["ERN"][1],
                        baseline=None, # NEW: no basline correction in this step
                        detrend=detrend, # either None or 1,
                        proj=False,
                        reject_by_annotation=False, 
                        preload=True)
    
    # manuelle baseline correction
    for i in range(1, epochs.events.shape[0]): # use epochs.events instead of events, in case one is dropped during epoching
        epochs._data[i] = epochs._data[i] - baselines[i-1][:, np.newaxis] # remove the previous baseline (stim) from the current trial (resp)
    epochs.drop(0) # drop 1st epoch: because not corrected. It should be a stimulus anyway, and if not, it will be a faulty response

    # verwerfe die stimulus epochen
    epochs = epochs[["correct", "incorrect"]]

    return epochs

# autoreject parser
def autorej(epochs, version="int"):
    """
    Run Autoreject on trials.
    
    Args:
        epochs (mne.epochs): epochs object
        
    Returns:
        epochs_ar (mne.epochs): artifact rejected epochs
        drop_log (pandas.DataFrame): drop log  
    
    """
    if version == "int":
        n_interpolate = [len(epochs.copy().pick('eeg').ch_names)] # must be list for hyperparam opt; pick 'eeg' in case eog channels are still present
        consensus = [len(epochs.copy().pick('eeg').ch_names)]
    elif version == "rej":
        n_interpolate = [0]
        consensus = np.linspace(0.2, 0.8, 4)
    elif version == "intrej":
        n_interpolate = [4, 8, 16, 32]
        consensus = np.linspace(0.2, 0.8, 4)
    

    epochs.del_proj()  # remove proj, don't proj while interpolating (https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html)
    
    # sample 25% of epochs to increase speed
    n_epochs = len(epochs)
    np.random.seed(12) # for subsampling
    indices = np.random.choice(n_epochs, int(n_epochs * 0.25), replace=False)
    epochs_sample = epochs[indices].copy()
    
    # automated estimation of rejection threshold based on channel and trial per participant
    seed=11
    cv=5
    ar = AutoReject(n_interpolate=n_interpolate, 
                    consensus=consensus,
                    random_state=seed,
                    n_jobs=-1, 
                    cv=cv, 
                    verbose=False)
    ar.fit(epochs_sample)
    
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    
    # double check: if the threshold is too high, reduce the values iteratively until at least 25% of epochs are not rejected
    # check how many get rejected
    perc_kept = len(epochs_ar) / len(epochs)
    while perc_kept < 0.2: 
        #print(f"Threshold too conservative, only {perc_kept*100} percent kept. Increase threshold by x percent.")
        #ar.threshes_ = {key: value * 10 for key, value in ar.threshes_.items()} # The sensor-level thresholds with channel names as keys and the peak-to-peak thresholds as the values.
        #epochs_ar, reject_log = ar.transform(epochs.copy(), return_log=True)
        #perc_kept = len(epochs_ar) / len(epochs)
        print(f"Threshold too conservative, only {perc_kept*100} percent kept. Changing random seed.")
        seed += 1
        ar = AutoReject(n_interpolate=n_interpolate, 
                consensus=consensus,
                random_state=seed,
                n_jobs=-1, 
                cv=cv,
                verbose=False)
        ar.fit(epochs_sample)
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)
        perc_kept = len(epochs_ar) / len(epochs)
    
    return epochs_ar, reject_log

# autoreject parser
def autorej_seed(epochs, version="int", sampling_seed=1, ar_seed=1):
    """
    Run Autoreject on trials.
    
    Args:
        epochs (mne.epochs): epochs object
        version (str): "int" for only interpolated, "intrej" for interpolated and rejected
        sampling_seed (int): seed for random sampling of epochs
        ar_seed (int): seed for autoreject
        
    Returns:
        epochs_ar (mne.epochs): artifact rejected epochs
        drop_log (pandas.DataFrame): drop log  
    
    """
    if version == "int":
        n_interpolate = [len(epochs.copy().pick('eeg').ch_names)] # must be list for hyperparam opt; pick 'eeg' in case eog channels are still present
        consensus = [len(epochs.copy().pick('eeg').ch_names)]
    elif version == "rej":
        n_interpolate = [0]
        consensus = np.linspace(0.2, 0.8, 4)
    elif version == "intrej":
        n_interpolate = [4, 8, 16, 32]
        consensus = np.linspace(0.2, 0.8, 4)
    

    epochs.del_proj()  # remove proj, don't proj while interpolating (https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html)
    
    # sample 25% of epochs to increase speed
    n_epochs = len(epochs)
    np.random.seed(sampling_seed) # for subsampling
    indices = np.random.choice(n_epochs, int(n_epochs * 0.25), replace=False)
    epochs_sample = epochs[indices].copy()
    
    # automated estimation of rejection threshold based on channel and trial per participant
    cv=5
    ar = AutoReject(n_interpolate=n_interpolate, 
                    consensus=consensus,
                    random_state=ar_seed,
                    n_jobs=-1, 
                    cv=cv, 
                    verbose=False)
    ar.fit(epochs_sample)
    
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
        
    return epochs_ar, reject_log


# function to count bads / interpolated from AR reject log matrix
def summarize_artifact_interpolation(reject_log, version="int"):
    """if epochs are ONLY interpolated (not rejected),
    this function returns some summaries about which trials and channels
    or how many percent of them are interpolated.

    Args:
        reject_log (reject_log object): ar reject_log object
        version (str): "int" for only interpolated, "intrej" for interpolated and rejected

    Returns:
        dict: interp_frac_channels (key value pair of channel and percentage rejected)
        dict: interp_frac_trials (key value pair of trial and percentage rejected)
        float: total_interp_frac
        dict: rej_frac_channels (key value pair of channel and percentage rejected)
        dict: rej_frac_trials (key value pair of trial and percentage rejected)
        float: total_rej_frac    """
    ch_names = reject_log.ch_names
    armat = reject_log.labels
    
    # for both versions
    armat_binary = np.where(armat == 2.0, 1.0, armat) 
    # 2 means interpolated, 1 would be bad, 0 means ok
    mean_per_channel = np.mean(armat_binary, axis=0)
    mean_per_trial = np.mean(armat_binary, axis=1)
    interp_frac_channels = {channel: value for channel, value in zip(ch_names, mean_per_channel)}
    interp_frac_trials = {channel: value for channel, value in enumerate(mean_per_trial)}
    total_interp_frac = np.mean(mean_per_trial)
    
    # only for intrej
    if version == "intrej":
        armat_binary2 = np.where(armat == 2.0, 0.0, armat) 
        mean_per_channel = np.mean(armat_binary2, axis=0)
        mean_per_trial = np.mean(armat_binary2, axis=1)
        rej_frac_channels = {channel: value for channel, value in zip(ch_names, mean_per_channel)}
        rej_frac_trials = {channel: value for channel, value in enumerate(mean_per_trial)}
        total_rej_frac = np.mean(mean_per_trial)
    else:
        rej_frac_channels = 0
        rej_frac_trials = 0
        total_rej_frac = 0
    
    return interp_frac_channels, interp_frac_trials, total_interp_frac, rej_frac_channels, rej_frac_trials, total_rej_frac


""" ------------------- evoked ------------------- """

def get_forking_paths(
        base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        experiment="N170",
        subject="sub-001", 
        sample=None):
    files = sorted(glob(os.path.join(base_dir, "data", "processed", experiment, subject, "*.fif")))

    if sample:
        np.random.seed(108)
        files = np.random.choice(files, sample, replace=False)

    # get forking paths
    forking_paths = [re.search(r"/([^/]+)-epo\.fif", i).group(1) for i in files]
    
    # split the strings at the hyphen
    forking_paths_split = [i.split("_") for i in forking_paths]
    
    return forking_paths, files, forking_paths_split


""" ------------------- EEGNet (RSVP) ------------------- """
def recode_conditions(epochs, version="superordinate"):
    """ Recode the conditions in the epochs object.
    
    Args:
        epochs (mne.epochs): epochs object
        version (str): "categories" or "superordinate", or "pseudo-superordinate" with pseudo trials
    Returns:
        epochs (mne.epochs): epochs object with recoded conditions
    """
    assert version in ["categories", "superordinate", "pseudo-superordinate"], "Version not recognized."
    
    # if exist, drop -2 
    if "-2" in epochs.event_id.keys():
        # drop -2 epochs
        indices = np.where(epochs.events[:, 2] == 1)[0]
        epochs.drop(indices)
        epochs.event_id.pop("-2")
        
        # also shift the numbers of all other events by -1
        epochs.events[:, 2] = epochs.events[:, 2] - 1
        epochs.event_id = {key: value - 1 for key, value in epochs.event_id.items()}
        
    
    if version == "superordinate":
        epochs = mne.epochs.combine_event_ids(epochs, 
                            old_event_ids=['aquatic', 'bird', 'human', 'insect', 'mammal'], 
                            new_event_id={"animate": 20}, 
                            copy=True)
        epochs = mne.epochs.combine_event_ids(epochs, 
                            old_event_ids=['clothing', 'fruits', 'furniture', 'plants', 'tools'], 
                            new_event_id={"inanimate": 21}, 
                            copy=True)
        epochs.event_id = {"animate": 1, "inanimate": 2}
        epochs.events[:, 2][epochs.events[:, 2] == 20] = 1
        epochs.events[:, 2][epochs.events[:, 2] == 21] = 2
    
    elif version == "pseudo-superordinate":
        # equate stimulus counts
        epo_list = []
        for key in epochs.event_id.keys():
            epo_list.append(epochs[key])
        mne.epochs.equalize_epoch_counts(epo_list, method="mintime")
        equ_epochs = mne.epochs.concatenate_epochs(epo_list)
        # make pseudo trials
        conditions = {
            'animate': ['aquatic', 'bird', 'human', 'insect', 'mammal'],
            'inanimate':  ['clothing', 'fruits', 'furniture', 'plants', 'tools'],
        }
        
        n_per_cond = len(epo_list[0])
        pseudo_epochs = []
        for i in range(n_per_cond):
            for c, condition in enumerate(conditions):
                one_pseudo_epoch = []
                for category in conditions[condition]:
                    one_pseudo_epoch.append(equ_epochs[category][i])
                # average over the pseudo trials
                #break
            #break
                one_pseudo_epoch = mne.epochs.concatenate_epochs(one_pseudo_epoch)
                one_pseudo_evoked = one_pseudo_epoch.average(method="mean")
                # make an single epoch from the evoked
                event_times = [1]  # Example event times in seconds
                event_id=c
                events = [[int(time * one_pseudo_evoked.info['sfreq']), 0, event_id] for time in event_times]
                pseudo_epochs.append(
                    mne.EpochsArray(np.expand_dims(one_pseudo_evoked.get_data(),0),
                                    epochs.info, 
                                    events=events, 
                                    event_id={condition: event_id}, 
                                    tmin=epochs.tmin, #tmax=.8, 
                                    baseline=epochs.baseline)
                )

        pseudo_epochs = mne.concatenate_epochs(pseudo_epochs)        
        return pseudo_epochs            
    
    # version categories: leave as is
    
    return epochs


""" 5b sliding groups and 4z-aggregate_results """

def get_age():
    dfage = pd.read_csv("/u/kroma/m4d/data/mipdb/participants.csv")
    bins = [6, 9, 11, 13, 17, float('inf')]  # The last bin is for 18+
    labels = ['6-9', '10-11', '12-13', '14-17', '18+'] # these are the official groups (without 25+ ers)
    # Bin the 'Age' column
    dfage['Age_Group'] = pd.cut(dfage['Age'], bins=bins, labels=labels)
    return dfage


""" 5b sliding groups """


def cluster_test(data, side=1):
    # Pivot the DataFrame
    wide_data = data.pivot(index='times', columns='subject', values='balanced accuracy')
    # Reset the index to make 'times' a regular column
    wide_data = wide_data.reset_index()
    times = wide_data['times']
    wide_data.drop(labels=['times'], axis=1, inplace = True)
    X = wide_data.values.T
    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
                                                X-0.5, # to make it zero centered (theoretically) 
                                                #threshold=None, # OLD: None = automatically determines the threshold for p=0.05
                                                threshold={"start": 0, "step": 0.2}, # NEW: TFCE
                                                n_jobs=-1, # NEW: use all cores
                                                n_permutations=1024, 
                                                tail=side, # 0: two-sided, 1: right, -1: left
                                                stat_fun=None, adjacency=None, 
                                                seed=None, max_step=1, exclude=None, step_down_p=0, 
                                                t_power=1, out_type='indices', check_disjoint=False, 
                                                buffer_size=1000, 
                                                verbose='ERROR')
    return t_obs, clusters, cluster_pv, H0, times 