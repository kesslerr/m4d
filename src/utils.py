import os
import numpy as np
import json
import mne
from glob import glob
import re
from mne.preprocessing import ICA
from autoreject import AutoReject, get_rejection_threshold, read_auto_reject

""" ------------------- pre-multiverse ------------------- """

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

# make artificial EOG channels by combining existing channels
def recalculate_eog_signal(raw):
    """
    Recalculates the EOG (Electrooculogram) signal by creating HEOG (Horizontal EOG) and VEOG (Vertical EOG) channels.

    Args:
        raw (mne.io.Raw): The raw data containing the original EOG channels.

    Returns:
        mne.io.Raw: The raw data with the recalculated EOG channels.

    """
    #Create HEOG channel...
    heog_info = mne.create_info(['HEOG'], 256, "eog")
    heog_data = raw['HEOG_left'][0]-raw['HEOG_right'][0]
    heog_raw = mne.io.RawArray(heog_data, heog_info)
    #...and VOEG
    veog_info = mne.create_info(['VEOG'], 256, "eog")
    veog_data = raw['VEOG_lower'][0]-raw['FP2'][0]
    veog_raw = mne.io.RawArray(heog_data, veog_info)
    #Append them to the data
    raw.add_channels([heog_raw, veog_raw],True)
    # delete original EOG channels
    raw.drop_channels([ 'HEOG_left', 'HEOG_right', 'VEOG_lower'])
    
    return raw

# set the Luck et al montage
def set_montage(raw):
    """
    Sets the montage for the Luck et al. raw data.

    Parameters:
    raw (mne.io.Raw): The raw data to set the montage for.

    Returns:
    mne.io.Raw: The raw data with the montage set.
    """
    # rename channels so they match with templates
    raw.rename_channels(dict(FP1 = 'Fp1', FP2 = 'Fp2'))

    mont1020 = mne.channels.make_standard_montage('standard_1020')
    # Choose what channels you want to keep
    kept_channels = raw.ch_names

    ind = [i for (i, channel) in enumerate(mont1020.ch_names) if (channel in kept_channels)] # or (channel in add_channels)
    mont = mont1020.copy()

    # Keep only the desired channels
    mont.ch_names = [mont1020.ch_names[x]for x in ind]
    kept_channel_info = [mont1020.dig[x + 3] for x in ind]

    # Keep the first three rows as they are the fiducial points information
    mont.dig = mont1020.dig[:3] + kept_channel_info

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
    
    print(f'Found {len(indices)} independent components correlating with {method.upper()}.')
    ica.exclude.extend(indices) 

    # because it is an "additive" process, the ica component removel on filtered data 
    # can be used on the unfiltered raw data (source: tutorial)
    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    ica.apply(raw_new)

    return raw_new, len(indices)


# autoreject parser
def autorej(epochs):
    """
    Run Autoreject on trials.
    
    Args:
        epochs (mne.epochs): epochs object
        
    Returns:
        epochs_ar (mne.epochs): artifact rejected epochs
        drop_log (pandas.DataFrame): drop log  
    
    """
    n_interpolate = [len(epochs.copy().pick('eeg').ch_names)] # must be list for hyperparam opt; pick 'eeg' in case eog channels are still present
    consensus = [len(epochs.copy().pick('eeg').ch_names)]

    epochs.del_proj()  # remove proj, don't proj while interpolating (https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html)
    
    # automated estimation of rejection threshold based on channel and trial per participant
    ar = AutoReject(n_interpolate=n_interpolate, 
                    consensus=consensus,
                    random_state=11,
                    n_jobs=-1, 
                    verbose=False)
    ar.fit(epochs)  # fit only a few epochs if you want to save time
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)

    return epochs_ar, reject_log

# function to count bads / interpolated from AR reject log matrix
def summarize_artifact_interpolation(reject_log):
    """if epochs are ONLY interpolated (not rejected),
    this function returns some summaries about which trials and channels
    or how many percent of them are interpolated.

    Args:
        reject_log (reject_log object): ar reject_log object

    Returns:
        dict: interp_frac_channels (key value pair of channel and percentage rejected)
        dict: interp_frac_trials (key value pair of trial and percentage rejected)
        float: total_interp_frac
    """
    ch_names = reject_log.ch_names
    armat = reject_log.labels
    armat_binary = np.where(armat == 2.0, 1.0, armat) 
    # 2 means interpolated, 1 would be bad(?), but we then interpolate anyway therfore not in the data, 0 means ok
    mean_per_channel = np.mean(armat_binary, axis=0)
    mean_per_trial = np.mean(armat_binary, axis=1)
    interp_frac_channels = {channel: value for channel, value in zip(ch_names, mean_per_channel)}
    interp_frac_trials = {channel: value for channel, value in enumerate(mean_per_trial)}
    total_interp_frac = np.mean(mean_per_trial)
    return interp_frac_channels, interp_frac_trials, total_interp_frac


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
                                                threshold=None, 
                                                n_permutations=1024, 
                                                tail=side, # 0: two-sided, 1: right, -1: left
                                                stat_fun=None, adjacency=None, n_jobs=None, 
                                                seed=None, max_step=1, exclude=None, step_down_p=0, 
                                                t_power=1, out_type='indices', check_disjoint=False, 
                                                buffer_size=1000, verbose=None)
    return t_obs, clusters, cluster_pv, H0, times 