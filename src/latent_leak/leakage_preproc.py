import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
import numpy as np
import itertools
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timezone # workaround for mne

from mne.preprocessing import ICA
from autoreject import AutoReject

# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#ptmp_dir = "/ptmp/kroma/m4d/"
os.chdir(base_dir)
sys.path.append(base_dir)

from src.utils import CharacteristicsManager, ica_eog_emg, autorej, summarize_artifact_interpolation, prestim_baseline_correction_ERN
from src.config import multiverse_params, epoch_windows, baseline_windows, translation_table

""" HEADER END """

experiment = "N170"# "N170"

""" 
Latent leakage -> would it affect the result?
The following is a quick and dirty implementation, to show the principle.
To test other steps or paradigms, it needs major adaption.
"""

# DEBUG
#subject = "sub-001"
#stragglers = 10 # number of stragglers to be processed in the end


# define subject and session by arguments to this script
if len(sys.argv) != 2:
    print("Usage: python script.py subject")
    sys.exit(1)
else:
    subject = sys.argv[1]
    print(f'Processing Experiment {experiment} Subject {subject}!')


""" SPECIFICATIONS END """
raw_folder = os.path.join(base_dir, "data", "raw", experiment)

interim_folder = os.path.join(base_dir, "data", "interim", "leakage", experiment, subject)
if not os.path.exists(interim_folder):
    os.makedirs(interim_folder)
processed_folder = f"/ptmp/kroma/m4d/data/processed/leakage/{experiment}/{subject}" #os.path.join(base_dir, "data", "processed", experiment, subject)
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)
# delete all files in processed_folder and interim_folder
print("Deleting all files in processed and interim folder.")
_ = [os.remove(file) for file in glob(os.path.join(interim_folder, "*"))]
_ = [os.remove(file) for file in glob(os.path.join(processed_folder, "*"))]

# read raw data
raw = mne.io.read_raw_fif(os.path.join(raw_folder, f"{subject}-raw.fif"), preload=True, verbose=None)


# get events again, because it doesn't survive i/o operations
events, event_dict = mne.events_from_annotations(raw)     

# new: apply Cz reference (preliminary) for all datasets, re-reference later in some pipelines
raw = raw.set_eeg_reference(['Cz'], projection=False)    


""" make 5 splits of the data """

# def split_continuous(raw, n_splits=5):
#     """
#     Splits the continuous raw data into n_splits equal parts.
#     """
#     # Get the total duration of the raw object
#     total_duration = raw.times[-1]  # Last time point in seconds

#     # Compute split boundaries
#     split_times = np.linspace(0, total_duration, num=n_splits + 1)  # n_splits + 1 points for n_splits segments

#     # Create n_splits equally long Raw objects
#     raw_splits = []
#     for i in range(n_splits):
#         start_time = split_times[i]
#         end_time = split_times[i + 1]
        
#         raw_part = raw.copy().crop(tmin=start_time, tmax=end_time)
#         raw_splits.append(raw_part)

#     return raw_splits


# CAUTION:
# in this version of MNE there seems to be a bug
# cropping, the timing is not adjusted in the annotations


def split_continuous_edge(raw, n_splits=5, start_margin=0.2, end_margin=1.0):
    """
    Splits the continuous raw data into n_splits equal parts and removes annotations 
    within `start_margin` seconds of the start and `end_margin` seconds of the end of each segment.
    
    Parameters:
    - raw: MNE Raw object
    - n_splits: int, number of equal splits
    - start_margin: float, time in seconds around the segment start where annotations will be removed
    - end_margin: float, time in seconds around the segment end where annotations will be removed
    
    Returns:
    - List of Raw objects with adjusted annotations
    """
    total_duration = raw.times[-1]  # Last time point in seconds
    split_times = np.linspace(0, total_duration, num=n_splits + 1)  # Boundaries for splits
    raw_splits = []
    #raw.set_meas_date(datetime.now(tz=timezone.utc))
    # DEBUG
    # raw=raw0.copy()
    # n_splits=5
    # start_margin=0.2
    # end_margin=1.0
    # i=2
    for i in range(n_splits):
        print(f"iteration {i} of {n_splits}")
        start_time, end_time = split_times[i], split_times[i + 1]
        raw_part = raw.copy().crop(tmin=start_time, tmax=end_time)

        # # Filter annotations: keep only those NOT within the defined margins
        # new_annotations = []
        # # No need to get first_onset because cropping already adjusts onsets
        # for annot in raw_part.annotations:
        #     annot_start = annot['onset']
        #     annot_end = annot_start + annot['duration']

        #     # Remove annotations near the beginning or end of the segment.
        #     if not (0 <= annot_start <= start_margin or 
        #             (raw_part.times[-1] - end_margin) <= annot_end <= raw_part.times[-1]):
        #         new_annotations.append(annot)

        # # Set the new annotations without further shifting
        # raw_part.set_annotations(mne.Annotations(
        #     onset=[a['onset'] for a in new_annotations],
        #     duration=[a['duration'] for a in new_annotations],
        #     description=[a['description'] for a in new_annotations]
        # ))
        # #raw_part.set_meas_date(datetime.now(tz=timezone.utc))

        # Create annotations for the bad segments
        t_end = raw_part.times[-1]
        #bad_annotations = mne.Annotations(
        #    onset=[0, t_end - 1.0],           # start at 0s and (end - 1s)
        #    duration=[0.2, 1.0],              # durations of 0.2s and 1s respectively
        #    description=['BAD', 'BAD']        # mark both as BAD segments
        #)

        # Add these annotations to your raw data (this replaces existing annotations;
        # if you want to append, use raw.annotations.append or combine them)
        #raw_part.annotations.append(bad_annotations)
        raw_part.annotations.append(onset=0, duration=0.2, description='BAD')
        raw_part.annotations.append(onset=t_end - 1.0, duration=1.0, description='BAD')

        # Debug check
        print(f"Split {i} annotations:", raw_part.annotations)
        print("Raw segment time range:", raw_part.times[0], raw_part.times[-1])

        assert len(raw_part.annotations) > 4, f"Not enough annotations in split {i}: {len(raw_part.annotations)}"
        raw_splits.append(raw_part.copy()) # DEBUG: maybe this is overwritten always?

    return raw_splits

# DEBUG:
# sealed_raw_splits = split_continuous_edge(raw0.copy(), n_splits=5)
# for i in sealed_raw_splits:
#     print(i.annotations)
    
# raw_splits = split_continuous_edge(raw.copy(), n_splits=5)
# for i in raw_splits:
#     print(i.annotations)
    
# # n=4er
# train_raw_ica_split
# for i in train_raw_ica_split:
#     print(i.annotations)


def split_epochs(epochs, n_splits=5):
    """
    Splits the epochs into n_splits equal parts, stratified by event type.
    """
    # Get the unique event IDs
    unique_event_ids = np.unique(epochs.events[:, 2])  # Column 2 contains event IDs

    # Create a list to hold the split epochs
    split_epochs_list = []

    # Iterate over each unique event ID
    for event_id in unique_event_ids:
        split_epochs_this_condition = []
        
        # Get the indices of the epochs corresponding to this event ID
        event_indices = np.where(epochs.events[:, 2] == event_id)[0]

        # Split the indices into n_splits equal parts
        split_indices = np.array_split(event_indices, n_splits)

        # Create a new Epochs object for each split
        for i, indices in enumerate(split_indices):
            split_epochs_i = epochs[indices]
            split_epochs_this_condition.append(split_epochs_i)

        split_epochs_list.append(split_epochs_this_condition)
    
    # now concatenate one element from each condition
    split_epochs_list = [mne.concatenate_epochs([split_epochs_list[j][i] for j in range(len(split_epochs_list))]) for i in range(n_splits)]

    return split_epochs_list

# # find all events in each raw
# events_splits = []
# for i, raw_part in enumerate(raw_splits):
#     events_part, event_dict_part = mne.events_from_annotations(raw_part)
#     events_splits.append(events_part)

#https://mne.tools/0.24/generated/mne.epochs.equalize_epoch_counts.html

""" test leakage via HPF """

raw_splits = split_continuous_edge(raw.copy(), n_splits=5)
# Now, raw_splits contains 5 Raw objects of equal duration

# sealed

raw_sealed = []
epochs_sealed = []
for split in raw_splits:
    # highpass filter: use 0.1 version because it is >30 seconds long!
    this_raw_sealed = split.copy().filter(l_freq=0.1, h_freq=None, method='fir', fir_design='firwin', skip_by_annotation='EDGE boundary', n_jobs=-1)
    raw_sealed.append(this_raw_sealed) 
    
    this_events, event_dict = mne.events_from_annotations(split)
    this_epochs = mne.Epochs(this_raw_sealed.copy(), 
                        events=this_events, 
                        event_id=event_dict,
                        tmin=epoch_windows[experiment][0], 
                        tmax=epoch_windows[experiment][1],
                        baseline=(-0.2, 0), #None, # for ICA baseline correction is not recommendet (-0.2, 0),
                        detrend=None,
                        proj=False,
                        reject_by_annotation=False, 
                        preload=True)

    epochs_sealed.append(this_epochs)
                 
# leaky
raw_hpf = raw.copy().filter(l_freq=0.1, h_freq=None, method='fir', fir_design='firwin', skip_by_annotation='EDGE boundary', n_jobs=-1) 
raw_leaky = split_continuous_edge(raw_hpf.copy(), n_splits=5)
                   
epochs_leaky = []
for split in raw_leaky:
    this_events, event_dict = mne.events_from_annotations(split)    
    this_epochs = mne.Epochs(split.copy(), 
                        events=this_events, 
                        event_id=event_dict,
                        tmin=epoch_windows[experiment][0], 
                        tmax=epoch_windows[experiment][1],
                        baseline=(-0.2, 0), #None, # for ICA baseline correction is not recommendet (-0.2, 0),
                        detrend=None,
                        proj=False,
                        reject_by_annotation=False, 
                        preload=True)

    epochs_leaky.append(this_epochs)
                                    
# save for later ML
for i, (leaky, sealed) in enumerate(zip(epochs_leaky, epochs_sealed)):
    assert len(leaky) == len(sealed), f"Leaky and sealed epochs have different n of epochs: {len(leaky)} vs {len(sealed)}"
    print(f"{len(leaky)}=={len(sealed)}")
    leaky.save(os.path.join(processed_folder, f"leaky_hpf_{i}-epo.fif"), overwrite=True)
    sealed.save(os.path.join(processed_folder, f"sealed_hpf_{i}-epo.fif"), overwrite=True)




""" test leakage through ocular ICA 

first test showed that on epochs ICA did not work so well. so instead we used concatenated raw timeseries
"""

# save number of dropped ic components per analysis
df_dropped_components = pd.DataFrame(columns=["subject", "experiment", "pipeline", "i", "n_components_dropped"])

raw0 = raw.copy().filter(l_freq=0.5, h_freq=45, method='fir', fir_design='firwin', skip_by_annotation='EDGE boundary', n_jobs=-1) # highpass filter to remove slow drifts
# 0.5 to be close to what is done in my 2-filter approach in the multiverse (1 hz)
# avoid doing multiple filters here in this toy example

# leaky
leaky_raw = raw0.copy()

ica = ICA(n_components=20, max_iter="auto", method='picard', random_state=97)
ica.fit(leaky_raw) # bads seem to be ignored by default
ica.exclude = []
indices, scores = ica.find_bads_eog(leaky_raw)
print(f'Found {len(indices)} independent components.')

df_dropped_components = pd.concat([df_dropped_components, 
                                   pd.DataFrame({"subject": [subject], 
                                                  "experiment": [experiment], 
                                                  "pipeline": ["leaky"],
                                                  "i": [0],
                                                  "n_components_dropped": [len(indices)]})], 
                                  ignore_index=True)

ica.exclude.extend(indices) 
ica.apply(leaky_raw)

leaky_raw_splits = split_continuous_edge(leaky_raw.copy(), n_splits=5)
epochs_leaky = []
for i, split in enumerate(leaky_raw_splits):
    this_events, event_dict = mne.events_from_annotations(split)    
    this_epochs = mne.Epochs(split.copy(), 
                        events=this_events, 
                        event_id=event_dict,
                        tmin=epoch_windows[experiment][0], 
                        tmax=epoch_windows[experiment][1],
                        baseline=(-0.2, 0), #None, # for ICA baseline correction is not recommendet (-0.2, 0),
                        detrend=None,
                        proj=False,
                        reject_by_annotation=False, 
                        preload=True)

    epochs_leaky.append(this_epochs)
    this_epochs.save(os.path.join(processed_folder, f"leaky_ica_{i}-epo.fif"), overwrite=True)
epochs_leaky = mne.concatenate_epochs(epochs_leaky)


# sealed
sealed_raw_splits = split_continuous_edge(raw0.copy(), n_splits=5)

# DEBUG
#events, event_dict = mne.events_from_annotations(raw0)
#mne.viz.plot_events(events, event_id=event_dict, sfreq=raw0.info['sfreq'], show=True)

len_train = []
len_test  = []
for i, split in enumerate(sealed_raw_splits):
    # DEBUG
    #if i != 1:
    #    continue
    
    # concatenate 4 splits (train), and leave current split (test)
    train_splits = [sealed_raw_splits[j].copy() for j in range(len(sealed_raw_splits)) if j != i]
    train_raw = mne.concatenate_raws(train_splits.copy())
    
    train_raw_ica = train_raw.copy()
    test_raw_ica = split.copy()
    
    ica = ICA(n_components=20, max_iter="auto", method='picard', random_state=97)
    ica.fit(train_raw_ica) # bads seem to be ignored by default
    ica.exclude = []
    indices, scores = ica.find_bads_eog(train_raw_ica)
    print(f'Found {len(indices)} independent components.')
    df_dropped_components = pd.concat([df_dropped_components, 
                                   pd.DataFrame({"subject": [subject], 
                                                  "experiment": [experiment], 
                                                  "pipeline": ["sealed"],
                                                  "i": [i],
                                                  "n_components_dropped": [len(indices)]})], 
                                  ignore_index=True)
    ica.exclude.extend(indices) 
    ica.apply(train_raw_ica) # apply on train
    ica.apply(test_raw_ica) # apply on test
    
    # epoching
    # train
    
    
    #train_events = [e for j, e in enumerate(events_splits) if j != i]
    #train_events = np.concatenate(train_events)
    # TODO: first, need to recalculate the timings of the events, else a lot of events will be dropped,
    # and the timings will be incorrect (more incorrect for lower i's)
    

    # now also make it on 4 (!) splits, to have the same amount of drops from the border regions
    train_raw_ica_split = split_continuous_edge(train_raw_ica.copy(), n_splits=4)
    # DEBUG TODO delete:
    #train_events, train_event_dict = mne.events_from_annotations(train_raw_ica)   
    #train_events, train_event_dict = mne.events_from_annotations(train_raw)

    train_epochs = []
    for j, split2 in enumerate(train_raw_ica_split):
        train_events, train_event_dict = mne.events_from_annotations(split2) 
        print("train_event_dict:", train_event_dict)
  
        assert train_event_dict == {'cars': 1, 'faces': 2}, "Event dict is not correct, need to recalculate the timings of the events"
          
        train_epochs_i = mne.Epochs(split2.copy(), 
                            events=train_events, 
                            event_id=train_event_dict,
                            tmin=epoch_windows[experiment][0], 
                            tmax=epoch_windows[experiment][1],
                            baseline=(-0.2, 0), #None, # for ICA baseline correction is not recommendet (-0.2, 0),
                            detrend=None,
                            proj=False,
                            reject_by_annotation=False, 
                            preload=True)
        train_epochs.append(train_epochs_i)
    train_epochs = mne.concatenate_epochs(train_epochs)
    
    # test
    test_events,  test_event_dict  = mne.events_from_annotations(test_raw_ica)    
    assert test_event_dict == {'cars': 1, 'faces': 2}, "Event dict is not correct, need to recalculate the timings of the events"
 
    test_epochs = mne.Epochs(test_raw_ica.copy(), 
                    events=test_events, 
                    event_id=test_event_dict,
                    tmin=epoch_windows[experiment][0], 
                    tmax=epoch_windows[experiment][1],
                    baseline=(-0.2, 0), #None, # for ICA baseline correction is not recommendet (-0.2, 0),
                    detrend=None,
                    proj=False,
                    reject_by_annotation=False, 
                    preload=True)
    
    # save DEBUG TODO uncomment
    train_epochs.save(os.path.join(processed_folder, f"sealed_ica_train_{i}-epo.fif"), overwrite=True)
    test_epochs.save(os.path.join(processed_folder, f"sealed_ica_test_{i}-epo.fif"), overwrite=True)
    len_train.append(len(train_epochs))
    len_test.append(len(test_epochs))
    
    # TODO
    #if i>0:
    #    break

total_lengths = [i + j for i, j in zip(len_train, len_test)]
print(f"Total lengths: {total_lengths}")
assert len(np.unique(total_lengths)) == 1, f"Total lengths are not equal: {total_lengths}"

print(df_dropped_components)
df_dropped_components.to_csv(os.path.join(processed_folder, f"ica_dropped_components.csv"), index=False)  



""" leakage through autoreject """

# save number of interpolated channels per analysis
df_interpolated = pd.DataFrame(columns=["subject", "experiment", "pipeline", "i", "fraction_interpolated"])

# leaky pipeline


# use raw0
leaky_epochs = mne.Epochs(raw0.copy(), 
                events=events, 
                event_id=event_dict,
                tmin=epoch_windows[experiment][0], 
                tmax=epoch_windows[experiment][1],
                baseline=(-0.2, 0), #None, # for ICA baseline correction is not recommendet (-0.2, 0),
                detrend=None,
                proj=False,
                reject_by_annotation=False, 
                preload=True)

# just take 20% for hyperparameter tuning in autoreject
# intrej version:
#n_interpolate = [4, 8, 16, 32]
#consensus = np.linspace(0.2, 0.8, 4)
# int version:
n_interpolate = [len(leaky_epochs.copy().pick('eeg').ch_names)] # must be list for hyperparam opt; pick 'eeg' in case eog channels are still present
consensus = [len(leaky_epochs.copy().pick('eeg').ch_names)]

leaky_epochs.del_proj()  # remove proj, don't proj while interpolating (https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html)

# sample 20% of epochs to increase speed
n_epochs = len(leaky_epochs)
np.random.seed(12) # for subsampling
indices = np.random.choice(n_epochs, int(n_epochs * 0.2), replace=False)
epochs_sample = leaky_epochs[indices].copy()

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

epochs_ar_leaky, reject_log = ar.transform(leaky_epochs, return_log=True)

reject_log.plot('horizontal')

armat = reject_log.labels
armat = armat[~np.isnan(armat)] # remove nans
rej_frac = np.count_nonzero(armat) / len(armat) #get the fraction of non-zero values

df_interpolated = pd.concat([df_interpolated, 
                                pd.DataFrame({"subject": [subject], 
                                             "experiment": [experiment], 
                                             "pipeline": ["leaky"],
                                             "i": [0],
                                             "fraction_interpolated": [rej_frac]})], 
                                ignore_index=True)

leaky_epochs_ar_split = split_epochs(epochs_ar_leaky, n_splits=5)

for i, split in enumerate(leaky_epochs_ar_split):
    split.save(os.path.join(processed_folder, f"leaky_ar_{i}-epo.fif"), overwrite=True)


# sealed pipeline

sealed_epochs_split = split_epochs(leaky_epochs.copy(), n_splits=5)
len_train = []
len_test  = []
for i, split in enumerate(sealed_epochs_split): # i is the test split, rest is used for training

    # train data
    train_splits = [sealed_epochs_split[j] for j in range(len(sealed_epochs_split)) if j != i]
    train_epochs = mne.concatenate_epochs(train_splits)
    # sample 20% of epochs to increase speed
    n_epochs = len(train_epochs)
    np.random.seed(12) # for subsampling
    indices = np.random.choice(n_epochs, int(n_epochs * 0.25), replace=False) # here it is 25%, because it's only 80% of the whole data, so to get 20% of entire data, use here 25%
    epochs_sample = train_epochs[indices].copy()

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

    train_epochs_ar, reject_log1 = ar.transform(train_epochs, return_log=True)
    test_epochs_ar, reject_log2 = ar.transform(split, return_log=True)
    
    reject_log1.plot('horizontal')
    reject_log2.plot('horizontal')

    armat = reject_log1.labels
    armat = armat[~np.isnan(armat)] # remove nans
    rej_frac1 = np.count_nonzero(armat) / len(armat) #get the fraction of non-zero values
    armat = reject_log2.labels
    armat = armat[~np.isnan(armat)] # remove nans
    rej_frac2 = np.count_nonzero(armat) / len(armat) #get the fraction of non-zero values
    rej_frac = (rej_frac1 * 4 + rej_frac2) / 5

    df_interpolated = pd.concat([df_interpolated, 
                                    pd.DataFrame({"subject": [subject], 
                                                "experiment": [experiment], 
                                                "pipeline": ["sealed"],
                                                "i": [i],
                                                "fraction_interpolated": [rej_frac]})], 
                                    ignore_index=True)

    train_epochs_ar.save(os.path.join(processed_folder, f"sealed_ar_train_{i}-epo.fif"), overwrite=True)
    test_epochs_ar.save(os.path.join(processed_folder, f"sealed_ar_test_{i}-epo.fif"), overwrite=True)
    len_train.append(len(train_epochs_ar))
    len_test.append(len(test_epochs_ar))

# for intrej (not used currently), the following test wouldnt work
total_lengths = [i + j for i, j in zip(len_train, len_test)]
print(f"Total lengths: {total_lengths}")
assert len(np.unique(total_lengths)) == 1, f"Total lengths are not equal: {total_lengths}"


print(df_interpolated)
df_interpolated.to_csv(os.path.join(processed_folder, f"autoreject_interpolated.csv"), index=False)




# """ ADDON: later maybe: HPF test on continuous data, but this is a headache to implement correctly """
# """ find the distribution of events for splitting the raw data """


# # Extract event times (convert sample indices to seconds)
# event_times_sec = events[:, 0] / raw.info['sfreq']

# # Get unique event types
# unique_event_ids = np.unique(events[:, 2])  # Column 2 contains event IDs

# # Compute cumulative counts for each event type
# plt.figure(figsize=(10, 6))

# for event_id in unique_event_ids:
#     # Get times for this event type
#     event_times = event_times_sec[events[:, 2] == event_id]
    
#     # Compute cumulative sum
#     cumulative_counts = np.arange(1, len(event_times) + 1)
    
#     # Plot
#     plt.plot(event_times, cumulative_counts, label=f"Event {event_id}")

# # Labels and legend
# plt.xlabel("Time (seconds)")
# plt.ylabel("Cumulative Event Count")
# plt.title("Cumulative Sum of Events Over Time")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.6)

# # Show the plot
# plt.show()

# # The events are fully random, so we can not split easily and keep class balance.
# # We do it with imbalance instead, and drop randomly the overhead trials
# # remember the exact trials that were dropped, to also drop them drom the continuous experiment



