""" Multiverse preprocessing """

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
import pickle
from glob import glob
if sys.platform.startswith('darwin'):
    mne.viz.set_browser_backend('qt', verbose=None) # 'qt' or 'matplotlib'

# go to base directory and import globals
#os.chdir(os.path.dirname(os.getcwd())) # open base_dir to be able to import from src
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.append(base_dir)
from src.utils import discard_triggers, recalculate_eog_signal, recalculate_eog_signal_128egi, set_montage, rename_annotations
from src.config import subjects, experiments, delete_triggers, conditions_triggers, cichy_subjects_infants, cichy_subjects_adults, subjects_mipdb
from src.exceptions import exception_pre_preprocessing_annotations


""" HEADER END """

# DEBUG
# experiment = "MMN"
# subject = "sub-001"

""" CHILD MIND INSTITUTE """

errors = []

#subject = subjects_mipdb[0]
#for subject in subjects_mipdb:
for subject in errors[-2:]:

    print(f"Processing subject {subject}...")

    # file paths
    download_folder = os.path.join(base_dir, "data", "mipdb")
    raw_folder = os.path.join(base_dir, "data", "raw", "MIPDB", subject)
    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder)

    # read raw data
    raw_file = glob(os.path.join(download_folder, f"{subject}*.raw")) # only the relevant tasks should be here
    
    if len(raw_file) >= 3:
        pass
    else:
        errors.append(subject)
        continue
    
    #assert len(raw_file) >= 3, "Not at least 3 files available!"
    #raw_file = raw_file[0] # TODO loop over all

    # raw = mne.io.read_raw(raw_file, eog=(), preload=True, 
    #                             uint16_codec='utf-8',
    #                             #montage_units='auto',
    #                             verbose=None)

    # load and concatenate all raw_files
    raw = mne.io.read_raw_egi(raw_file[0])
    for file in raw_file[1:]:
        raw = mne.io.read_raw_egi(file)
        raw.append(raw)

    #raw = mne.io.read_raw_egi(raw_file)

    # events from stimulus channel
    events = mne.find_events(raw, stim_channel="STI 014")

    # only keep events which were "12 ", "13  "
    try:
        events_of_interest = raw.event_id["12  "], raw.event_id["13  "]
    except:
        print(f"Subject {subject} has no events 12 AND 13. Skipping this Subject.")
        continue
    # from events, only keep 3rd column which is one of the events_of_interest elements
    events = events[np.isin(events[:, 2], events_of_interest)]
    # replace "12 ", "13  " with "left", "right" in raw.event_id
    raw.event_id = {"left": raw.event_id["12  "], "right": raw.event_id["13  "]}


    #Downsample from the recorded sampling rate of 1024 Hz to 256 Hz to speed data processing
    raw, events = raw.resample(250, events = events) # TODO: remember, this is 250, whereas erpcore is 256
    raw.events = events

    # get and save the event counts
    event_counts = {}
    for key in raw.event_id.keys():
        event_counts[key] = len(events[events[:, 2] == raw.event_id[key]])
        print(key, len(events[events[:, 2] == raw.event_id[key]]))
    raw.event_counts = event_counts

    # save events and events_id and event_counts
    with open(os.path.join(raw_folder, "event_counts.pck"), "wb") as file:
        pickle.dump(event_counts, file)
    with open(os.path.join(raw_folder, "events.pck"), "wb") as file:
        pickle.dump(events, file)
    with open(os.path.join(raw_folder, "event_id.pck"), "wb") as file:
        pickle.dump(raw.event_id, file)
        
    print(f"{subject}: {raw.event_counts}")

    # recalculate EOG channels
    raw = recalculate_eog_signal_128egi(raw.copy())

    # set montage
    montage = mne.channels.read_custom_montage(os.path.join(base_dir, "data", "mipdb", "GSN_HydroCel_129.sfp"))
    raw.set_montage(montage, on_missing="warn") # TODO: is it correct? because E129 was missing .... 

    # TODO plot montage in 99-misc.py

    # save raw data
    raw.save(os.path.join(raw_folder, f"{subject}-raw.fif"), overwrite=True)


""" CICHY """
# experiment = "paperclip"
# group = "adults"
# #subject = "sub-01" # cichy only uses 1 leading 0

# for subject in cichy_subjects_adults:
#     # file paths
#     if group == "infants":
#         download_folder = os.path.join(base_dir, "data", "cichy", group, subject) # TODO: infants no folder "eeg"
#     elif group == "adults":
#         download_folder = os.path.join(base_dir, "data", "cichy", group, subject, "eeg")
#     raw_folder = os.path.join(base_dir, "data", "raw", f"{experiment}_{group}")
#     if not os.path.exists(raw_folder):
#         os.makedirs(raw_folder)

#     # read raw data
#     raw_file = glob(os.path.join(download_folder, "*.vhdr"))
#     assert len(raw_file) == 1, "More than one raw file found!"
#     raw_file = raw_file[0]
#     raw = mne.io.read_raw_brainvision(raw_file, eog=(), preload=True, # TODO: check if brainvision is correct for all groups
#                                 #montage_units='auto',
#                                 verbose=None)
    
#     raw = mne.io.read_raw_eeglab(raw_file, eog=(), preload=True, # TODO: check if brainvision is correct for all groups
#                             #montage_units='auto',
#                             verbose=None)
#     # TODO: CONTINUE HERE: INFANT MATLAB ERROR

#     # discard unnecessary triggers
#     raw = discard_triggers(raw.copy(), delete_triggers[experiment])

#     # collate triggers into conditions in annotations
#     raw = rename_annotations(raw.copy(), conditions_triggers[experiment])
#     #raw = exception_pre_preprocessing_annotations(experiment, subject, raw.copy())

#     # get events
#     events, event_dict = mne.events_from_annotations(raw)

#     #Shift the stimulus event codes forward in time to account for the LCD monitor delay
#     #(26 ms on our monitor, as measured with a photosensor)
#     #raw.annotations.onset = raw.annotations.onset+.026

#     #Downsample from the recorded sampling rate of 1024 Hz to 256 Hz to speed data processing
#     raw, events = raw.resample(256, events = events)
#     raw.events = events

#     # get and save the event counts
#     event_counts = {}
#     for key in event_dict.keys():
#         event_counts[key] = len(events[events[:, 2] == event_dict[key]])
#         print(key, len(events[events[:, 2] == event_dict[key]]))

#     # for some experiments, assure that the number of trials per condition is equal
#     if experiment in ['N170', 'N2pc', 'N400']:
#         assert len(set(event_counts.values())) == 1, "Not all conditions have same number of trials."
#     raw.event_counts = event_counts

#     # recalculate EOG channels
#     #raw = recalculate_eog_signal(raw.copy())

#     # set montage
#     raw = set_montage(raw.copy(), experiment="paperclip")

#     # save raw data
#     raw.save(os.path.join(raw_folder, f"{subject}-raw.fif"), overwrite=True)

# experiment = "paperclip"
# group = "infants"



""" ERPCORE """

for experiment in experiments:
    print(f"Processing experiment {experiment}...")
    
    for subject in subjects:
        print(f"Processing subject {subject}...")

        # file paths
        download_folder = os.path.join(base_dir, "data", "erpcore", experiment, subject, "eeg")
        raw_folder = os.path.join(base_dir, "data", "raw", experiment)
        if not os.path.exists(raw_folder):
            os.makedirs(raw_folder)
        
        # read raw data
        raw_file = glob(os.path.join(download_folder, "*.set"))
        assert len(raw_file) == 1, "More than one raw file found!"
        raw_file = raw_file[0]
        raw = mne.io.read_raw_eeglab(raw_file, eog=(), preload=True, 
                                    uint16_codec='utf-8',
                                    #montage_units='auto',
                                    verbose=None)

        # discard unnecessary triggers
        raw = discard_triggers(raw.copy(), delete_triggers[experiment])
        
        # collate triggers into conditions in annotations
        raw = rename_annotations(raw.copy(), conditions_triggers[experiment])
        raw = exception_pre_preprocessing_annotations(experiment, subject, raw.copy())
        
        # get events
        events, event_dict = mne.events_from_annotations(raw)

        #Shift the stimulus event codes forward in time to account for the LCD monitor delay
        #(26 ms on our monitor, as measured with a photosensor)
        raw.annotations.onset = raw.annotations.onset+.026

        #Downsample from the recorded sampling rate of 1024 Hz to 256 Hz to speed data processing
        raw, events = raw.resample(256, events = events)
        raw.events = events

        # get and save the event counts
        event_counts = {}
        for key in event_dict.keys():
            event_counts[key] = len(events[events[:, 2] == event_dict[key]])
            print(key, len(events[events[:, 2] == event_dict[key]]))

        # for some experiments, assure that the number of trials per condition is equal
        if experiment in ['N170', 'N2pc', 'N400']:
            assert len(set(event_counts.values())) == 1, "Not all conditions have same number of trials."
        raw.event_counts = event_counts
        
        # recalculate EOG channels
        raw = recalculate_eog_signal(raw.copy())

        # set montage
        raw = set_montage(raw.copy())

        # save raw data
        raw.save(os.path.join(raw_folder, f"{subject}-raw.fif"), overwrite=True)
