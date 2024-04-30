
import numpy as np
import pandas as pd
import mne
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.append(base_dir)
from src.utils import discard_triggers, recalculate_eog_signal, recalculate_eog_signal_128egi, set_montage, rename_annotations
from src.config import category_triggers, supraordinate_triggers, delete_triggers

participant_file = base_dir + "/data/ds005106-download/participants.tsv"
download_data_dir = base_dir + "/data/ds005106-download/"
raw_data_dir = base_dir + "/data/raw/RSVP/"

part_df = pd.read_csv(participant_file, sep="\t")

included_subs = part_df[part_df.exclude==0]["#participant_id"]

print(f"Number of included subjects: {len(included_subs)}")

#subject = included_subs[0]
for subject in included_subs:

    # load eeg data
    eeg_file = download_data_dir + f"{subject}/eeg/{subject}_task-fix_eeg.set"
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

    # delete -1 from triggers
    raw = discard_triggers(raw.copy(), delete_triggers["RSVP"])

    # collate triggers into conditions in annotations, TODO: think about supercategories, if decoding within this one is not possible
    raw = rename_annotations(raw.copy(), category_triggers)



    # get events
    events, event_dict = mne.events_from_annotations(raw)

    #Shift the stimulus event codes forward in time to account for the LCD monitor delay
    #(26 ms on our monitor, as measured with a photosensor)
    #raw.annotations.onset = raw.annotations.onset+.026

    #Downsample from the recorded sampling rate of 1024 Hz to 256 Hz to speed data processing
    raw, events = raw.resample(250, events = events)
    raw.events = events

    # get and save the event counts
    event_counts = {}
    for key in event_dict.keys():
        event_counts[key] = len(events[events[:, 2] == event_dict[key]])
        print(key, len(events[events[:, 2] == event_dict[key]]))
    raw.event_counts = event_counts

    # recalculate EOG channels
    raw = recalculate_eog_signal(raw.copy(), sfreq=250, has_EOG=False)

    # set montage
    raw = set_montage(raw.copy(), experiment="RSVP")

    # save raw data
    raw.save(os.path.join(raw_data_dir, f"{subject}-raw.fif"), overwrite=True)

