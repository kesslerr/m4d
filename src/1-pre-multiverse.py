""" Multiverse preprocessing """

import itertools
from matplotlib import pyplot as plt
import seaborn as sns
import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
import json
import re
import numpy as np
import itertools
import pandas as pd
if sys.platform.startswith('darwin'):
    mne.viz.set_browser_backend('qt', verbose=None) # 'qt' or 'matplotlib'

# go to base directory and import globals
#os.chdir(os.path.dirname(os.getcwd())) # open base_dir to be able to import from src
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.append(base_dir)

# DEBUG
sub = "sub-001"
experiment = "N170"

# # define subject and session by arguments to this script
# if len(sys.argv) != 2:
#     print("Usage: python script.py sub")
#     sys.exit(1)
# else:
#     sub = sys.argv[1]
#     experiment = sys.argv[2]
#     print(f'Processing Subject {sub} Experiment {experiment}!')

# file paths
raw_folder = os.path.join(base_dir, "data", "erpcore", experiment, sub, "eeg")
trigger_file = os.path.join(base_dir, "data", "erpcore", experiment, f"task-{experiment}_events.json")

# read raw data
raw_file = glob(os.path.join(raw_folder, "*.set"))
assert len(raw_file) == 1, "More than one raw file found!"
raw_file = raw_file[0]

raw = mne.io.read_raw_eeglab(raw_file, eog=(), preload=True, 
                             uint16_codec='utf-8', #latin1', #'utf-8', #None, 
                             #montage_units='auto',
                             verbose=None)


""" annotations and events """

# delete particular triggers from the data
raw.annotations.delete([i for i, x in enumerate(raw.annotations.description) if x in ['201', '202']]) # correct and error

# load trigger info from json file
triggers = json.load(open(trigger_file))['value']['Levels']

# reorganize the trigger dict
new_triggers = {}
for key, value in triggers.items():
    if '-' in key:
        start, end = map(int, key.split('-')) 
        for i in range(start, end + 1):
            new_triggers[str(i)] = value
    else:
        new_triggers[str(key)] = value

print(new_triggers)

# Rename trigger values
rename_mapping = {
    'Stimulus - faces': 'face',
    'Stimulus - cars': 'car',
    'Stimulus - scrambled faces': 'scrambled_face',
    'Stimulus - scrambled cars': 'scrambled_car',
    'Response - correct': 'correct',
    'Response - error': 'error'
}

for key, value in new_triggers.items():
    if value in rename_mapping:
        new_triggers[key] = rename_mapping[value]

print(new_triggers)

# # remove the responses from the triggers
# values_to_remove = ['correct', 'error']
# keys_to_remove = [key for key, value in new_triggers.items() if value in values_to_remove]
# for key in keys_to_remove:
#     del new_triggers[key]

# print(new_triggers)


raw.annotations.description = np.array(
    [
        new_triggers[i] if i in new_triggers else i
        for i in raw.annotations.description
    ]
)

# get events
events, event_dict = mne.events_from_annotations(raw) # , event_id=new_triggers

raw.events = events


#Shift the stimulus event codes forward in time to account for the LCD monitor delay
#(26 ms on our monitor, as measured with a photosensor)
raw.annotations.onset = raw.annotations.onset+.026


#Downsample from the recorded sampling rate of 1024 Hz to 256 Hz to speed data processing
raw, events = raw.resample(256, events = events)

# plot events
mne.viz.plot_events(events, event_id=event_dict)

event_counts = {}
for key in event_dict.keys():
    event_counts[key] = len(events[events[:, 2] == event_dict[key]])
    print(key, len(events[events[:, 2] == event_dict[key]]))

assert event_counts['face'] == event_counts['car'] == event_counts['scrambled_face'] == event_counts['scrambled_car'], "Number of trials per condition is not equal!"

""" rereferencing """

#Rereference to the average of P9 and P10
raw = raw.set_eeg_reference(['P9','P10']) # there is no mastoids in the data

""" eog channels """

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

# plot
#raw.plot(start = 33)

""" montage """

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
mont.plot()

# Apply the montage
raw.set_montage(mont)

# save raw data
raw.save(os.path.join(raw_folder, "raw.fif"), overwrite=True)
