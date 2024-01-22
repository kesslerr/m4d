""" Multiverse preprocessing """

from matplotlib import pyplot as plt
import seaborn as sns
import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
if sys.platform.startswith('darwin'):
    mne.viz.set_browser_backend('qt', verbose=None) # 'qt' or 'matplotlib'

# go to base directory and import globals
#os.chdir(os.path.dirname(os.getcwd())) # open base_dir to be able to import from src
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.append(base_dir)
from src.utils import discard_triggers, recalculate_eog_signal, set_montage, rename_annotations
from src.config import subjects, experiments, delete_triggers, conditions_triggers
from src.exceptions import exception_pre_preprocessing_annotations

""" HEADER END """

# DEBUG
# experiment = "MMN"
# subject = "sub-001"

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
