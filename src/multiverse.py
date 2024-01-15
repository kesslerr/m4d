import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
import numpy as np
import itertools
import pandas as pd


# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#ptmp_dir = "/ptmp/kroma/m4d/"
os.chdir(base_dir)
sys.path.append(base_dir)

from src.utils import CharacteristicsManager, ica_eog_emg, autorej, summarize_artifact_interpolation
from src.config import multiverse_params, epoch_windows, baseline_windows

""" HEADER END """


# define subject and session by arguments to this script
if len(sys.argv) != 3:
    print("Usage: python script.py experiment subject")
    sys.exit(1)
else:
    experiment = sys.argv[1]
    subject = sys.argv[2]
    print(f'Processing Experiment {experiment} Subject {subject}!')

# DEBUG
# experiment = "N170"
# subject = "sub-001"

""" SPECIFICATIONS END"""

raw_folder = os.path.join(base_dir, "data", "raw", experiment)
interim_folder = os.path.join(base_dir, "data", "interim", experiment, subject)
if not os.path.exists(interim_folder):
    os.makedirs(interim_folder)
processed_folder = f"/ptmp/kroma/m4d/data/processed/{experiment}/{subject}" #os.path.join(base_dir, "data", "processed", experiment, subject)
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)
# delete all files in processed_folder and interim_folder
_ = [os.remove(file) for file in glob(os.path.join(interim_folder, "*"))]
_ = [os.remove(file) for file in glob(os.path.join(processed_folder, "*"))]

# read raw data
raw = mne.io.read_raw_fif(os.path.join(raw_folder, f"{subject}-raw.fif"), preload=True, verbose=None)

manager = CharacteristicsManager(f"{interim_folder}/characteristics.json", force_new=True)

# calculate again event_counts, because it doesnt surive i/o operations
event_counts = {}
for key in sorted(set(raw.annotations.description)):
    event_counts[key] = np.count_nonzero(raw.annotations.description == key)
    print(key, event_counts[key])
manager.update_characteristic('event_counts', event_counts)

# get events again, because it doesn't survive i/o operations
events, event_dict = mne.events_from_annotations(raw)

# run multiverse
path_id = 1
total_iterations = len(list(itertools.product(*multiverse_params.values())))
print(f'Number of parameter combinations: {total_iterations}')


with tqdm(total=total_iterations) as pbar:
    

    for ref in multiverse_params['ref']:
        # ref
        param_str = f'{ref}'
        _raw0 = raw.copy().set_eeg_reference(ref, projection=False) # projection must be false so that it is really re-referenced when using "average", and not only a projecten channel created
    
        for hpf in multiverse_params['hpf']:
            for lpf in multiverse_params['lpf']:
                # hpf + lpf
                if hpf is None and lpf is None:
                    _raw1 = _raw0.copy()
                else:
                    # CAVE: l_freq is HPF cutoff, and h_freq is LPF cutoff
                    _raw1 = _raw0.copy().filter(l_freq=hpf, h_freq=lpf, method='fir', fir_design='firwin', skip_by_annotation='EDGE boundary', n_jobs=-1)

                for emc in multiverse_params['emc']:
                    # emc
                    param_str = f'{ref}_{hpf}_{lpf}_{emc}'
                    if emc == 'ica':
                        _raw2, n1 = ica_eog_emg(_raw1.copy(), method='eog')
                        manager.update_subsubfield('ICA EOG', param_str, 'n_components', n1)
    
                    elif emc is None:
                        _raw2 = _raw1.copy()

                    # drop non-eeg channels (eog)
                    _raw2.pick_types(eeg=True)
                    
                    for mus in multiverse_params['mus']:
                        # mus
                        param_str = f'{ref}_{hpf}_{lpf}_{emc}_{mus}'
                        if mus == 'ica':
                            _raw3, n1 = ica_eog_emg(_raw2.copy(), method='emg')
                            manager.update_subsubfield('ICA EMG', param_str, 'n_components', n1)
                            
                        elif mus is None:
                            _raw3 = _raw2.copy()

                        for base in multiverse_params['base']:
                            # base (baseline correction and epoching)
                            if base == None:
                                baseline=None
                            else:
                                baseline = baseline_windows[base][experiment]

                            for det in multiverse_params['det']:
                                # detrend
                                if det == 'linear':
                                    detrend = 1
                                elif det == 'offset':
                                    detrend = 0
                                else:
                                    detrend = None
                                
                                # epoching
                                epochs = mne.Epochs(_raw3.copy(), 
                                                    events=events, 
                                                    event_id=event_dict,
                                                    tmin=epoch_windows[experiment][0], 
                                                    tmax=epoch_windows[experiment][1],
                                                    baseline=baseline,
                                                    detrend=detrend,
                                                    proj=False,
                                                    reject_by_annotation=False, 
                                                    preload=True)
                                
                                # TODO: is the detrending performed as expected?
                                
                                for ar in multiverse_params['ar']:
                                    # ar

                                    # string that describes the current parameter combination
                                    param_str = f'{ref}_{hpf}_{lpf}_{emc}_{mus}_{base}_{det}_{ar}'

                                    # add metadata to epochs
                                    epochs.metadata = pd.DataFrame(
                                                data=[[path_id, ref, hpf, lpf, emc, mus, base, det, ar]] * len(epochs), 
                                                columns=['path_id', 'ref', 'hpf', 'lpf', 'emc', 'mus', 'base', 'det', 'ar'], 
                                                index=range(len(epochs)),
                                                )

                                    # add identifier / index for each trial to trace which trials will be separated to which subsets later (historically to trace them through oversampling)
                                    epochs.metadata["trial_id"] = list(range(len(epochs)))

                                    if ar:
                                        # Autoreject
                                        
                                        # estimate autoreject model on all epochs (not only training epochs), 
                                        # TODO: mention, that this is potential data leakage, but the only feasible way to do it
                                        epochs_ar, n1 = autorej(epochs.copy())
                                        interp_frac_channels, interp_frac_trials, total_interp_frac = summarize_artifact_interpolation(n1)
                                        manager.update_subsubfield('autoreject', param_str, 'total_interp_frac', total_interp_frac)
                                        manager.update_subsubfield('autoreject', param_str, 'interp_frac_channels', interp_frac_channels)
                                        manager.update_subsubfield('autoreject', param_str, 'interp_frac_trials', interp_frac_trials)
                                                                                
                                    else:
                                        epochs_ar = epochs.copy()

                                    # save epochs
                                    epochs_ar.save(f"{processed_folder}/{param_str}-epo.fif", overwrite=True)

                                    # update iteration
                                    path_id += 1
                                    pbar.update(1)
                                    