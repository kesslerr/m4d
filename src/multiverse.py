import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
import numpy as np
import itertools
import pandas as pd
import pickle

# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#ptmp_dir = "/ptmp/kroma/m4d/"
os.chdir(base_dir)
sys.path.append(base_dir)

from src.utils import CharacteristicsManager, ica_eog_emg, autorej, summarize_artifact_interpolation, prestim_baseline_correction_ERN
from src.config import multiverse_params, epoch_windows, baseline_windows, translation_table

""" HEADER END """

# DEBUG
#experiment = "MMN"
#subject = "sub-008"
#stragglers = 10 # number of stragglers to be processed in the end


# define subject and session by arguments to this script
if len(sys.argv) != 3:
    print("Usage: python script.py experiment subject")
    sys.exit(1)
else:
    experiment = sys.argv[1]
    subject = sys.argv[2]
    print(f'Processing Experiment {experiment} Subject {subject}!')
    stragglers=None

""" SPECIFICATIONS END """

if experiment == "MIPDB":
    raw_folder = os.path.join(base_dir, "data", "raw", experiment, subject)
else:
    raw_folder = os.path.join(base_dir, "data", "raw", experiment)

interim_folder = os.path.join(base_dir, "data", "interim", experiment, subject)
if not os.path.exists(interim_folder):
    os.makedirs(interim_folder)
processed_folder = f"/ptmp/kroma/m4d/data/processed/{experiment}/{subject}" #os.path.join(base_dir, "data", "processed", experiment, subject)
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)
# delete all files in processed_folder and interim_folder
if not stragglers:
    print("Deleting all files in processed and interim folder.")
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
# get events again, because it doesn't survive i/o operations
events, event_dict = mne.events_from_annotations(raw)     

manager.update_characteristic('event_counts', event_counts)


# run multiverse
path_id = 1
total_iterations = len(list(itertools.product(*multiverse_params.values())))
print(f'Number of parameter combinations: {total_iterations}')

# new: apply Cz reference (preliminary) for all datasets, re-reference later in some pipelines
raw = raw.set_eeg_reference(['Cz'], projection=False)    

with tqdm(total=total_iterations) as pbar:
    
    for emc in multiverse_params['emc']:
        # emc
        param_str = f'{emc}'.translate(translation_table)
        if emc == 'ica':
            _raw0, n1, n2 = ica_eog_emg(raw.copy(), method='eog')
            manager.update_subsubfield('ICA EOG', param_str, 'n_components', n1)
            manager.update_subsubfield('ICA EOG', param_str, 'var_expl_ratio', n2)
                
        elif emc is None:
            _raw0 = raw.copy()

        # drop non-eeg channels (eog)
        _raw0.pick_types(eeg=True)
        
        for mus in multiverse_params['mus']:
            # mus
            param_str = f'{emc}_{mus}'.translate(translation_table)
            if mus == 'ica':
                _raw1, n1, n2 = ica_eog_emg(_raw0.copy(), method='emg')
                manager.update_subsubfield('ICA EMG', param_str, 'n_components', n1)
                manager.update_subsubfield('ICA EMG', param_str, 'var_expl_ratio', n2)
                
            elif mus is None:
                _raw1 = _raw0.copy()

            for lpf in multiverse_params['lpf']:
                for hpf in multiverse_params['hpf']:
                    # hpf + lpf
                    if hpf is None and lpf is None:
                        _raw2 = _raw1.copy()
                    else:
                        # Attention: l_freq is HPF cutoff, and h_freq is LPF cutoff
                        _raw2 = _raw1.copy().filter(l_freq=hpf, h_freq=lpf, method='fir', fir_design='firwin', skip_by_annotation='EDGE boundary', n_jobs=-1)

                    
                    for ref in multiverse_params['ref']:
                        # ref
                        param_str = f'{emc}_{mus}_{lpf}_{hpf}_{ref}'.translate(translation_table)
                        if experiment == "MIPDB": # for MIPDB; the naming doesnt follow 10/10 convention, therefore make some exceptions:
                            if ref == ['Cz']:
                                _raw3 = _raw2.copy()  # Cz already is the online reference
                            elif ref == "average":
                                _raw3 = _raw2.copy().set_eeg_reference(ref, projection=False)
                            elif ref == ['P9', 'P10']:
                                _raw3 = _raw2.copy().set_eeg_reference(['E58', 'E96'], projection=False) # projection must be false so that it is really re-referenced when using "average", and not only a projecten channel created     
                            else:
                                raise ValueError(f"Reference {ref} not implemented for MIPDB dataset.")       
                        else:
                            _raw3 = _raw2.copy().set_eeg_reference(ref, projection=False) # projection must be false so that it is really re-referenced when using "average", and not only a projecten channel created
    

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
                                
                                
                                if (experiment != "ERN") or (experiment == "ERN" and not base):
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
                                    # delete stimulus epochs in ERN
                                    if experiment == "ERN":
                                        epochs = epochs[["correct", "incorrect"]]
                                    
                                elif experiment == "ERN":
                                    # manual baseline correction, drawn from pre-stimulus interval (not pre-response)
                                    epochs = prestim_baseline_correction_ERN(_raw3.copy(), events, event_dict, detrend=detrend, baseline=int(base[0:3]))
                                    
                                                                
                                for ar in multiverse_params['ar']:
                                    # ar

                                    # string that describes the current parameter combination
                                    param_str = f'{emc}_{mus}_{lpf}_{hpf}_{ref}_{det}_{base}_{ar}'.translate(translation_table)

                                    # add metadata to epochs
                                    epochs.metadata = pd.DataFrame(
                                                data=[[path_id, emc, mus, lpf, hpf, ref, det, base, ar]] * len(epochs), 
                                                columns=['path_id', 'emc', 'mus', 'lpf', 'hpf', 'ref', 'det', 'base', 'ar'], 
                                                index=range(len(epochs)),
                                                )

                                    # add identifier / index for each trial to trace which trials will be separated to which subsets later (historically to trace them through oversampling)
                                    epochs.metadata["trial_id"] = list(range(len(epochs)))

                                    if ar:
                                        # Autoreject
                                        
                                        # if MIPDB , E129 doesnt have channel location, therefore drop it before autoreject (it would throw error)
                                        if experiment == "MIPDB":
                                            epochs = epochs.copy().drop_channels(['E129'])
                                        
                                        # estimate autoreject model on all epochs (not only training epochs), 
                                        epochs_ar, n1 = autorej(epochs.copy(), version=ar)

                                        # adjusted to new AR methods
                                        interp_frac_channels, interp_frac_trials, total_interp_frac, rej_frac_channels, rej_frac_trials, total_rej_frac = summarize_artifact_interpolation(n1, version = ar)
                                        manager.update_subsubfield('autoreject', param_str, 'total_interp_frac', total_interp_frac)
                                        manager.update_subsubfield('autoreject', param_str, 'interp_frac_channels', interp_frac_channels)
                                        manager.update_subsubfield('autoreject', param_str, 'interp_frac_trials', interp_frac_trials)
                                        if ar == "intrej":
                                            manager.update_subsubfield('autoreject', param_str, 'rej_frac_channels', rej_frac_channels)
                                            manager.update_subsubfield('autoreject', param_str, 'rej_frac_trials', rej_frac_trials)
                                            manager.update_subsubfield('autoreject', param_str, 'total_rej_frac', total_rej_frac)
                                            
                                                                                
                                    else:
                                        epochs_ar = epochs.copy()

                                    # save epochs
                                    epochs_ar.save(f"{processed_folder}/{param_str}-epo.fif", overwrite=True)

                                    # update iteration
                                    path_id += 1
                                    pbar.update(1)
                         
