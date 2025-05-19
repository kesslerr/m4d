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

from src.utils import CharacteristicsManager, ica_eog_emg, autorej, autorej_seed, summarize_artifact_interpolation, prestim_baseline_correction_ERN
from src.config import multiverse_params, epoch_windows, baseline_windows, translation_table

""" HEADER END """
experiment = "N170"

# DEBUG
#subject = "sub-001"


# define subject and session by arguments to this script
if len(sys.argv) != 2:
    print("Usage: python script.py subject")
    sys.exit(1)
else:
    subject = sys.argv[1]
    print(f'Processing Experiment {experiment} Subject {subject}!')
    


""" SPECIFICATIONS END """

processed_folder = f"/ptmp/kroma/m4d/data/processed/{experiment}/{subject}" #os.path.join(base_dir, "data", "processed", experiment, subject)
ar_folder = f"/ptmp/kroma/m4d/data/processed/ar_seeds/{experiment}/{subject}"
if not os.path.exists(ar_folder):
    os.makedirs(ar_folder)


pipelines = ["ica_ica_45_0.5_average_linear_200ms_False",
             "None_None_45_0.5_average_linear_200ms_False",
             "None_None_45_0.1_average_linear_None_False"]

ar_versions = ["int", "intrej"] # autoreject versions to be used
sampling_seeds = [0, 1, 2, 3, 4] # random seeds for autoreject
ar_seeds = [0, 1, 2, 3, 4] # random seeds for autoreject


for pipeline in pipelines:
    # read epochs of a pipeline without AR conducted
    epochs = mne.read_epochs(f"{processed_folder}/{pipeline}-epo.fif", preload=True)

    for ar_version in ar_versions:
        updated_pipeline_name = pipeline.replace("_False", f"_{ar_version}")
        
        # keep sampling seed constant, but change autoreject seed
        for ar_seed in ar_seeds:
            epochs_ar, n1 = autorej_seed(epochs.copy(), version=ar_version, sampling_seed=1, ar_seed=ar_seed)
            epochs_ar.save(f"{ar_folder}/{updated_pipeline_name}_arseed_{ar_seed}-epo.fif", overwrite=True)

        # change sampling seed, keep ar seed constant
        for sampling_seed in sampling_seeds:
            epochs_ar, n1 = autorej_seed(epochs.copy(), version=ar_version, sampling_seed=sampling_seed, ar_seed=1)
            epochs_ar.save(f"{ar_folder}/{updated_pipeline_name}_samplingseed_{sampling_seed}-epo.fif", overwrite=True)

