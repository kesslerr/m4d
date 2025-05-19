import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
import numpy as np
import itertools
import pandas as pd

try:
    from braindecode.preprocessing import exponential_moving_standardize
    from braindecode import EEGClassifier
    from braindecode.util import set_random_seeds

except:
    from braindecode.preprocessing import exponential_moving_standardize # workaround, because the second import usually works
    from braindecode import EEGClassifier
    from braindecode.util import set_random_seeds
    
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.utils import compute_class_weight
import torch
torch.set_num_threads(1) # check if this helps with overload

from joblib import Parallel, delayed, dump

# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#ptmp_dir = "/ptmp/kroma/m4d/"
os.chdir(base_dir)
sys.path.append(base_dir)

from src.utils import get_forking_paths, recode_conditions
#from src.config import multiverse_params, epoch_windows, baseline_windows

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


interim_folder = os.path.join(base_dir, "data", "interim", "leakage", experiment, subject)
processed_folder = f"/ptmp/kroma/m4d/data/processed/leakage/{experiment}/{subject}" #os.path.join(base_dir, "data", "processed", experiment, subject)
model_folder = os.path.join("/ptmp/kroma/m4d/", "models", "leakage", "eegnet", experiment, subject)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
else:
    _ = [os.remove(file) for file in glob(os.path.join(model_folder, "*"))]



def run_eegnet(X_train, y_train, X_test, y_test):
    
    for i in range(X_train.shape[0]):
        X_train[i,:,:] = exponential_moving_standardize(X_train[i,:,:], factor_new=0.001, init_block_size=None, eps=1e-4)
    for i in range(X_test.shape[0]):
        X_test[i,:,:] = exponential_moving_standardize(X_test[i,:,:], factor_new=0.001, init_block_size=None, eps=1e-4)
        
    # set random seed    
    set_random_seeds(108, cuda=False)
    
    # class weight for imbalanced learning
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    
    # get some information about the data
    sfreq = 256
    
    # train model
    net = EEGClassifier(
        "EEGNetv4", 
        criterion__weight=torch.Tensor(class_weights).to('cpu'), # class weight
        module__final_conv_length='auto',
        train_split=None, 
        max_epochs=200, 
        batch_size=16, 
        module__sfreq=sfreq, 
        #scoring="balanced_accuracy",
    )

    # Fit on training data
    net.fit(X_train, y_train, )

    # Evaluate on test data
    validation_acc = net.score(X_test, y_test)  # Scoring function uses balanced accuracy by default

    return validation_acc

# TODO: parallelize, because every function run oses only one core

def run_chunks(step, leakage, df): # run chunks individually
    files = sorted(glob(os.path.join(processed_folder, f"{leakage}_{step}_*-epo.fif")))
    for i, file in enumerate(files):
        # load epochs
        epochs_train = mne.concatenate_epochs([mne.read_epochs(j) for j in files if file != j])    
        epochs_test = mne.read_epochs(file)

        # get data and labels
        X_train = epochs_train.get_data()
        y_train = epochs_train.events[:,-1] - 1 # subtract 1 to get 0 and 1 instead of 1 and 2
        X_test = epochs_test.get_data()
        y_test = epochs_test.events[:,-1] - 1 # subtract 1 to get 0 and 1 instead of 1 and 2
            
        # run eegnet
        validation_acc = run_eegnet(X_train, y_train, X_test, y_test)
        
        # save performance 
        df = pd.concat([df, 
                        pd.DataFrame([{"subject": subject, "experiment": experiment, "step": step, 
                                       "leakage": leakage, "split": i, "accuracy": validation_acc,
                                       "n_train": len(y_train), "n_test": len(y_test)}])], 
                    ignore_index=True)
    return df
    
    
    
def run_versions(step, leakage, df): # run versions with different datasets used
    train_files = sorted(glob(os.path.join(processed_folder, f"{leakage}_{step}_train_*-epo.fif")))
    test_files = sorted(glob(os.path.join(processed_folder, f"{leakage}_{step}_test_*-epo.fif")))
    for i, (train_file, test_file) in enumerate(zip(train_files, test_files)):
        # load epochs
        epochs_train = mne.read_epochs(train_file)
        epochs_test = mne.read_epochs(test_file)

        # get data and labels
        X_train = epochs_train.get_data()
        y_train = epochs_train.events[:,-1] - 1 # subtract 1 to get 0 and 1 instead of 1 and 2
        X_test = epochs_test.get_data()
        y_test = epochs_test.events[:,-1] - 1 # subtract 1 to get 0 and 1 instead of 1 and 2
            
        # run eegnet
        validation_acc = run_eegnet(X_train, y_train, X_test, y_test)
        
        # save performance 
        df = pd.concat([df, 
                        pd.DataFrame([{"subject": subject, "experiment": experiment, "step": step, 
                                       "leakage": leakage, "split": i, "accuracy": validation_acc,
                                       "n_train": len(y_train), "n_test": len(y_test)}])], 
                    ignore_index=True)
    return df


df = pd.DataFrame(columns=["subject", "experiment", "step", "leakage", "split", "accuracy", "n_train", "n_test"])


""" hpf """

# leaky
df = run_chunks("hpf", "leaky", df)
# sealed
df = run_chunks("hpf", "sealed", df)

""" ica """

# leaky
df = run_chunks("ica", "leaky", df)
# sealed
df = run_versions("ica", "sealed", df)

""" ar """

# leaky
df = run_chunks("ar", "leaky", df)
# sealed
df = run_versions("ar", "sealed", df)



print(df)

df.to_csv(os.path.join(model_folder, f"eegnet_leakage.csv"), index=False)