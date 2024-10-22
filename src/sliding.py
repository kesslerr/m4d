import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, cross_validate, StratifiedKFold
from sklearn.utils import compute_class_weight
from joblib import Parallel, delayed, dump

# import svm
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)
from mne.stats import permutation_cluster_1samp_test

# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#ptmp_dir = "/ptmp/kroma/m4d/"
os.chdir(base_dir)
sys.path.append(base_dir)

from src.utils import get_forking_paths, recode_conditions
from src.config import translation_table, baseline_windows, decoding_windows

""" HEADER END """

# DEBUG
#experiment = "MMN"
#subject = "sub-015"


# define subject and session by arguments to this script
if len(sys.argv) != 3:
    print("Usage: python script.py experiment subject")
    sys.exit(1)
else:
    experiment = sys.argv[1]
    subject = sys.argv[2]
    print(f'Processing Experiment {experiment} Subject {subject}!')


raw_folder = os.path.join(base_dir, "data", "raw", experiment)
interim_folder = os.path.join(base_dir, "data", "interim", experiment, subject)
processed_folder = f"/ptmp/kroma/m4d/data/processed/{experiment}/{subject}" #os.path.join(base_dir, "data", "processed", experiment, subject)
model_folder = os.path.join("/ptmp/kroma/m4d/", "models", "sliding", experiment, subject)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# delete previous files if existent
for file in glob(os.path.join(model_folder, "*.npy")):
    os.remove(file)

forking_paths, files, forking_paths_split = get_forking_paths(
                            base_dir="/ptmp/kroma/m4d/", 
                            experiment=experiment,
                            subject=subject, 
                            sample=None)

# We will train the classifier on all left visual vs auditory trials on MEG
def slider(X,y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", class_weight='balanced'))
    time_decod = SlidingEstimator(clf, n_jobs=1, scoring="balanced_accuracy", verbose=True) # "roc_auc" balanced_accuracy  # new, n_jobs=1 to prevent overload
    # here we use cv=3 just for speed
    scores = cross_val_multiscore(time_decod, X, y, cv=10, n_jobs=1) # new, n_jobs=1 to prevent overload
    # Mean scores across cross-validation splits
    return np.mean(scores, axis=0)


def slider_permut(X,y, iter=1000): 
    """ permute labels, and then run slider """
    n_tp = X.shape[2]
    results = np.zeros((iter, n_tp))
    for i in range(iter):
        y_permut = np.random.permutation(y)
        results[i,:] = slider(X,y_permut)
    return results
# this takes too long for all fps.... therefore just do group stats vs. 0.5
# and then instead if HLM: LM: per forking path across subjects, separately for each experiment  


def slider_parallel(forking_path, file):  
    
    # load epochs
    epochs = mne.read_epochs(file, preload=True, verbose=None)
    if experiment == "RSVP":
        # recode conditions
        epochs = recode_conditions(epochs.copy(), version="superordinate")

    #n_tp = len(epochs.times)
    tmin = decoding_windows[experiment][0]
    tmax = decoding_windows[experiment][1]
    
    # extract data from epochs
    X = epochs.copy().crop(tmin=tmin, tmax=tmax).get_data()
    y = epochs.events[:,-1] - 1 # subtract 1 to get 0 and 1 instead of 1 and 2
    
    try:
        scores = slider(X.copy(),y.copy())
        np.save(os.path.join(model_folder, f"{forking_path.translate(translation_table)}.npy"), scores)
        
    except ValueError as e: # a forking path could have no trials of one condition (heavy AR might be the reason)
        print(f"Error in {forking_path}")
        print(e)
        return


Parallel(n_jobs=-1)(delayed(slider_parallel)(forking_path, file) for forking_path, file in zip(forking_paths, files))

# DEBUG
# file = files[0]
# forking_path = forking_paths[0]



