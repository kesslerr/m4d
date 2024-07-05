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
experiment = "RSVP"
#subject = "sub-015"

# TODO: MAYBE? decoding should only be done after the baseline period ended!

# # define subject and session by arguments to this script
# if len(sys.argv) != 3:
#     print("Usage: python script.py experiment subject")
#     sys.exit(1)
# else:
#     experiment = sys.argv[1]
#     subject = sys.argv[2]
#     print(f'Processing Experiment {experiment} Subject {subject}!')

subjects = ['sub-001', 'sub-002', 'sub-005', 'sub-007', 'sub-008', 'sub-009', 'sub-011', 'sub-013', 'sub-014', 'sub-015', 'sub-017', 'sub-018', 'sub-020', 'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025', 'sub-026', 'sub-027', 'sub-028', 'sub-030', 'sub-035', 'sub-036', 'sub-037', 'sub-039', 'sub-040', 'sub-042', 'sub-043', 'sub-044', 'sub-045', 'sub-046', 'sub-048', 'sub-049']

    

def slider_parallel_subject(subject):  
    
    raw_folder = os.path.join(base_dir, "data", "raw", experiment)
    interim_folder = os.path.join(base_dir, "data", "interim", experiment, subject)
    processed_folder = f"/ptmp/kroma/m4d/data/processed/{experiment}/{subject}" #os.path.join(base_dir, "data", "processed", experiment, subject)
    model_folder = os.path.join(base_dir, "models", "sliding", experiment, subject)
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

    assert len(forking_paths) == 1152, "Number of forking paths is not 1152"

    quek_forking_path = 'None_None_45_0.5_average_200ms_offset_False'
    full_forking_path = 'ica_ica_6_0.5_average_200ms_linear_True'
    forking_path = quek_forking_path
    file = [i for i in files if forking_path in i][0]

    
    # load epochs
    epochs = mne.read_epochs(file, preload=True, verbose=None)
    if experiment == "RSVP":
        # recode conditions
        epochs = recode_conditions(epochs.copy(), version="categories") 

    # TODO maybe: crop epochs as in EEGNet, but maybe here it is nice to see the basline period

    #n_tp = len(epochs.times)
    tmin = decoding_windows[experiment][0]
    tmax = decoding_windows[experiment][1]
    
    
    
    
    # extract data from epochs
    X = epochs.copy().crop(tmin=tmin, tmax=tmax).get_data()
    y = epochs.events[:,-1] - 1 # subtract 1 to get 0 and 1 instead of 1 and 2
    
    scores = slider(X.copy(),y.copy())
    
    #permut_scores = slider_permut(X.copy(),y.copy(), iter=10)
    
    # save scores to file
    np.save(os.path.join(model_folder, f"{forking_path.translate(translation_table)}.npy"), scores)
    # save permutation scores to file
    #np.save(os.path.join(model_folder, f"permutations_{forking_path.translate(translation_table)}.npy"), permut_scores)
    
    
    #return scores, permut_scores    
    





# We will train the classifier on all left visual vs auditory trials on MEG
def slider(X,y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", class_weight='balanced'))
    time_decod = SlidingEstimator(clf, n_jobs=1, scoring="accuracy", verbose=True) # "roc_auc" is only available for 2 class case, was balanced_accuracy  # new, n_jobs=1 to prevent overload
    # here we use cv=3 just for speed
    scores = cross_val_multiscore(time_decod, X, y, cv=10, n_jobs=1) # new, n_jobs=1 to prevent overload
    # Mean scores across cross-validation splits
    return np.mean(scores, axis=0)


def slider_permut(X,y, iter=100): # TODO increase iter
    """ permute labels, and then run slider """
    n_tp = X.shape[2]
    results = np.zeros((iter, n_tp))
    for i in range(iter):
        y_permut = np.random.permutation(y)
        results[i,:] = slider(X,y_permut)
    return results
# this takes too long for all fps.... maybe just do group stats vs. 0.5
# and then instead if HLM: LM: per forking path across subjects, separately for each experiment  


def slider_parallel(forking_path, file):  
    
    # extract the string "XXXms" from forking path
    #baseline_ms = re.search(r"_(\d{3}ms)_", forking_path).group(1)
    #baseline_end = baseline_windows[baseline_ms][experiment][-1]
    
    
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
    
    scores = slider(X.copy(),y.copy())
    
    #permut_scores = slider_permut(X.copy(),y.copy(), iter=10)
    
    # save scores to file
    np.save(os.path.join(model_folder, f"{forking_path.translate(translation_table)}.npy"), scores)
    # save permutation scores to file
    #np.save(os.path.join(model_folder, f"permutations_{forking_path.translate(translation_table)}.npy"), permut_scores)
    
    
    #return scores, permut_scores


#Parallel(n_jobs=-1)(delayed(slider_parallel)(forking_path, file) for forking_path, file in zip(forking_paths, files))
Parallel(n_jobs=-1)(delayed(slider_parallel_subject)(subject) for subject in subjects)


# DEBUG
# file = files[0]
# forking_path = forking_paths[0]



# TODO: the following is for one-participant / one-fp statistics... computationally very long. If want to use that kind of statistics,
# need to find one single value that corresponds to decoding performance

# epochs = mne.read_epochs(file, preload=True, verbose=None)

# # extract data from epochs
# X = epochs.get_data()
# y = epochs.events[:,-1] - 1 # subtract 1 to get 0 and 1 instead of 1 and 2

# scores = slider(X,y)

# permut_scores = slider_permut(X,y)

# avg_permut_scores = np.mean(permut_scores, axis=0)


# # for each time tpoint, test how many permut_scores are larger than score
# p_values = np.mean(permut_scores > scores, axis=0)
# # binary vector of p values < 0.05
# p_val_mask = (p_values < 0.05).astype(int)

# # Plot
# fig, ax = plt.subplots()
# ax.plot(epochs.times, scores, label="score")
# ax.plot(epochs.times, avg_permut_scores, label="H0")
# ax.axhline(0.5, color="k", linestyle="--", label="chance")
# ax.set_xlabel("Times")
# ax.set_ylabel("AUC")  # Area Under the Curve
# ax.legend()
# ax.axvline(0.0, color="k", linestyle="-")
# ax.set_title("Sensor space decoding")

# # # add the p_val_mask as shaded regions
# ax.fill_between(epochs.times, 0.5, 0.55, where=p_val_mask==1, color='green', alpha=0.5)

