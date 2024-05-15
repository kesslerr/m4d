
import numpy as np 
import pandas as pd
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.join(os.path.dirname(os.getcwd()))
sys.path.append(base_dir)


experiment = "RSVP"
subjects = ['sub-001', 'sub-002', 'sub-005', 'sub-007', 'sub-008', 'sub-009', 'sub-011', 'sub-013', 'sub-014', 'sub-015', 'sub-017', 'sub-018', 'sub-020', 'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025', 'sub-026', 'sub-027', 'sub-028', 'sub-030', 'sub-035', 'sub-036', 'sub-037', 'sub-039', 'sub-040', 'sub-042', 'sub-043', 'sub-044', 'sub-045', 'sub-046', 'sub-048', 'sub-049']
categories = ['aquatic', 'bird', 'clothing', 'fruits', 'furniture',  
              'human', 'insect', 'mammal', 'plants', 'tools']

# DEBUG
#subject=subjects[0]

###### EEGNET

# load matrices
results = np.empty((10,10,len(subjects)))
for s, subject in enumerate(subjects):
    # import numpy matrix
    mat2d = np.load(os.path.join(base_dir, 'models', 'eegnet', 'RSVP', subject, '10x10.csv.npy'))
    results[:,:,s] = mat2d
    


# heatmap  mean across participants
avgResults = np.mean(results, axis=2)
plt.figure(figsize=(10,10))
divcmap = sns.diverging_palette(300, 145, s=60, as_cmap=True, sep=100)
sns.heatmap(avgResults, 
            cmap=divcmap, #'PiYG', 
            center=0.5, annot=True, fmt=".3f",
            cbar_kws={'label': 'Accuracy'},
            xticklabels=categories, yticklabels=categories,
            )
plt.title("Average accuracy across all participants")
plt.show()


# for each participant
plt.figure(figsize=(5*5,5*7))
for s, subject in enumerate(subjects):
    plt.subplot(7,5,s+1)
    sns.heatmap(results[:,:,s], 
                cmap=divcmap, #'PiYG', 
                center=0.5, annot=True, fmt=".2f",
                cbar=False,
                xticklabels=categories, yticklabels=categories)
    plt.title(subject)
    plt.axis('off')
#plt.suptitle("Accuracy per participant")
plt.tight_layout()
plt.show()



###### TIMERESOLVED

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



from src.utils import get_forking_paths, cluster_test, get_age
from src.config import translation_table, luck_forking_paths, subjects as subjects_erpcore, experiments as experiments_erpcore, decoding_windows, subjects_mipdb_dem, age_groups, groups_subjects_mipdb, luck_forking_paths







    _, files, forking_paths_split = get_forking_paths(
                                base_dir="/ptmp/kroma/m4d/", 
                                experiment=experiment,
                                subject=subject, 
                                sample=None)

    assert len(files) == 1152, "Number of forking paths is not 1152"

    quek_forking_path = 'None_None_45_0.5_average_200ms_offset_False'
    full_forking_path = 'ica_ica_6_0.5_average_200ms_linear_True'
    forking_path = full_forking_path
    file = [i for i in files if forking_path in i][0]
    
    
