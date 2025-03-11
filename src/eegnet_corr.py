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
    
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, cross_validate, StratifiedKFold
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
from src.config import subjects #multiverse_params, epoch_windows, baseline_windows

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
experiment = "N170_corr"

#subject = "sub-001"
# TODO: loop over subs or use slurm


raw_folder = os.path.join(base_dir, "data", "raw", experiment)
interim_folder = os.path.join(base_dir, "data", "interim", experiment, subject)
processed_folder = f"/u/kroma/m4d/data/processed/{experiment}/{subject}" #os.path.join(base_dir, "data", "processed", experiment, subject)
model_folder = os.path.join("/u/kroma/m4d/", "models", "eegnet_corr", experiment, subject)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
else:
    _ = [os.remove(file) for file in glob(os.path.join(model_folder, "*"))]


forking_paths, files, forking_paths_split = get_forking_paths(
                            base_dir="/u/kroma/m4d/", 
                            experiment=experiment,
                            subject=subject, 
                            sample=None) 

forking_path = forking_paths[0]
file = files[0]

conditions = ['faces', 'cars', 'faces_scr', 'cars_scr']

import itertools
combinations = list(itertools.combinations(conditions, 2))


""" SPECIFICATIONS END"""

rdm=False

# DEBUG
#forking_path = forking_paths[0]
#file = files[0]

def parallel_eegnet(combi1, combi2):  
    
    try: # this is a quick and dirty workaround for an error, related to a RAVEN update and braindecode incompatibilities
        from braindecode.preprocessing import exponential_moving_standardize # workaround, because the second import usually works
        from braindecode import EEGClassifier
        from braindecode.util import set_random_seeds
    except:
        from braindecode.preprocessing import exponential_moving_standardize
        from braindecode import EEGClassifier
        from braindecode.util import set_random_seeds
    
    # load epochs
    epochs = mne.read_epochs(file, preload=True, verbose=None)
    
    epochs1 = epochs[combi1]
    epochs2 = epochs[combi2]    
    
    
    # extract data from epochs
    X1 = epochs1.get_data()
    y1 = np.zeros(X1.shape[0])
    X2 = epochs2.get_data()
    y2 = np.ones(X2.shape[0])
    
    X = np.vstack([X1,X2])
    y = np.hstack([y1,y2])
     
    # The exponential moving standardization function seems to work on single trials, therefore:
    #X_train_ems = np.zeros(X_train.shape)
    for i in range(X.shape[0]):
        X[i,:,:] = exponential_moving_standardize(X[i,:,:], factor_new=0.001, init_block_size=None, eps=1e-4)
    # This is probably more important than conversion to mV, as this also brings data in the similar range. 
    
    # set random seed    
    set_random_seeds(108, cuda=False)
    
    # create stratified k folds (same percentage (nearly) of each class in each fold, relative to original ratio)
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None) # TODO: set seed?
    
    # class weight for imbalanced learning
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    
    # get some information about the data
    sfreq = 256
    
    # train model
    net = EEGClassifier(
        "EEGNetv4", 
        criterion__weight=torch.Tensor(class_weights).to('cpu'), # class weight
        module__final_conv_length='auto',
        train_split=None, #ValidSplit(0.2),
        max_epochs=200, # TODO maybe increase, I saw consistent increase from 30 to 100, so maybe we can even more increase
        batch_size=16, # this worked better than no batch size (which is then very large compared to the amount of data available)
        module__sfreq=sfreq, 
        #optimizer__lr=lr,
        #module__drop_prob=0.25,
    )

    cvs = cross_validate(net, 
                        X, 
                        y, 
                        scoring="balanced_accuracy", # for balanced classes, this corresponds to accuracy,
                        # chance level might be 0 (adjusted = False), or 0.X (adjusted = True)
                        cv=skfold, 
                        n_jobs=1, # this avoids overload
                        return_estimator=False, # if you need the model to estimate on another test set
                        return_train_score=False,
                        )
    
    # save performance 
    validation_acc = np.mean(cvs['test_score'])

    dfi = pd.DataFrame({'forking_path': [forking_path],
                    'combi1': [combi1],
                    'combi2': [combi2],
                    'subject': [subject],
                    'accuracy': [validation_acc],
                    })
    dfi.to_csv(f"{model_folder}/{combi1}_{combi2}.csv", index=False)


# parallel processing
Parallel(n_jobs=-1)(delayed(parallel_eegnet)(combi1, combi2) for combi1, combi2 in combinations)
