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

# DEBUG
#experiment = "ERN"
#subject = "sub-001"

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
model_folder = os.path.join("/ptmp/kroma/m4d/", "models", "eegnet", experiment, subject)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
else:
    _ = [os.remove(file) for file in glob(os.path.join(model_folder, "*"))]


forking_paths, files, forking_paths_split = get_forking_paths(
                            base_dir="/ptmp/kroma/m4d/", 
                            experiment=experiment,
                            subject=subject, 
                            sample=None) 
    
""" SPECIFICATIONS END"""

rdm=False

# DEBUG
#forking_path = forking_paths[0]
#file = files[0]

def parallel_eegnet(forking_path, file):  
    
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
    
    if experiment == "RSVP":
        if rdm == False:
            # recode conditions
            epochs = recode_conditions(epochs.copy(), version="superordinate")
        elif rdm == True:
            epochs = recode_conditions(epochs.copy(), version="categories")
    
    # extract data from epochs
    X = epochs.get_data()
    y = epochs.events[:,-1] - 1 # subtract 1 to get 0 and 1 instead of 1 and 2
    
    # The exponential moving standardization function works on single trials, therefore:
    for i in range(X.shape[0]):
        X[i,:,:] = exponential_moving_standardize(X[i,:,:], factor_new=0.001, init_block_size=None, eps=1e-4)
    
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
        train_split=None, 
        max_epochs=200, 
        batch_size=16, 
        module__sfreq=sfreq, 
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
                    'accuracy': [validation_acc],
                    })
    dfi.to_csv(f"{model_folder}/{forking_path}.csv", index=False)


# parallel processing
Parallel(n_jobs=-1)(delayed(parallel_eegnet)(forking_path, file) for forking_path, file in zip(forking_paths, files))

