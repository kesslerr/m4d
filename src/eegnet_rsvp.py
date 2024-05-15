import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
import numpy as np
import itertools
import pandas as pd
import time

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
torch.set_num_threads(1) # TODO: check if this helps with overload

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
experiment = "RSVP"
#subject = "sub-001"
rdm=False # TODO: code if a whole rdm shall be created

# define subject and session by arguments to this script
#if len(sys.argv) != 3:
#    print("Usage: python script.py experiment subject")
#    sys.exit(1)
#else:
#    experiment = sys.argv[1]
#    subject = sys.argv[2]
#    print(f'Processing Experiment {experiment} Subject {subject}!')

#df = pd.DataFrame()

#for subject in ['sub-001', 'sub-002', 'sub-005', 'sub-007', 'sub-008', 'sub-009', 'sub-011', 'sub-013', 'sub-014', 'sub-015', 'sub-017', 'sub-018', 'sub-020', 'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025', 'sub-026', 'sub-027', 'sub-028', 'sub-030', 'sub-035', 'sub-036', 'sub-037', 'sub-039', 'sub-040', 'sub-042', 'sub-043', 'sub-044', 'sub-045', 'sub-046', 'sub-048', 'sub-049']:
subjects = ['sub-001', 'sub-002', 'sub-005', 'sub-007', 'sub-008', 'sub-009', 'sub-011', 'sub-013', 'sub-014', 'sub-015', 'sub-017', 'sub-018', 'sub-020', 'sub-021', 'sub-022', 'sub-023', 'sub-024', 'sub-025', 'sub-026', 'sub-027', 'sub-028', 'sub-030', 'sub-035', 'sub-036', 'sub-037', 'sub-039', 'sub-040', 'sub-042', 'sub-043', 'sub-044', 'sub-045', 'sub-046', 'sub-048', 'sub-049']

def parallel_subject(subject):    
    df = pd.DataFrame()

    try: # this is a quick and dirty workaround for an error, related to a RAVEN update and braindecode incompatibilities
        from braindecode.preprocessing import exponential_moving_standardize # workaround, because the second import usually works
        from braindecode import EEGClassifier
        from braindecode.util import set_random_seeds
    except:
        from braindecode.preprocessing import exponential_moving_standardize
        from braindecode import EEGClassifier
        from braindecode.util import set_random_seeds

    raw_folder = os.path.join(base_dir, "data", "raw", experiment)
    interim_folder = os.path.join(base_dir, "data", "interim", experiment, subject)
    processed_folder = f"/ptmp/kroma/m4d/data/processed/{experiment}/{subject}" #os.path.join(base_dir, "data", "processed", experiment, subject)
    model_folder = os.path.join(base_dir, "models", "eegnet", experiment, subject)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    forking_paths, files, forking_paths_split = get_forking_paths(
                                base_dir="/ptmp/kroma/m4d/", 
                                experiment=experiment,
                                subject=subject, 
                                sample=None) # DEBUG 5 TODO None
        
    """ SPECIFICATIONS END"""

    # DEBUG
    #forking_path = forking_paths[0]
    #file = files[0]
    quek_forking_path = 'None_None_45_0.5_average_200ms_offset_False'
    full_forking_path = 'ica_ica_6_0.5_average_200ms_linear_True'
    forking_path = full_forking_path
    file = [i for i in files if forking_path in i][0]


    epochs = mne.read_epochs(file, preload=True, verbose=None)
    epochs = recode_conditions(epochs.copy(), version="categories") 
    #epochs = recode_conditions(epochs.copy(), version="superordinate") #pseudo-superordinate

    # new, make pseudo-trials to speed up everything: use from each category 1 trial (animate: 5 cat, inanimate: 5 cat)

    # cropping must be done just before extracting data
    epochs.crop(0.0, 0.8) # TODO new: crop epochs at 0 to avoid the baseline which is NOT predictable but highly variable in this dataset

    #X = epochs.get_data()
    #y = epochs.events[:,-1] # TODO new: not -1 to be consistent later with subsetting
    #y = epochs.events[:,-1] - 1 # subtract 1 to get 0 and 1 instead of 1 and 2

    #for i in range(X.shape[0]):
    #    X[i,:,:] = exponential_moving_standardize(X[i,:,:], factor_new=0.001, init_block_size=None, eps=1e-4)
    set_random_seeds(108, cuda=False)
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=None) # TODO: set seed?

    # get some information about the data
    if "RSVP" in experiment:
        sfreq = 250 # TODO: maybe this was the MIPDB error?
    else:
        sfreq = 256

    start_time = time.time()

    test_accs = np.full((10, 10), np.nan)
    for class_i, i in epochs.event_id.items():
        for class_j, j in epochs.event_id.items():
            
            print(f"Processing classes {i} and {j}")
            if i >= j:
                continue
            # subsetting
            this_epochs = epochs[[class_i, class_j]]
            this_X = this_epochs.get_data()
            this_y = this_epochs.events[:,-1] #-1
            this_y = np.where(this_y == i, 0, 1)
            
            class_weights = compute_class_weight('balanced', classes=np.unique(this_y), y=this_y)

            # new standardization is not within the loop!!
            for u in range(this_X.shape[0]):
                this_X[u,:,:] = exponential_moving_standardize(this_X[u,:,:], factor_new=0.001, init_block_size=None, eps=1e-4)
    
            
            # train model
            net = EEGClassifier(
                "EEGNetv4", 
                criterion__weight=torch.Tensor(class_weights).to('cpu'), # class weight
                module__final_conv_length='auto',
                train_split=None, #ValidSplit(0.2),
                max_epochs=200, # TODO was 200 maybe increase, I saw consistent increase from 30 to 100, so maybe we can even more increase
                batch_size=16,# TODO, was 16 # this worked better than no batch size (which is then very large compared to the amount of data available)
                module__sfreq=sfreq, # TODO variable
                #device='cuda', # TODO THIS IS A TEST
                #optimizer__lr=lr,
                #module__drop_prob=0.25,
            )

            cvs = cross_validate(net, 
                                this_X, 
                                this_y, 
                                scoring="balanced_accuracy", # for balanced classes, this corresponds to accuracy,
                                # chance level might be 0 (adjusted = False), or 0.X (adjusted = True)
                                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=None), #TODO: skfold, 
                                n_jobs=1, # TODO: check if this avoids overload
                                return_estimator=False, # if you need the model to estimate on another test set
                                return_train_score=False,
                                )

            #cvs['test_score']
            print(f"test acc: {np.mean(cvs['test_score'])}")
            # save performance TODO    
            test_accs[i-1,j-1] = np.mean(cvs['test_score'])
    
    np.save(f"{model_folder}/10x10.csv", test_accs)

            #dfi = pd.DataFrame({'forking_path': [forking_path],
            #    'subject': [subject],
            #    'accuracy': [np.mean(cvs['test_score'])],
            #    })
            #df = pd.concat([df, dfi])
            #dfi.to_csv(f"{model_folder}/pseudo-superordinate_full_fp.csv", index=False)


    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")



# save np file
#np.save(f"test_accs2.npy", test_accs)
#pd.to_csv(f"valid_accs_pseudo_superordinate_full_forking_path.csv", index=False)

#import matplotlib.pyplot as plt
#import seaborn as sns
#cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
#sns.heatmap(test, cmap=cmap)









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
    
    if rdm == False:
        # recode conditions
        epochs = recode_conditions(epochs.copy(), version="superordinate")
    
    elif rdm == True:
        epochs = recode_conditions(epochs.copy(), version="categories")
    
    # extract data from epochs
    X = epochs.get_data()
    y = epochs.events[:,-1] - 1 # subtract 1 to get 0 and 1 instead of 1 and 2
    
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
    if "RSVP" in experiment:
        sfreq = 250 # TODO: maybe this was the MIPDB error?
    else:
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
                        n_jobs=1, # TODO: check if this avoids overload
                        return_estimator=False, # if you need the model to estimate on another test set
                        return_train_score=False,
                        )
    
    # save performance TODO    
    validation_acc = np.mean(cvs['test_score'])

    dfi = pd.DataFrame({'forking_path': [forking_path],
                    'accuracy': [validation_acc],
                    })
    dfi.to_csv(f"{model_folder}/{forking_path}.csv", index=False)
    # TODO: instead in a shared object?


# parallel processing
#Parallel(n_jobs=-1)(delayed(parallel_eegnet)(forking_path, file) for forking_path, file in zip(forking_paths, files))
Parallel(n_jobs=-1)(delayed(parallel_subject)(subject) for subject in subjects)

# TODO: summarize all these across subjects and experiments in a big results dataframe (other script)

# DEBUG
#parallel_eegnet(forking_paths[0], files[0])
#Parallel(n_jobs=10)(delayed(parallel_eegnet)(forking_path, file) for forking_path, file in zip(forking_paths, files))
