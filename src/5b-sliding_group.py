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


# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#ptmp_dir = "/ptmp/kroma/m4d/"
os.chdir(base_dir)
sys.path.append(base_dir)

from src.utils import get_forking_paths, cluster_test
from src.config import translation_table, luck_forking_paths, subjects, experiments



forking_paths, files, forking_paths_split = get_forking_paths(
                            base_dir="/ptmp/kroma/m4d/", 
                            experiment="N170",
                            subject="sub-001", 
                            sample=None)



# TODO: across fps and experiments
#experiment = "N170"
results = []
histo = {}
for experiment in experiments:
    times = mne.read_epochs(f"/ptmp/kroma/m4d/data/processed/{experiment}/sub-001/average_0.5_45_None_None_400ms_linear_False-epo.fif").times
    #histo[experiment] = []
    #histo[experiment]['times'] = times # TODO: use this to convert indices to times
    #histo[experiment]['significant'] = [] # TODO: fill this with the exact time points
    print(f"Experiment: {experiment}")
    for forking_path in forking_paths:
        forking_path = forking_path.translate(translation_table)
        dfs = []
        for i, subject in enumerate(subjects):
            model_folder = os.path.join(base_dir, "models", "sliding", experiment, subject)
            df = pd.DataFrame({'balanced accuracy': np.load(os.path.join(model_folder, f"{forking_path}.npy")),
                            'times': times,
                            'subject': [subject] * times.shape[0],
                            'experiment': [experiment] * times.shape[0],
                            })
            dfs.append(df)
        df = pd.concat(dfs)

        # statistical test
        t_obs, clusters, cluster_pv, H0, times = cluster_test(df[df.experiment==experiment], side=1) # right-sided test

        # quantiy, how "strong" decoding was: 
        # idea: for each "significant" time point, sum up the difference to 0.5
        # window of interest? probably not, as there might be predictive time courses for a long period of time, which, for successfull decoding, should be helpful

        mean_ba = df.groupby(['experiment', 'times']).agg({'balanced accuracy': 'mean'}).reset_index()

        cumulative_decoding_accuracy = []
        timepoints = []
        
        #predictive_timepoints = []
        for i_c, c in enumerate(clusters):
            c = c[0] # get rid of the empty second element
            if cluster_pv[i_c] <= 0.05:
                print(f"{times[c[0]]}s - {times[c[-1]]}s: p={cluster_pv[i_c]:.3f}")
                cumulative_decoding_accuracy.append(((mean_ba[(mean_ba.times >= times[c[0]]) & (mean_ba.times <= times[c[-1]])]['balanced accuracy']-0.5)*2).values)
                timepoints.append((times[c[0]], times[c[-1]]))
            break
        break
    break

                # -.5, *2, to normalize between 0 and 1
                #predictive_timepoints.append(len(cumulative_decoding_accuracy[-1]))
        
        # Flatten the list of arrays to a 1D array
        if len(cumulative_decoding_accuracy) > 1:
            cumulative_decoding_accuracy = np.concatenate([arr.ravel() for arr in cumulative_decoding_accuracy])
        else: 
            cumulative_decoding_accuracy = cumulative_decoding_accuracy[0]

        n_predictive_timepoints = cumulative_decoding_accuracy.shape[0]
        
        cumulative_decoding_accuracy = np.sum(cumulative_decoding_accuracy)

        results.append(pd.DataFrame({'experiment': [experiment], 
                                                'forking path': [forking_path],
                                                'cumulative decoding accuracy': [cumulative_decoding_accuracy], 
                                                'predictive timepoints': [n_predictive_timepoints]}))

results = pd.concat(results)
results.to_csv(f"{base_dir}/models/sliding/results.csv", index=False)


# TODO: also write each predictive time point to results, so we can have a histogram across fps


# analyze and plot the performances only on the luck forking path for each experiment for exemplary visualization


# plot results with confidence intervals
ci = 95
n_subplots = len(experiments) 
fig, ax = plt.subplots(ncols=1, nrows=n_subplots, figsize=(10, 20), sharex=True, sharey=True)

for experiment in experiments:
    times = mne.read_epochs(f"/ptmp/kroma/m4d/data/processed/{experiment}/sub-001/average_0.5_45_None_None_400ms_linear_False-epo.fif").times

    forking_path = luck_forking_paths[experiment].translate(translation_table)

    dfs = []
    for i, subject in enumerate(subjects):
        model_folder = os.path.join(base_dir, "models", "sliding", experiment, subject)
        df = pd.DataFrame({'balanced accuracy': np.load(os.path.join(model_folder, f"{forking_path}.npy")),
                        'times': times,
                        'subject': [subject] * times.shape[0],
                        'experiment': [experiment] * times.shape[0],
                        })
        dfs.append(df)
    df = pd.concat(dfs)

    # statistical test
    t_obs, clusters, cluster_pv, H0, times = cluster_test(df[df.experiment==experiment], side=1)

    g = sns.lineplot(data=df[df.experiment==experiment], 
                     x='times', y='balanced accuracy', 
                     errorbar=("ci", ci), dashes=False, ax=ax[i]) #
    ax[i].set_title(f"{experiment}")
    ax[i].axhline(0.5, color='k')
    ax[i].axvline(0.0, color='k')

    # plot statistics    
    for i_c, c in enumerate(clusters):
        c = c[0] # get rid of the empty second element
        if cluster_pv[i_c] <= 0.05:
            h = ax[i].axvspan(times[c[0]], times[c[-1]], color="r", alpha=0.3)
        else:
            ax[i].axvspan(times[c[0]], times[c[-1]], color=(0.3, 0.3, 0.3), alpha=0.3)
        
        # write cluster p-value as txt
        if cluster_pv[i_c] <= 0.05:
            ax[i].text(times[c[0]], 0.48, f"p={cluster_pv[i_c]:.3f}", fontsize=8)
    

plt.tight_layout()
plt.savefig(f"{base_dir}/plots/sliding.png", dpi=300)
plt.show()
