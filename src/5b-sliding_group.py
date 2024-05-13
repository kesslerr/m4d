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

from src.utils import get_forking_paths, cluster_test, get_age
from src.config import translation_table, luck_forking_paths, subjects as subjects_erpcore, experiments as experiments_erpcore, decoding_windows, subjects_mipdb_dem, age_groups, groups_subjects_mipdb, luck_forking_paths



forking_paths, files, forking_paths_split = get_forking_paths(
                            base_dir="/ptmp/kroma/m4d/", 
                            experiment="N170",
                            subject="sub-001", 
                            sample=None)

## SUMMARIZE RESULTS AND STATS FOR EXPORT

df_results = []
df_tsums = []

failed_subs = set()
for dataset in ['erpcore']: # TODO: add 'mipdb', 
    
    if dataset == "erpcore":
        experiments = experiments_erpcore
        subjects = subjects_erpcore
    elif dataset == "mipdb":
        experiments = age_groups.keys()
    
    for experiment in experiments: # in MIPDB, experiment is in the subject group
        
        
        print(f"Experiment: {experiment}")

        if dataset == "erpcore":
            epoch_example_file = f"/ptmp/kroma/m4d/data/processed/{experiment}/sub-001/None_None_45_0.5_average_400ms_linear_False-epo.fif"
            #epoch_example_file = f"/ptmp/kroma/m4d/data/processed/{experiment}/sub-001/average_0.5_45_None_None_400ms_linear_False-epo.fif"
            tmin = decoding_windows[experiment][0]
            tmax = decoding_windows[experiment][1]
        elif dataset == "mipdb":
            epoch_example_file = f"/ptmp/kroma/m4d/data/processed/MIPDB/A00053597/average_0.5_45_None_None_400ms_linear_False-epo.fif"
            tmin = decoding_windows["MIPDB"][0]
            tmax = decoding_windows["MIPDB"][1]
            subjects = groups_subjects_mipdb[experiment]
            
        times = mne.read_epochs(
            epoch_example_file).crop(
            tmin=tmin,
            tmax=tmax).times

        for forking_path, forking_paths_split_i in zip(forking_paths, forking_paths_split):
            forking_path = forking_path.translate(translation_table) # might be redundant if all special characters are already removed

            dfs = []
            for i, subject in enumerate(subjects):
                if dataset == "erpcore":
                    model_folder = os.path.join(base_dir, "models", "sliding", experiment, subject)
                elif dataset == "mipdb":
                    model_folder = os.path.join(base_dir, "models", "sliding", "MIPDB", subject)
                
                if os.path.exists(os.path.join(model_folder, f"{forking_path}.npy")):
                    df = pd.DataFrame({'balanced accuracy': np.load(os.path.join(model_folder, f"{forking_path}.npy")),
                                    'times': times,
                                    'subject': [subject] * times.shape[0],
                                    'experiment': [experiment] * times.shape[0],
                                    })
                else:
                    #print(f"\n \n !!! \n Missing file: {subject} \n !!! \n \n")
                    failed_subs.add(subject)
                    continue
                dfs.append(df)
            df = pd.concat(dfs)

            # statistical test
            t_obs, clusters, cluster_pv, H0, times = cluster_test(df, side=1) # right-sided test

            # 1
            tsum = 0
            # 2
            df_mean = df.groupby(['times']).agg({'balanced accuracy': 'mean'}).reset_index()
            df_mean["significance"] = False
            df_mean["p"] = np.nan
            
            
            for i_c, c in enumerate(clusters):
                c = c[0] # get rid of the empty second element which refers to frequency dimension
                if cluster_pv[i_c] <= 0.05:
                    #print(f"{times[c[0]]}s - {times[c[-1]]}s: p={cluster_pv[i_c]:.3f}")
                    
                    # sum of tvalues in cluster
                    tsum += np.sum(t_obs[c])
                    
                    # write p value to df
                    df_mean.loc[c[0]:c[-1]+1, "p"] = cluster_pv[i_c]
                    df_mean.loc[c[0]:c[-1]+1, "significance"] = True

            #df_mean[['ref','hpf','lpf','emc','mac','base','det','ar']] = forking_paths_split_i
            df_mean[['emc','mac','lpf','hpf','ref','base','det','ar']] = forking_paths_split_i
            df_mean['forking_path'] = forking_path
            df_mean['experiment'] = experiment
            df_mean['dataset'] = dataset
            df_results.append(df_mean)      
            
            df_tsum = pd.DataFrame({
                'tsum': tsum,
                'experiment': experiment,
                'dataset': dataset,
                }, index=[0])
            df_tsum[['emc','mac','lpf','hpf','ref','base','det','ar']] = forking_paths_split_i
            #df_tsum[['ref','hpf','lpf','emc','mac','base','det','ar']] = forking_paths_split_i
            df_tsum['forking_path'] = forking_path
            df_tsums.append(df_tsum)
    
df_results = pd.concat(df_results)
df_results.to_csv(f"{base_dir}/models/sliding/sliding_reordered.csv", index=False)

df_tsums = pd.concat(df_tsums)
df_tsums.to_csv(f"{base_dir}/models/sliding/sliding_tsums_reordered.csv", index=False)


## PLOT exemplary FP results, and maybe stats

# analyze and plot the performances only on the luck forking path for each experiment for exemplary visualization
# TODO: continue here with some result plots


# df_results = pd.read_csv(f"{base_dir}/models/sliding/sliding.csv")
# df_results.head()

# # select the forking path of Luck et al.
# # DEBUG
# #experiment = "N170"
# dfs_single = []
# for experiment in experiments:

#     forking_path = luck_forking_paths[experiment].translate(translation_table)
#     df_single = df_results[df_results['forking_path'] == forking_path]
#     df_single.shape
#     dfs_single.append(df_single)

# dfs_single = pd.concat(dfs_single)

# for dataset in ['erpcore']: # TODO: add 'mipdb'
#     fig, ax = plt.subplots(len(experiments), 1, figsize=(15, 5))
#     for i, experiment in enumerate(experiments):
#         g = sns.relplot(data=dfs_single[dfs_single.dataset==dataset], 
#                     x='times', y='balanced accuracy', 
#                     hue='significance',
#                     #row='experiment',
#                     ax=ax[i],
#                     #kind='line'
#                     )
#         # ax[i].axhline(0.5, color='k')
#         for ax in g.axes.flat:
#             for line in ax.lines:
#                 x = line.get_xdata()
#                 y = line.get_ydata()
#                 significance = dfs_single.loc[dfs_single['experiment'] == ax.get_title()].loc[(x >= min(x)) & (x <= max(x)), 'significance']
#                 start = None
#                 for i in range(len(significance)):
#                     if significance.iloc[i]:
#                         if start is None:
#                             start = i
#                     elif start is not None:
#                         ax.fill_between(x[start:i], y[start:i], color='gray', alpha=0.3)
#                         start = None

#     plt.show()


# sns.relplot(data=df, x='x', y='y', hue='color_var', col='facet_var', kind='line')









