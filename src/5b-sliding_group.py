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
plot_dir = os.path.join(base_dir, "plots")

from src.utils import get_forking_paths, cluster_test, get_age
from src.config import translation_table, luck_forking_paths, subjects as subjects_erpcore, experiments as experiments_erpcore, decoding_windows, baseline_end



forking_paths, files, forking_paths_split = get_forking_paths(
                            base_dir="/ptmp/kroma/m4d/", 
                            experiment="P3",
                            subject="sub-026", # this session and experiment has all possible FPs 
                            sample=None)

## SUMMARIZE RESULTS AND STATS FOR EXPORT

df_results = []
df_tsums = []
df_results_single = [] # new: also extract and save single participant timeseries
df_avg_accs_single = []

failed_subs = []
failed_complete_fps = []

experiments = experiments_erpcore
subjects = subjects_erpcore

for experiment in experiments: # in MIPDB, experiment is in the subject group
    
    
    print(f"Experiment: {experiment}")

    epoch_example_file = f"/ptmp/kroma/m4d/data/processed/{experiment}/sub-001/None_None_45_0.5_average_linear_400ms_False-epo.fif"
    #epoch_example_file = f"/ptmp/kroma/m4d/data/processed/{experiment}/sub-001/average_0.5_45_None_None_400ms_linear_False-epo.fif"
    tmin = decoding_windows[experiment][0]
    tmax = decoding_windows[experiment][1]
        
    times = mne.read_epochs(
        epoch_example_file).crop(
        tmin=tmin,
        tmax=tmax).times

    for forking_path, forking_paths_split_i in zip(forking_paths, forking_paths_split):
        forking_path = forking_path.translate(translation_table) # might be redundant if all special characters are already removed

        dfs = []
        for i, subject in enumerate(subjects):
            model_folder = os.path.join(base_dir, "models", "sliding", experiment, subject)
            
            if os.path.exists(os.path.join(model_folder, f"{forking_path}.npy")):
                df = pd.DataFrame({'balanced accuracy': np.load(os.path.join(model_folder, f"{forking_path}.npy")),
                                'times': times,
                                'subject': [subject] * times.shape[0],
                                'experiment': [experiment] * times.shape[0],
                                })
            else:
                #print(f"\n \n !!! \n Missing file: {subject} \n !!! \n \n")
                failed_subs.append(subject + " --> " + forking_path)
                continue
            dfs.append(df)
            
        if len(df) < 20:
            failed_complete_fps.append(forking_path)
            continue
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

        #df_mean[['ref','hpf','lpf','emc','mac','det','base','ar']] = forking_paths_split_i
        df_mean[['emc','mac','lpf','hpf','ref','det','base','ar']] = forking_paths_split_i
        df_mean['forking_path'] = forking_path
        df_mean['experiment'] = experiment
        df_results.append(df_mean)      
        
        df[['emc','mac','lpf','hpf','ref','det','base','ar']] = forking_paths_split_i
        df['forking_path'] = forking_path
        df['experiment'] = experiment
        df_results_single.append(df)      

        df_tsum = pd.DataFrame({
            'tsum': tsum,
            'experiment': experiment,
            }, index=[0])
        df_tsum[['emc','mac','lpf','hpf','ref','det','base','ar']] = forking_paths_split_i
        #df_tsum[['ref','hpf','lpf','emc','mac','det','base','ar']] = forking_paths_split_i
        df_tsum['forking_path'] = forking_path
        df_tsums.append(df_tsum)
        
        # extract average accuracies for each subject
        dftmp = df[df.times >= baseline_end[experiment]]
        df_avg_acc = dftmp.groupby(['subject']).agg({'balanced accuracy': 'mean'}).reset_index()
        df_avg_acc[['emc','mac','lpf','hpf','ref','det','base','ar']] = forking_paths_split_i
        #df_tsum[['ref','hpf','lpf','emc','mac','det','base','ar']] = forking_paths_split_i
        df_avg_acc['experiment'] = experiment
        df_avg_acc['forking_path'] = forking_path
        df_avg_accs_single.append(df_avg_acc)
    
df_results = pd.concat(df_results)
df_results.to_csv(f"{base_dir}/models/sliding/sliding_extended.csv", index=False)

df_tsums = pd.concat(df_tsums)
df_tsums.to_csv(f"{base_dir}/models/sliding/sliding_tsums_extended.csv", index=False)

df_results_single = pd.concat(df_results_single)
df_results_single.to_csv(f"{base_dir}/models/sliding/sliding_single_extended.csv", index=False)

df_avg_accs_single = pd.concat(df_avg_accs_single)
df_avg_accs_single = df_avg_accs_single.rename(columns={'balanced accuracy': 'accuracy'})
df_avg_accs_single.drop('forking_path', axis=1, inplace=True)
df_avg_accs_single.to_csv(f"{base_dir}/models/sliding/sliding_avgacc_single_extended.csv", index=False)



""" PLOT exemplary FP results, and stats """


# analyze and plot the performances only on the luck forking path for each experiment for exemplary visualization

df_results = pd.read_csv(f"{base_dir}/models/sliding/sliding_extended.csv")
# rename dfs_single balanced accuracy column to accuracy
df_results = df_results.rename(columns={'balanced accuracy': 'Accuracy'})
df_results.head()

# select the forking path of Luck et al.

dfs_single = []
for experiment in experiments_erpcore:

    forking_path = luck_forking_paths[experiment].translate(translation_table)
    df_single = df_results[df_results['forking_path'] == forking_path]
    df_single.shape
    dfs_single.append(df_single)

dfs_single = pd.concat(dfs_single)
dfs_single


# Calculate the y-axis limit
max_y = max(0.7, dfs_single['Accuracy'].max()) # new, to make the axis nicer, include 0.7

# plot

g = sns.relplot(data=dfs_single, 
            x='times', y='Accuracy', 
            row='experiment',
            #ax=ax[i],
            kind='line',
            color='black',
            height=1.5,# of each facet
            aspect=5, # aspect * height = width
            #errorbar=('ci', 99), # there is already only the mean in the data, so no CI possible, if wanted, then extract single sub data
            #err_kws={"alpha": .4},
            facet_kws={'sharey': True, 'sharex': False}
            )

axes = g.fig.axes
for ax, experiment in zip(axes, experiments_erpcore):
    # new, include 0.7
    ax.set_ylim(.45, max_y)
    
    ax.axhline(0.5, color='k', linestyle='-')
    if experiment in ["LRP", "ERN"]:
        ax.axvline(0., color='k', linestyle=':') # response onset
    else:      
        ax.axvline(0., color='k', linestyle='--') # stimulus onset
    
    ax.set_title(experiment)
    ax.title.set_backgroundcolor('lightgrey')
    ax.title.set_fontsize(14)
    # significances
    for line in ax.lines:
        x = line.get_xdata()
        y = line.get_ydata()
        significance = dfs_single.loc[dfs_single['experiment'] == ax.get_title(), 'significance'] #.loc[(x >= min(x)) & (x <= max(x))
        start = None
        for i in range(len(significance)):
            if significance.iloc[i]:
                if start is None:
                    start = i
            elif start is not None:
                ax.fill_between(x[start:i], y[start:i], 0.45, color='gray', alpha=0.3)
                start = None
    
axes[-1].set_xlabel("Time [s]")
#g.map.set_titles("{experiment}")
plt.tight_layout()
# save plot
g.savefig(os.path.join(plot_dir, f"timeresolved_luck.png"), dpi=300)
plt.show()
