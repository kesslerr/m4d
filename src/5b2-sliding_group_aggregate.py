import sys, os
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

""" concatenate the single experiment outputs """

experiments = experiments_erpcore    

df_results = []
for experiment in experiments:
    dfi = pd.read_csv(f"{base_dir}/models/sliding/sliding_extended_R1_{experiment}.csv")
    df_results.append(dfi)    
df_results = pd.concat(df_results)
df_results.to_csv(f"{base_dir}/models/sliding_extended_R1.csv", index=False)


df_tsums = []
for experiment in experiments:
    dfi = pd.read_csv(f"{base_dir}/models/sliding/sliding_tsums_extended_R1_{experiment}.csv")
    df_tsums.append(dfi)        
df_tsums = pd.concat(df_tsums)
df_tsums.to_csv(f"{base_dir}/models/sliding_tsums_extended_R1.csv", index=False)

# due to memory constraints, write in chunks
csv_files = sorted(glob(f"{base_dir}/models/sliding/sliding_single_extended_R1_*.csv"))
with open(f"{base_dir}/models/sliding_single_extended_R1.csv", "w") as fout:
    for i, file in tqdm(enumerate(csv_files)):
        chunks = pd.read_csv(file, chunksize=10000)  # Adjust chunk size as needed
        for chunk in chunks:
            chunk.to_csv(fout, index=False, header=(i == 0), mode="a")

# df_results_single = []
# for experiment in experiments:
#     dfi = pd.read_csv(f"{base_dir}/models/sliding/sliding_single_extended_R1_{experiment}.csv")
#     df_results_single.append(dfi)        
# df_results_single = pd.concat(df_results_single)
# df_results_single.to_csv(f"{base_dir}/models/sliding_single_extended_R1.csv", index=False)

df_avg_accs_single = []
for experiment in experiments:
    dfi = pd.read_csv(f"{base_dir}/models/sliding/sliding_avgacc_single_extended_R1_{experiment}.csv")
    df_avg_accs_single.append(dfi)        
df_avg_accs_single = pd.concat(df_avg_accs_single)
# df_avg_accs_single = df_avg_accs_single.rename(columns={'balanced accuracy': 'accuracy'})
# df_avg_accs_single.drop('forking_path', axis=1, inplace=True)
df_avg_accs_single.to_csv(f"{base_dir}/models/sliding_avgacc_single_extended_R1.csv", index=False)



""" PLOT exemplary FP results, and stats """


# analyze and plot the performances only on the luck forking path for each experiment for exemplary visualization

df_results = pd.read_csv(f"{base_dir}/models/sliding_extended_R1.csv")
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
g.savefig(os.path.join(plot_dir, f"timeresolved_luck_tfce.png"), dpi=300)
plt.show()
