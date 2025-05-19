import numpy as np
import pandas as pd
import os
import sys
from glob import glob
from tqdm import tqdm
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats





files = sorted(glob(f"/ptmp/kroma/m4d/models/leakage/eegnet/N170/*/eegnet_leakage.csv"))
df = pd.concat([pd.read_csv(file) for file in files]).reset_index(drop=True)

# for each subject, experiment, step, and leakage, get the mean accuracy across splits (drop the split column)
df = df.groupby(["subject", "experiment", "step", "leakage"]).mean().reset_index().drop(["split"], axis=1)

# rename step column: ar: autoreject, hpf: high-pass, ica: ICA
df["step"] = df["step"].replace({"ar": "autoreject", "hpf": "high-pass", "ica": "ICA"})

""" statistical test """

# for each step, compare leaky vs sealed accuracy
# for each step, compare leaky vs sealed accuracy

for step in df["step"].unique():
    print(f"Step: {step}")
    df_step = df[df["step"] == step]
    
    # Perform a paired t-test for leaky vs sealed accuracy
    leaky_acc = df_step[df_step["leakage"] == "leaky"]["accuracy"]
    sealed_acc = df_step[df_step["leakage"] == "sealed"]["accuracy"]
    
    t_stat, p_val = stats.ttest_rel(leaky_acc, sealed_acc, alternative="greater")
    # tests H1: acc_leaky > acc_sealed 
    
    # Compute degrees of freedom (df)
    df_t = len(leaky_acc) - 1
    
    # Calculate means & standard deviations
    mean_leaky, std_leaky = np.mean(leaky_acc), np.std(leaky_acc, ddof=1)
    mean_sealed, std_sealed = np.mean(sealed_acc), np.std(sealed_acc, ddof=1)
    
    
    print(f"T({df_t}) = {t_stat}, p = {p_val}, mean_leaky = {mean_leaky:.3f} ± {std_leaky:.3f}, mean_sealed = {mean_sealed:.3f} ± {std_sealed:.3f}")




""" scatter plots """
# Pivot the DataFrame to get leaky vs. sealed accuracy in separate columns
df_wide = df.pivot(index=["subject", "step"], columns="leakage", values="accuracy").reset_index()

# Rename columns for clarity
df_wide.rename(columns={"leaky": "Accuracy_Leaky", "sealed": "Accuracy_Sealed"}, inplace=True)

# Set up the FacetGrid for each "step"
g = sns.FacetGrid(df_wide, col="step", sharex=True, sharey=True, height=4, aspect=1)
g.map_dataframe(sns.scatterplot, x="Accuracy_Leaky", y="Accuracy_Sealed", alpha=0.6, color="black")

# Add x = y diagonal line
for ax in g.axes.flat:
    min_val = min(df_wide[["Accuracy_Leaky", "Accuracy_Sealed"]].min())  # Get min value
    max_val = max(df_wide[["Accuracy_Leaky", "Accuracy_Sealed"]].max())  # Get max value
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="darkred")

# Improve layout
g.set_axis_labels("Accuracy (Leaky)", "Accuracy (Sealed)")
g.set_titles("Step: {col_name}")

# Add p-values
for ax in g.axes.flat:
    title = ax.get_title()
    step = title.split(": ")[-1]  # Extract step from title
    # Select valid (non-NaN) data
    mask = df_wide["step"] == step
    leaky_acc = df_wide.loc[mask, "Accuracy_Leaky"].dropna()
    sealed_acc = df_wide.loc[mask, "Accuracy_Sealed"].dropna()
    
    if len(leaky_acc) > 1:  # Ensure enough data for t-test
        t_stat, p_val = stats.ttest_rel(leaky_acc, sealed_acc, alternative="greater")
        df_t = len(leaky_acc) - 1  # Degrees of freedom

        # Add text to the plot
        ax.text(0.05, 0.95, f"T({df_t}) = {t_stat:.2f}, p = {p_val:.3f}", 
                ha="left", va="top", transform=ax.transAxes)
    else:
        ax.text(0.05, 0.95, "N/A", ha="left", va="top", transform=ax.transAxes)

# Save and show
plt.savefig("../plots/leakage_scatter.png", dpi=300)
plt.show()




# """ barplot """

# # barplot with mean accuracy for each leakage and step
# plt.figure(figsize=(10, 6))
# sns.barplot(data=df, x="step", y="accuracy", hue="leakage",
#             errorbar=("ci", 95), palette="Set2")
# plt.ylim(0.5,1)
# plt.show()

""" analyze the logfile / parameters """

files = sorted(glob(f"/ptmp/kroma/m4d/data/processed/leakage/N170/*/ica_dropped_components.csv"))
df = pd.concat([pd.read_csv(file) for file in files]).reset_index(drop=True).drop("experiment", axis=1)


df_avg = df.groupby(["subject", "pipeline"], as_index=False).agg({
    "n_components_dropped": "mean"  # Only average accuracy
})

# pair-plot, one pair per subject
plt.figure(figsize=(8, 6))
sns.lineplot(
    data=df_avg,
    x="pipeline",
    y="n_components_dropped",
    alpha=0.6,
    #s=100,  # Adjust dot size
    hue="subject",  # Color by subject
)

# t-test leaky vs sealed

import scipy.stats as stats
leaky = df_avg[df_avg.pipeline=="leaky"]["n_components_dropped"]
sealed = df_avg[df_avg.pipeline=="sealed"]["n_components_dropped"]

# Perform a paired t-test for leaky vs sealed accuracy
T, p = stats.ttest_rel(leaky, sealed, alternative="greater")


# plot the n_components_cropped on y,
# the subject at x,
# the pipeline step on hue