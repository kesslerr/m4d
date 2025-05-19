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
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats


files = sorted(glob(f"/ptmp/kroma/m4d/models/ar_seed/eegnet/N170/*/*.csv"))
df = pd.concat([pd.read_csv(file) for file in files]).reset_index(drop=True)


""" ICC calculation """

import pingouin as pg
print(pg.__version__)

seed_types = df["seed_type"].unique()
ar_versions = df["ar_version"].unique()

iccs1 = []
iccs2 = []
for seed_type in seed_types:
    for ar_version in ar_versions:
        forking_paths = df["forking_path"].unique()
        
        for forking_path in forking_paths:

            #data = df[(df["forking_path"] == "None_None_45_0.1_average_linear_None_int" ) & (df["seed_type"] == "arseed")]
            data = df[(df["seed_type"] == seed_type) & (df["forking_path"] == forking_path)] # ar_version is implicit in forking_path
            icc = pg.intraclass_corr(data=data, 
                                    targets='subject',  # wine
                                    raters='seed_number', # judge
                                    ratings='accuracy') # scores
            icc["seed_type"] = seed_type
            icc["ar_version"] = ar_version
            icc["forking_path"] = forking_path
            iccs1.append(icc.iloc[0])  # only take the first row, which contains the ICC1,1 value
            iccs2.append(icc.iloc[1])  # only take the second row, which contains the ICC2,1 value

iccs1 = pd.DataFrame(iccs1).reset_index(drop=True)
iccs2 = pd.DataFrame(iccs2).reset_index(drop=True)

# multiple comparison correction:
iccs1["pval_corrected"] = multipletests(iccs1["pval"], method='fdr_bh')[1]
iccs2["pval_corrected"] = multipletests(iccs2["pval"], method='fdr_bh')[1]

iccs1.to_csv("/u/kroma/m4d/models/iccs1_participants_vs_seed.csv", index=False)
iccs2.to_csv("/u/kroma/m4d/models/iccs2_participants_vs_seed.csv", index=False)
# row with minimum icc1 value across entire dataset
iccs1min = iccs1.loc[iccs1["ICC"].idxmin()]
iccs2min = iccs2.loc[iccs2["ICC"].idxmin()]
#iccs1min.to_csv("/u/kroma/m4d/models/iccs1min.csv", index=False)
#iccs2min.to_csv("/u/kroma/m4d/models/iccs2min.csv", index=False)
print(iccs1min)

# are the iccs higher for different seed types?
#sampling_seed = iccs1.groupby(["seed_type", "forking_path"]).agg({"ICC": "mean"}).reset_index()

# are the iccs higher for different ar_versions?
iccs1 = []
iccs2 = []
for seed_type in seed_types:
    data = df[df["seed_type"] == seed_type]
    # mean accuracy across participants
    data = data.groupby(["forking_path", "seed_number", "ar_version"], as_index=False).agg({"accuracy": "mean"})
    
    icc = pg.intraclass_corr(data=data, 
                        targets='forking_path',  # wine
                        raters='seed_number', # judge
                        ratings='accuracy') # scores
    icc["seed_type"] = seed_type
    iccs1.append(icc.iloc[0])  # only take the first row, which contains the ICC1,1 value
    iccs2.append(icc.iloc[1])  # only take the second row, which contains the ICC2,1 value

iccs1 = pd.DataFrame(iccs1).reset_index(drop=True)
iccs2 = pd.DataFrame(iccs2).reset_index(drop=True)

# multiple comparison correction:
iccs1["pval_corrected"] = multipletests(iccs1["pval"], method='fdr_bh')[1]
iccs2["pval_corrected"] = multipletests(iccs2["pval"], method='fdr_bh')[1]

iccs1.to_csv("/u/kroma/m4d/models/iccs1_forkingpath_vs_seed.csv", index=False)
iccs2.to_csv("/u/kroma/m4d/models/iccs2_forkingpath_vs_seed.csv", index=False)




# # Function to compute ICC from ANOVA
# def compute_icc(model, factor):
#     # Perform the ANOVA test
#     anova_table = sm.stats.anova_lm(model, typ=2)
    
#     # Extract Mean Squares
#     MS_factor = anova_table.loc[factor, 'mean_sq']   # Mean Square of the factor
#     MS_residual = anova_table.loc['Residual', 'mean_sq']  # Mean Square of Residuals
    
#     # Number of levels in the factor
#     k = len(model.model.data.orig_exog)
    
#     # Compute ICC
#     ICC = (MS_factor - MS_residual) / (MS_factor + (k - 1) * MS_residual)
#     return ICC

# # Compute ICC for Seed (within each Seed Version)
# def icc_seed(data):
#     icc_seed_values = {}
    
#     for seed_version in data['seed_type'].unique():
#         subset = data[data['seed_type'] == seed_version]
#         model = ols('Accuracy ~ C(seed_number)', data=subset).fit()  # Fit model for each Seed Version
#         icc_seed_values[seed_version] = compute_icc(model, 'C(Seed)')
    
#     return icc_seed_values

# icc_seed_results = icc_seed(data)
# print("ICC for Seed (by Seed Version):")
# print(icc_seed_results)

# # Compute ICC for Participant (within each Seed Version, FPS, and AR Version)
# def icc_participant(data):
#     icc_participant_values = {}
    
#     for seed_version in data['Seed_Version'].unique():
#         subset = data[data['Seed_Version'] == seed_version]
#         model = ols('Accuracy ~ C(Participant)', data=subset).fit()  # Fit model for each Seed Version
#         icc_participant_values[seed_version] = compute_icc(model, 'C(Participant)')
    
#     return icc_participant_values

# icc_participant_seed = icc_participant(data)
# print("ICC for Participant (by Seed Version):")
# print(icc_participant_seed)

# # Compute ICC for FPS (Forking Path)
# model_fps = ols('Accuracy ~ C(FPS)', data=data).fit()
# icc_fps = compute_icc(model_fps, 'C(FPS)')
# print(f"ICC for FPS: {icc_fps}")

# # Compute ICC for AR Version
# model_ar = ols('Accuracy ~ C(AR_Version)', data=data).fit()
# icc_ar = compute_icc(model_ar, 'C(AR_Version)')
# print(f"ICC for AR Version: {icc_ar}")






# # one example pipeline, one seed type

# fp = "None_None_45_0.1_average_linear_None_int"
# seed = "arseed"

# dfi = df[(df["forking_path"] == fp) & (df["seed_type"] == seed)]


# # plot accuracy for each subject, scatterplot
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=dfi, x="subject", y="accuracy", style="seed_number", s=50, color="k") #hue="seed", 
# plt.title(f"Accuracy for {fp}, seed type: {seed}")
# plt.ylabel("Accuracy")
# plt.xlabel("Participant")
# plt.xticks(rotation=90)
# plt.tight_layout()
# #plt.savefig(f"/../plots/{fp}_{seed}_accuracy.png", dpi=300)
# plt.show()


# # now 1 for int, 1 for intrej

# df_filtered = df[df["ar_version"]=="int"]
# # Create a facet grid with seed_type on x and forking_path on y
# g = sns.relplot(
#     data=df_filtered,
#     x="subject",
#     y="accuracy",
#     col="seed_type",
#     row="forking_path",
#     style="seed_number",
#     s=50, color="k",
#     kind="scatter",
#     facet_kws={"margin_titles": True},  # Adds small titles for better readability
# )

# # Improve layout
# g.set_axis_labels("Participant", "Accuracy")
# g.set_titles(row_template="{row_name}", col_template="{col_name}")
# for ax in g.axes.flat:
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# plt.tight_layout()

# plt.show()


# # 2nd analysis:

# # avg accuracy across subjects ; per seed type and forking path and seed number

# df_avg = df.groupby(["forking_path", "seed_type", "seed_number", "ar_version"], as_index=False).agg({
#     "accuracy": "mean"  # Only average accuracy
# })

# # scatter plot: accuracy on y, forking path on x, seed type in x facets, ar_version in y facets
# g = sns.relplot(
#     data=df_avg,
#     x="forking_path",
#     y="accuracy",
#     col="seed_type",
#     #row="ar_version",
#     style="seed_number",
#     s=50, color="k",
#     kind="scatter",
#     facet_kws={"margin_titles": True},  # Adds small titles for better readability
# )

# # Improve layout
# g.set_axis_labels("Forking path", "Accuracy")
# g.set_titles(row_template="{row_name}", col_template="{col_name}")
# for ax in g.axes.flat:
#     ax.set_xticklabels("", rotation=90)
# plt.tight_layout()

# plt.show()
