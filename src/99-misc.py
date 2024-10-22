import os, sys
import mne
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import seaborn as sns 
# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.append(base_dir)

""" HEADER END"""

# plot montage --> ERPCORE
file = sorted(glob("/ptmp/kroma/m4d/data/processed/N170/sub-001/*.fif"))[0]
epochs = mne.read_epochs(file, preload=True, verbose=None)
fig, ax = plt.subplots(figsize=(6, 6))
epochs.get_montage().plot(
    scale_factor=20,
    show_names=True,
    kind="topomap",
    show=False,
    sphere=None,
    axes=ax,
    verbose=None,
);
plt.title("Montage: ERPCORE")
plt.tight_layout()
fig.savefig(os.path.join(base_dir, "plots", "montage_erpcore.png"), dpi=300)



