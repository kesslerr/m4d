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


# plot montage --> infants 
epochs = mne.io.read_raw_fif(base_dir + "/data/raw/RSVP/sub-001-raw.fif", preload=True, verbose=None)
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
plt.title("Montage: RSVP")
plt.tight_layout()
fig.savefig(os.path.join(base_dir, "plots", "montage_rsvp.png"), dpi=300)


# plot montage --> MIPDB
file = sorted(glob("/ptmp/kroma/m4d/data/processed/MIPDB/A00051826/*.fif"))[0]
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
plt.title("EEG channels")
plt.tight_layout()
fig.savefig(os.path.join(base_dir, "plots", "montage_mipdb.png"), dpi=300)




#fig.savefig(os.path.join(base_dir, "manuscript", "montage.png"), dpi=300)

# test epoch compressing
#file = sorted(glob(base_dir + "/data/processed/N170/sub-001/*.fif"))[0]
#epochs = mne.read_epochs(file, preload=True, verbose=None)

#epochs32 = epochs.copy()
#epochs32._data = epochs._data.astype(np.float32)
# CAVE: saving int32 doesnt work, as mne forbids it
# gz saves maybe 5% of space, so save as fif.gz

#epochs.save('unc_epo.fif.gz', fmt='double', overwrite=True) # single is standard, and saves around 10-20%
#epochs.save('com_epo.fif.gz', fmt='single', overwrite=True)
