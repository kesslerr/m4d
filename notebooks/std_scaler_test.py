
# test the standard scaler
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

StandardScaler()

import mne
epochs = mne.read_epochs(f"/ptmp/kroma/m4d/data/processed/N170/sub-001/ica_ica_None_0.1_Cz_linear_None_int-epo.fif", 
                         preload=True)


X = epochs.get_data() # (n_epochs, n_channels, n_times)

print(X.shape)


# for one timepoint
Xtp = X[:, :, 0]

sXtp = StandardScaler().fit_transform(Xtp)

# show the data distribution across the 2 dimensions of xStp
import matplotlib.pyplot as plt

# show for feature 1 distribution (across channels)
plt.hist(Xtp[:, 0], bins=50)
plt.title("Feature 1")
plt.show()

# show for epoch 1 distribution
plt.hist(Xtp[0, :], bins=50)
plt.title("Epoch 1")
plt.show()




# X: (n_samples, n_dimensional_features,)
#X = np.random.randint(0, 10, (5,5))
#sX = StandardScaler().fit_transform(X)

