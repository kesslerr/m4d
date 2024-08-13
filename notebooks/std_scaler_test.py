
# test the standard scaler
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

StandardScaler()

# X: (n_samples, n_dimensional_features,)
X = np.random.randint(0, 10, (5,5))

sX = StandardScaler().fit_transform(X)

