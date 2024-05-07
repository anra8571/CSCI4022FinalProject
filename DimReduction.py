# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA

import numpy as np
from sklearn import decomposition

# Input: flat_fv (Participant.fv_reduced, the 2D array)
def feature_reduction(fv):
    fv = np.array(fv)
    pca = decomposition.PCA()
    transformed = pca.fit_transform(fv)

    return transformed