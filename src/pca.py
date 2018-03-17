'''
File:        Principle Component Analysis (PCA)
Date:        02/21/18
Authors:     Robert Neff, Nathan Butler
Description: Extracts the principle features of a DRT generated sinogram.
'''

import numpy as np
from sklearn.decomposition import PCA

'''
Function: sklearn_pca
---------------------
Returns the PCA reconstruction of the provided data using "n" greatest 
components via the sklearn library.
'''
def sklearn_pca(data, n=10):
    pca = PCA(n_components=n)
    pca.fit(data)
    components = pca.transform(data)
    filtered = pca.inverse_transform(components)
    return filtered

'''
Function: get_pca
-----------------
Extracts the principle components from the drt generated sinogram,
keeping the P greatest contributing features from angle features represented.
'''
def get_pca(sinogram, P=2):
    samples, features = sinogram.shape
    
    # Get average DRT features per angle
    R_ave = np.mean(sinogram, axis=0)
    sinogram -= R_ave
    
    # Get eigenvectors from svd
    # The right singular vectors (rows of VT) = eigenvectors of the 
    # covariance matrix, i.e. sinogram * sinogram^T
    U, s, VT = np.linalg.svd(sinogram, full_matrices=False)
    U, VT = svd_flip(U, VT)
    eig_vecs = VT
    
    # Compute eigen signatures
    eigen_signatures = np.zeros((features, features))
    for i in range(features):
        for j in range(features):
            eigen_signatures[i, :] += eig_vecs[i, j] * sinogram[j, :]
    
    return eigen_signatures[:P]
    
'''
Function: svd_flip
------------------
Adjusts the columns of u and the rows of v such that the loadings in the
columns in u that are largest in absolute value are always positive.

Ref: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py#L503
'''  
def svd_flip(U, V):
    # Rows of V, columns of U
    max_abs_rows = np.argmax(np.abs(V), axis=1)
    signs = np.sign(V[xrange(V.shape[0]), max_abs_rows])
    U *= signs
    V *= signs[:, np.newaxis]
    return U, V
