'''
File:        Discrete Cosine Transform (DCT)
Date:        03/16/18
Authors:     Robert Neff, Nathan Butler
Description: Implements the discrete cosine transformation of a 2D image via
             custom and builtin methods, along with component extraction 
             for each method.
'''

import numpy as np
import invariant_drt
import scipy.fftpack

'''
Function: builtin_dct_2d
------------------------
Runs the scipy implementation of DCT for a 2D array.
'''
def builtin_dct_2d(img):
    return scipy.fftpack.dct(scipy.fftpack.dct(img.T).T)

'''
Function: dct_1d
----------------
Approximates the 1D Type 2 DCT by the real component of the discrete Fourier
transform (DFT). That is, the real component of the double-length FFT is very
similar to the DCT with a half-length phase shift. 
'''
def dct_1d(vec):
    N = len(vec)
	
    # Build doulbe-length vector
    vec2 = np.empty(2 * N,float)
    vec2[:N] = vec[:]
    vec2[N:] = vec[::-1]

    x = np.fft.rfft(vec2) # compute fft
    shift = np.exp(-1j * np.pi * np.arange(N) / (2 * N)) # find sinusoidal phase shift
    return np.real(shift * x[:N]) # return real component (first half)

'''
Function: compute_dct_2d
------------------------
Computes the 2D Type 2 DCT by taking the 1D DCT in both the row and
column directions.

Ref:
Mathematical help - 
https://www.dsprelated.com/freebooks/mdft/Discrete_Cosine_Transform_DCT.html

Code for FFT DCT modelled after Mark Newman's implementation - 
http://www-personal.umich.edu/~mejn/computational-physics/dcst.py
'''
def fft_dct_2d(img):
    M, N = img.shape[:2]
    A = np.empty([M,N],float)
    B = np.empty([M,N],float)
 
    # Get 1D DCT in along rows of img
    for i in range(M):
        A[i, :] = dct_1d(img[i, :])

    # Get 1D DCT along cols of A
    for i in range(N):
        B[:, i] = dct_1d(A[:, i])

    return B

'''
Function: get_largest_freqs
---------------------------
Returns the MxN largest coefficients of the DCT matrix computed by fft_dct2. 
The largest spatial frequencies are found as the coefficients in the upper 
left corner of the DCT.
'''
def get_largest_freqs(dct, M=10, N=10):
    components = dct[:M, :N]
    
    row_sums = components.sum(axis=1)
    normalized_comps = components / row_sums[:, np.newaxis]
	
    return normalized_comps # return 2-norm to make invariant
