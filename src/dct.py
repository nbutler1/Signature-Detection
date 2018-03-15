'''
File:        Discrete Cosine Transform (DCT)
Date:        03/14/18
Authors:     Robert Neff, Nathan Butler
Description: Implements the discrete cosine transformation of a 2D image.

Ref:
Mathematical help - 
https://www.dsprelated.com/freebooks/mdft/Discrete_Cosine_Transform_DCT.html

Code modelled after Mark Newman's implementation - 
http://www-personal.umich.edu/~mejn/computational-physics/dcst.py
'''

from numpy import empty, arange, exp, real, imag, pi, newaxis
from numpy.fft import rfft, irfft

'''
Function: dct_1d
----------------
Approximates the 1D Type 2 DCT by the real component of the discrete Fourier
transform (DFT). That is, the real component of the double-length FFT is the
same as the DCT with a half-length phase shift. 
'''
def dct_1d(vec):
    N = len(vec)
	
    # Build doulbe-length vector
    vec2 = empty(2 * N,float)
    vec2[:N] = vec[:]
    vec2[N:] = vec[::-1]

    x = rfft(vec2) # compute fft
    shift = exp(-1j * pi * arange(N) / (2 * N)) # find sinusoidal phase shift
    return real(shift * shift[:N]) # return real component

'''
Function: compute_dct_2d
------------------------
 Computes the 2D Type 2 DCT by taking the 1D DCT in both the row and
 column directions.
'''
def compute_dct_2d(img):
    M, N = img.shape[:2]
    A = empty([M,N],float)
    B = empty([M,N],float)
 
    # Get 1D DCT in along rows of img
    for i in range(M):
        A[i, :] = dct_1d(img[i, :])
    
    # Get 1D DCT along cols of A
    for i in range(N):
        B[:, i] = dct_1d(A[:, i])

    return B

'''
Function: get_components
------------------------
Returns the MxN largest coefficients of the DCT. The largest spatial frequencies
are found as the coefficients in the upper left corner of the DCT.
'''
def get_components(dct, M=10, N=10):
    components = dct[:M, :N]

    row_sums = components.sum(axis=1)
    normalized_comp = components / row_sums[:, newaxis]
    
    return normalized_comp

