'''
File:        Discrete Cosine Transform (DCT)
Date:        03/16/18
Authors:     Robert Neff, Nathan Butler
Description: Implements the discrete cosine transformation of a 2D image via
             multiple methods, along with component extraction for each method.
'''

import numpy as np
import invariant_drt

'''
Function: block_dct
-------------------
Computes the largest DCT generated cofficients for each block of the 
image. 

Ref: Breaking up of image and fast DCT computation modelled after
https://www.math.cuhk.edu.hk/~lmlui/dct.pdf 
'''
def block_dct(img, num_blocks=1000):
    # TODO adaptive block size, whole image any different?
    M, N = img.shape[:2]
    block = 64
    
    dct_M = np.zeros((block, block))
    for i in range(block):
        for j in range(block):
            if (i == 0):
                dct_M[i, j] = 1 / np.sqrt(block)
            else:
                dct_M[i, j] = np.sqrt(2 / block) * np.cos((2 * j + 1) * i * np.pi / (2 * block))
	
    dct = []
    for i in range(0, M - block, block):
        for j in range(0, N - block, block):
            block_M = img[i:i+block,j:j+block] - 128
            dct.append(dct_M.dot(block_M).dot(dct_M.T)[0][0])
    
	return np.array(dct)
    
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
    return np.real(shift * shift[:N]) # return real component

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
Function: get_block_comps
-------------------------
Resizes desired number of components returned by block dct array
resizing as necessary.
'''
def get_block_comps(dct, n_components=1000):
    components = invariant_drt.interp_resize_1d(dct, n_components)
    return np.reshape(components, (2, components.shape[0] / 2)) # reshape for consistency with util fns
	
'''
Function: get_largest_freqs
---------------------------
Returns the MxN largest coefficients of the DCT matrix computed by fft_dct. 
The largest spatial frequencies are found as the coefficients in the upper 
left corner of the DCT.
'''
def get_largest_freqs(dct, M=10, N=10):
    components = dct[:M, :N]

    row_sums = components.sum(axis=1)
    normalized_comp = components / row_sums[:, np.newaxis]
    
    return normalized_comp
