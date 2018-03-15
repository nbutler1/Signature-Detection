'''
File:        Image Filter
Date:        02/21/18
Authors:     Robert Neff, Nathan Butler
Description: Defines functions for reading and filtering images.
'''

import os
from scipy.signal import medfilt
from scipy import misc

'''
Function: filter_dir
--------------------
Filters all images in inpath directory, saves results
to the outpath directory and returns the new images.
'''
def filter_dir(inpath, outpath="", save=False):
    
    resulting_imgs = []
    
    for filename in os.listdir(inpath):
        if (filename.endswith(".jpg") or filename.endswith(".png")): 
            resulting_imgs.append(get_filtered_image(inpath, filename, outpath, save))
    
    return resulting_imgs

'''
Function: get_filtered_image
----------------------------
Loads and filters an image, saving it if desired.
'''
def get_filtered_image(inpath, filename, outpath, save=False):
    # Load image as grayscale image
    img = misc.imread(inpath + filename, mode='L')
    
	# Median filter it
    median = medfilt(img, 1) # less loss of definition than above option
	
    # Convert to binary
    threshold = 127
    for i in range(median.shape[0]):
        for j in range(median.shape[1]):
            if (median[i, j] > threshold):
                median[i, j] = 255
            else:
                median[i, j] = 0
   
     
    # Save new file
    if (save):
        misc.imsave(outpath + filename, median)
        
    return median
    