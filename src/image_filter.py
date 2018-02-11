'''
File:        Image Filter
Date:        02/07/18
Authors:     Robert Neff, Nathan Butler
Description: Defines functions for reading and filtering images.
'''

import cv2
import os
from scipy.signal import medfilt

'''
Function: filter_images
-----------------------
Filters all images in inpath directory, saves results
to the outpath directory and returns the new images
'''
def filter_images(inpath, outpath):
    
    resulting_imgs = []
    
    for filename in os.listdir(inpath):
        if (filename.endswith(".jpg") or filename.endswith(".png")): 
            resulting_imgs.append(get_filtered_image(inpath, filename, outpath, True))
    
    return resulting_imgs

'''
Function: get_filtered_image
----------------------------
Loads and filters an image, saving it if desired.
'''
def get_filtered_image(inpath, filename, outpath, save=False):
    # Load image as grayscale image
    img = cv2.imread(inpath + filename, 0)
    
    # Convert to binary
    threshold = 127
    img_binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
    
    # Median filter it
    # median = cv2.medianBlur(img_binary, 3)
    median = medfilt(img_binary, 1) # less loss of definition than above option
     
    # Save new file
    if (save):
        cv2.imwrite(outpath + filename, median)
        
    return median
    