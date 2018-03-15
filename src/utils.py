'''
File:        Utils
Date:        03/14/18
Authors:     Robert Neff, Nathan Butler
Description: Defines utlity functions for processing and loading data.
'''

import image_filter
import invariant_drt
import dct
import os
import pca
import pickle
import random
import numpy as np

"""
Function: parse_data
--------------------
This function randomly splits data into test and train sets
based on probability r of being in the test set.
"""
def parse_data(data, r=0.95):
    train, test = [], []
    for i in range(len(data)):
        val = random.uniform(0, 1)
        if val< r:
            test.append(data[i])
        else:
            train.append(data[i])
    return train, test

"""
Function: transform
-------------------
This function takes as input data in the form returned by
the pickling operation.  Comp_method takes two inputs (or two
pca components) and outputs a single np array.  This function
transforms the data points into a single array like object.
"""
def transform(data, comp_method=None):
    batch, labels = [], []
    if comp_method is None:
        def comp_method(p1, p2):
            return p1 - p2
    for i in range(len(data)):
        batch.append(comp_method(data[i]['1'], data[i]['2']))
        s = batch[-1].shape
        batch[-1] = np.reshape(batch[-1], (s[0]*s[1]))
        labels.append(data[i]['label'])
    return [np.stack(batch, axis = 0), np.array(labels)]

"""
Function: get_images_pipeline
-----------------------------
Given a file that contains multiple images, filters the images,
runs drt on them, then pca, then returns the images components.
"""
def get_images_pipeline2(in_file):
    print "Filtering images ..."
    images = image_filter.filter_dir(in_file)
    image_components = []
    
    print "Getting image components drt..."
    for img in images:
        # Compute DRT and then PCA
        sinogram = invariant_drt.compute_drt(img)
        new_sinogram = invariant_drt.post_process_sinogram(sinogram, 400)
        components = pca.get_pca(new_sinogram)
        image_components.append(components)
    return image_components	

"""
Function: get_images_pipeline2
------------------------------
Given a file that contains multiple images, filters the images,
runs dct on them, then returns the 10 x 10 greatest signals per image.
"""
def get_images_pipeline(in_file):
    print "Filtering images ..."
    images = image_filter.filter_dir(in_file)
    image_components = []
    
    print "Getting image components dct ..."
    for img in images:
        # Compute DCT and then components
        image_dct = dct.compute_dct_2d(img)
        components = dct.get_components(image_dct)
        image_components.append(components)
    return image_components
	
"""
Function: pickle_data
---------------------
This function takes in the path to training data and a file name to
save pickled data to.  If the outfile exists, it loads the data
and returns it.  If not, it loops through every test users image
set.  For each image, it first filters each image using DRT
and PCA.  Then, it creates a dictionary of training instances
and pickles the data to the outfile.  The path should point to 
a directory of the following form:
path - 
    userA -
        Reference - 
            *** all genuine signatures ***
        Simulated - 
            *** all forgeries ***
    userB
    ...
Data output is all combinations of genuine and forged signatures
(either genuine-genuine pairs or genuine-forged pairs).  Each
datapoint is of the form:
    {'1'     : pca of sig 1
     '2'     : pca of sig 2
     'label' : 0 if sig 2 forged, 100 if genuine (representing persentages
    }
"""
def pickle_data(path, outfile):
    if os.path.isfile(outfile):
        return pickle.load(open(outfile, 'r'))
    data = []
    for f in os.listdir(path):
        print f
        if f[0] != '.':
            genuine = []
            fakes = []
            for filename in os.listdir(path + f + "/"):
                if filename == "Reference":
                    print "Reference"
                    genuine = get_images_pipeline(path + f + "/" + filename + "/")
                elif filename == "Simulated":
                    print "Simulated"
                    fakes = get_images_pipeline(path + f + "/" + filename + "/")
            for i in range(len(genuine)):
                comp1 = genuine[i]
                label = 100.0
                for j in range(len(genuine)):
                    comp2 = genuine[j]
                    data.append({'1': comp1, '2': comp2, 'label': label})
                label = 0.0
                for j in range(len(fakes)):
                    comp2 = fakes[j]
                    data.append({'1': comp1, '2': comp2, 'label': label})
    pickle.dump(data, open(outfile, 'w+'))
    return pickle.load(open(outfile, 'r'))
