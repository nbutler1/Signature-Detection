'''
File:        Utils
Date:        03/16/18
Authors:     Robert Neff, Nathan Butler
Description: Defines utlity functions for processing, loading and assessing data.
'''

import numpy as np
import image_filter
import invariant_drt
import dct
import os
import pca
import pickle
import random

'''
Function: parse_data
--------------------
This function randomly splits data into test and train sets
based on probability r of being in the test set.
'''
def parse_data(data, r=0.95):
    train, test = [], []
    for i in range(len(data)):
        val = random.uniform(0, 1)
        if val< r:
            test.append(data[i])
        else:
            train.append(data[i])
    return train, test

'''
Function: transform
-------------------
This function takes as input data in the form returned by
the pickling operation. Comp_method takes two inputs (or two
pca components) and outputs a single np array. This function
transforms the data points into a single array like object.
'''
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

'''
Function: get_images_pipeline
-----------------------------
Filters images in the provided file, then processes them per specified 
version:
1) DRT and PCA processing,
2) FFT DCT and greatest frequency coefficients from DCT,
3) Block DCT and largest coefficient from each block.
'''
def get_images_pipeline(in_file, version=3):
    print "Filtering images ..."
    images = image_filter.filter_dir(in_file)
    image_components = []
    
    print "Getting image components version", version, "..."
    for img in images:
        components = 0.
		
        if (version == 1): # use DRT, PCA
            sinogram = invariant_drt.compute_drt(img)
            new_sinogram = invariant_drt.post_process_sinogram(sinogram)
            components = pca.get_pca(new_sinogram)
        elif (version==2): # use DCT FFT method
            image_dct = dct.fft_dct_2d(img)
            components = dct.get_largest_freqs(image_dct)
        else: # use DCT block method
            image_dct = dct.block_dct(img)
            components = dct.get_block_comps(image_dct)
		
	image_components.append(components)
    return image_components
	
'''
Function: pickle_data
---------------------
This function takes in the path to training data and a file name to
save pickled data to. If the outfile exists, it loads the data
and returns it. If not, it loops through the provided test user's image
set. Each image is processed according to pipeline specifications.
This creates a dictionary of training instances, which are pickled (saved)
the outfile. The path should point to a directory of the following form:
path - 
    ../userA/
        Genuine/
            *** all genuine signatures ***
        Forgeries/
            *** all forgeries ***
	
Data output is all combinations of genuine and forged signatures
(either genuine-genuine pairs or genuine-forged pairs). Each
datapoint is of the form:
    {'1'     : components of sig 1
     '2'     : components of sig 2
     'label' : 0 if sig 2 forged, 100 if genuine (representing persentages)
    }
'''
def pickle_data(path, outfile):
    # Stored pickle data, return without processing new data
    if os.path.isfile(outfile):
        return pickle.load(open(outfile, 'r'))
    
    data = []
    genuine = []
    fakes = []
    # Process images
    for dir in os.listdir(path):
        if dir[0] == ".": # skip parent path
            continue
		
        print "Current = ", dir
        if dir == "Genuine":
            print "Processing Genuine Signatures ..."
            genuine = get_images_pipeline(path + dir + "/")
        elif dir == "Forgeries":
            print "Processing Forged Signatures ..."
            fakes = get_images_pipeline(path + dir + "/")
	
        # Build dataset for analysis
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
    
    # Save data for later use and return
    pickle.dump(data, open(outfile, 'w+'))
    return pickle.load(open(outfile, 'r'))
	
'''
Function: get_datasets
----------------------
Gets data from the provided paths (processing if no pickled data), and randomly 
generates the training, development and test data sets (which are returned).
'''
def get_datasets(data_path, pickle_path, shouldPrint=False):
    data = pickle_data(data_path, pickle_path)
    train, test = parse_data(data)
    train, dev = parse_data(data, r=.9)
    dev = transform(dev)
    train = transform(train)
    test = transform(test)
    
    if (shouldPrint):
        print "Data set sizes:"
        print "Training = ", len(train[0]), ", Dev = ", len(dev[0]), ", Test = ", len(test[0])
    
    return train, dev, test

'''
Function: score_datasets
------------------------
Uses SVM to score provided data sets, returning (and reporting is desired)
the results. 

Precision = (true pos) / (total pos), i.e. correct accecptance rate or
inversely, 1 - FAR (false acceptance rate).
Recall = (true pos) / (true pos + false neg), i.e. correct recongition rate or
inversely, 1 - FRR (false recognition rate).

Note: dataset[0] is components, dataset[1] is solutions.
'''
def score_datasets(train, dev, test, shouldPrint=False):
    clf = LinearSVC(C=1., tol=1e-6, max_iter=10000, fit_intercept=True, loss='hinge')
    clf.fit(train[0], train[1])
	
    # Score to return
    score_train = clf.score(train[0], train[1])
    score_dev = clf.score(dev[0], dev[1])
    score_test = clf.score(test[0], test[1])
	
    # Precision and recall rates
    predicted = clf.predict(test[0])
    precision, recall = precision_recall_fscore_support(test[1], predicted)[:2]
	
    if (shouldPrint):
        print "Train Score is: " + str(score_train)
        print "Dev Score is: " + str(score_dev)
        print "Test Score is: " + str(score_test)
		
        # Print precision, recall stats
        target_names = ['Genuine', 'Forgeries']
        print(classification_report(test[1], predicted, target_names=target_names))

    # Return scores for analysis
    return np.array([score_train, score_dev, score_test]), np.array(precision), np.array(recall)

'''
Function: run_test
------------------
Gets data from:
1. provided dataset and processes it before saving/returning pickled file, or
2. pickled data file of already processed data.
Then performs SVM data fitting on randomized training and test sets, printing
average results over provided number of runs. 
'''
def run_tests(data_path, pickle_path, num_runs=20):
    total_sizes = np.zeros(3) # test, dev, train
    total_scores = np.zeros(3) # (train, dev, test) scores
    precision = recall = np.zeros(2) # test prediction from train fit stats
    
    print "Running tests for current subject ..."
    for i in range(num_runs):
        train, dev, test = get_datasets(data_path, pickle_path)
        total_sizes += np.array([len(train[0]), len(dev[0]), len(test[0])])
        scores, p, r = score_datasets(train, dev, test)
        total_scores += scores
        precision = np.add(precision, p.astype(float))
        recall += r
    total_sizes /= num_runs
    total_scores /= num_runs
    precision /= num_runs
    recall /= num_runs
    
    print "Number of runs = ", num_runs
    print "Average data set sizes:"
    print "Training = ", total_sizes[0], ", Dev = ", total_sizes[1], ", Test = ", total_sizes[2]
    print "Average scores: (1 is best for all below)"
    print "Training = ", total_scores[0], ", Dev = ", total_scores[1], ", Test = ", total_scores[2] 
    print "Precision: Genuine = ", precision[0], ", Forged = ", precision[1]
    print "Recall: Genuine = ", recall[0], ", Forged = ", recall[1]
