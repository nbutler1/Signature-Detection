'''
File:        Main
Date:        02/21/18
Authors:     Robert Neff, Nathan Butler
Description: Runs singature-verification components.

Useful sources:
Paper 1:
https://ac.els-cdn.com/S1568494615007577/1-s2.0-S1568494615007577-main.pdf?_tid=046b6bdc-0e3c-11e8-89cf-00000aacb35e&acdnat=1518251337_bef3cd4d78db771b353595816db8682f
Paper 2:
https://link.springer.com/content/pdf/10.1155%2FS1110865704309042.pdf
Paper 3:
http://www.ijirst.org/articles/IJIRSTV3I1015.pdf
'''

import numpy as np
from scipy import misc
import image_filter
import invariant_drt
import pca
import os
import pickle
import random
import matplotlib.pyplot as plt
from comp_net import ComparisonNet
from utils import pickle_data, parse_data, transform
from sklearn.svm import LinearSVC

'''
Function: main
--------------
Runs the project.
'''
def main():
    outfile = "../../Data/pickle_data"
    path = "../../Dataset_4NSigComp2010/TrainingData/"
    data = pickle_data(path, outfile)
    train, test = parse_data(data)
    train, dev = parse_data(data, r=.1)
    dev = transform(dev)
    train = transform(train)
    test = transform(test)
    print "Training SVM"
    clf = LinearSVC(C=1, tol=1e-6, max_iter=10000, fit_intercept=True, loss='hinge')
    clf.fit(train[0], train[1])
    score = clf.score(train[0], train[1])
    print "train Score is: " + str(score)
    score = clf.score(dev[0], dev[1])
    print "Dev Score is: " + str(score)
    score = clf.score(test[0], test[1])
    print "Test Score is: " + str(score)
    """
    (P, n) = data[0]['1'].shape
    dev = transform(dev)
    comp_model = ComparisonNet(P, n, 250, 125)
    comp_model.fit(train, dev)
    """

if __name__ == '__main__':
    main()
