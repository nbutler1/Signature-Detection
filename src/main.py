'''
File:        Main
Date:        03/14/18
Authors:     Robert Neff, Nathan Butler
Description: Runs singature-verification components.
'''

import numpy as np
from utils import pickle_data, parse_data, transform
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.metrics import classification_report, precision_recall_fscore_support

'''
Function: get_datasets
----------------------
Gets the transformed data sets for the provided paths.
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
Uses SVM to score the provided sets. 

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
    # Precision is ability of clf to nor label + a sample that is -
    # Recall is ability of clf to find all the + samples
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
Function: main
--------------
Runs the project.
'''
def main():
    total_sizes = np.zeros(3) # test, dev, train
    total_scores = np.zeros(3) # (train, dev, test) scores
    precision = recall = np.zeros(2) # test prediction from train fit stats
    num_runs = 1
    print "Running tests ..."
    for i in range(num_runs):
        train, dev, test = get_datasets("../../Dataset_4NSigComp2010/TrainingData/", 
            "../../Data/pickle_data")
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
	
if __name__ == '__main__':
    main()
