'''
File:        Main
Date:        03/16/18
Authors:     Robert Neff, Nathan Butler
Description: Runs singature-verification components.
'''

# TODO: build pickle sets, compare results, get images for presentation and report

import utils
import os

'''
Function: main
--------------
Runs the project.
'''
def main():
    print "Starting!"
	
    m = 1
    
    '''
    data_path = "../../Dataset_4NSigComp2010/TrainingData/"
    pickle_path = "../../Data/pickle_"
    for user_dir in os.listdir(data_path):
        if user_dir[0] == ".": # skip parent path
            continue
		
        print "User = ", user_dir
        utils.run_tests(data_path + user_dir + "/", pickle_path + user_dir + "_v" + str(m), num_runs=20, method=m)
    
	data_path = "../../signDist1.0/images1/"
    for user_dir in os.listdir(data_path):
        if user_dir[0] == ".": # skip parent path
            continue

        print "User = ", user_dir
        utils.run_tests(data_path + user_dir + "/",  pickle_path + user_dir + "_v" + str(m), method=m, num_runs=1)
    '''
	
    data_path = "../../signDist1.0/images1/subject_025/"
    pickle_path = "../../Data/test"
	
    utils.run_tests(data_path, pickle_path, num_runs=1, method=m)
	
    print "Done!"
	
if __name__ == '__main__':
    main()
