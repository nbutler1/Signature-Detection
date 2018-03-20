'''
File:        Main
Date:        03/16/18
Authors:     Robert Neff, Nathan Butler
Description: Runs singature-verification components.
'''

# TODO: build pickle sets, compare results, get results plots

import utils
import os

'''
Function: main
--------------
Runs the project.
'''
def main():
    print "Starting!"
	
    m = 2 # method to use
    
    # Database tests
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
    
    # Test individual subject
    data_path = "../../signDist1.0/images2/subject_002/"
    pickle_path = "../../Data/test"
    utils.run_tests(data_path, pickle_path, num_runs=10, method=m)
	
    print "Done!"
	
if __name__ == '__main__':
    main()
