'''
File:        Main
Date:        03/16/18
Authors:     Robert Neff, Nathan Butler
Description: Runs singature-verification components.
'''

# TODO: loop over users and pickle data, print results (build master database)

import utils
import os

'''
Function: main
--------------
Runs the project.
'''
def main():
    print "Starting!"
	
    data_path = "../../Dataset_4NSigComp2010/TrainingData/"
    pickle_path = "../../Data/pickle_data_"
   
    for user_dir in os.listdir(data_path):
        if user_dir[0] == ".": # skip parent path
            continue
		
        print "User = ", user_dir
        utils.run_tests(data_path + user_dir + "/", pickle_path + user_dir)
    '''
    data_path = "../../signDist1.0/images1/"
    for user_dir in os.listdir(data_path):
        if user_dir[o] == ".": # skip parent path
            continue
        
        print "User = ", user_dir"
        utils.run_tests(data_path + user_dir + "/", pickle_path + user_dir)
    '''
    print "Done!"
	
if __name__ == '__main__':
    main()
