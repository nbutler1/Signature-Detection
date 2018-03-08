import image_filter
import invariant_drt
import pca
import pickle
import os

"""
Function: get_images_pipeline
-----------------------------
Given a file that contains multiple images, filters the images,
runs drt on them, then pca, then returns the images components.
"""
def get_images_pipeline(in_file):
    # Filter images
    images = image_filter.filter_dir(in_file, "", False)
    image_components = []
    for img in images:
        # Compute DRT and then PCA
        sinogram = invariant_drt.compute_drt(img, np.arange(180))
        new_sinogram = invariant_drt.post_process_sinogram(sinogram, 400)
        components = pca.get_pca(new_sinogram)
        image_components.append(components)
    return image_components

"""
Function: pickle_data
---------------------
This function takes in the path to training data and a file to
save pickled data to.  It first filters each image using DRT
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
Data output is the combination of all genuine and forged signatures
(either genuine-genuine pairs or genuine-forged pairs).  Each
datapoint is of the form:
    {'1'     : pca of sig 1
     '2'     : pca of sig 2
     'label' : [0, 100] if sig 2 forged, [100, 0] if genuine
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
                    genuine = get_images_pipeline(path + f + "/" + filename + "/")
                elif filename == "Simulated":
                    fakes = get_images_pipeline(path + f + "/" + filename + "/")
            for i in range(len(genuine)):
                comp1 = genuine[i]
                label = [100.0, 0.0]
                for j in range(len(genuine)):
                    comp2 = genuine[j]
                    data.append({'1': comp1, '2': comp2, 'label': label})
                label = [0.0, 100.0]
                for j in range(len(fakes)):
                    comp2 = fakes[j]
                    data.append({'1': comp1, '2': comp2, 'label': label})
    pickle.dump(data, open(outfile, 'w+'))
    return pickle.load(open(outfile, 'r'))
