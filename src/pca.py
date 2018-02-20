from scipy import misc
from sklearn.decomposition import PCA

import numpy as np


def takePCA(data, comps):
    pca = PCA(n_components=comps)
    pca.fit(data)
    components = pca.transform(data)
    filtered = pca.inverse_transform(components)
    return filtered

def main():
    # first read in files
    data = misc.imread("../output/sinogram_processed_img.png")
    # then output files
    out = takePCA(data, 10)
    misc.imsave("../output/singoram_processed_pca.png", out)
    # take pca

if __name__ == "__main__":
    main()
