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

'''
Function: main
--------------
Runs the project.
'''
def main():
    # Filter the images
    images = image_filter.filter_dir("../test_images/", "../output/")
    
	# Compute DRT 
    sinogram = invariant_drt.compute_drt(images[0], np.arange(180))
    misc.imsave("../output/sinogram_base_img.png", sinogram)
    
	# Make DRT shift, scale and rotationally invariant
    new_sinogram = invariant_drt.post_process_sinogram(sinogram, 400)
    misc.imsave("../output/sinogram_processed_img.png", new_sinogram)
    
	# Take PCA per paper definition
    components = pca.get_pca(new_sinogram)
    print components.shape
	
	# Perform PCA using sklearn
    data = misc.imread("../output/sinogram_processed_img.png")
	
    # Reconstruct the image from components
    out = pca.sklearn_pca(data, 10)
    misc.imsave("../output/singoram_processed_pca.png", out)
    
if __name__ == '__main__':
    main()