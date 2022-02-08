from dmd_patterns import generate_patterns
from image_reconstruct import reconstruct

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray, rgba2rgb
from scipy.linalg import hadamard

#reconstruction size
px = 32

#load image
file = rgb2gray(rgba2rgb(plt.imread('originals/mario.png')))
image = resize(file, (px, px))

#Hadamrd basis for a DMD
patterns = generate_patterns(px)


meas = np.empty(2*px**2)
#simulate DMD measurement
for i in range(2*px**2):
    
    #measurement
    meas[i] = np.sum(np.multiply(patterns[:,:,i],image))

# post processing
diff_meas = meas[::2] - meas[1::2]

# Hadamard matrix
hadamard_mat = hadamard(px**2)
       
reconstruct((px, px), diff_meas, hadamard_mat)

