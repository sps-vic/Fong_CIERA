#!/usr/bin/env python

"Python script showing how to perform and measure subpixel shifts of data in python."
"Author: Kerry Paterson"

import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.feature import register_translation

N = 30 #N needs to larger than the size of the and the shift you want to apply

xshift = 2.5 #x shift in pixels
yshift = 1.5 #y shift in pixels

xi, yi = np.meshgrid(np.linspace(-1,1,25), np.linspace(-1,1,25)) #create a grid for sample data
d = np.sqrt(xi**2+yi**2) #calculate distance on grid
data = np.exp(-((d)**2/(2.0*0.5**2))) #create 2D guassian for example

f = np.zeros((N,N)) #create a new array with the new size for the shifted data
f[:data.shape[0],:data.shape[1]] = data[:,:] #add the data to the new array

shifted = np.fft.ifft2(scipy.ndimage.fourier_shift(np.fft.fft2(f),(2.5,1.5))).real #shift data using FFTs

plt.figure() #plot to show the original data and the shifted data
plt.subplot(121)
plt.imshow(f, interpolation='none')
plt.title('Original data')
plt.subplot(122)
plt.imshow(shifted, interpolation='none')
plt.title('Shifted data')
plt.show()

#how to measure the shift is to use image registration:
n = np.random.normal(0,0.1,900).reshape(30,30) #create some noise
f = f+n #add the noise to the original data to simulate real data

n = np.random.normal(0,0.1,900).reshape(30,30) #create some noise
shifted = shifted+n #add the noise to the shifted data to make it slightly different to the original

shift, error, diffphase = register_translation(shifted,f,100) #the 100 gives subpixel measurements

print('The measured shift between data is x = '+str(shift[0])+' and y = '+str(shift[1])) #without the noise the measured shifted = the actual shift
