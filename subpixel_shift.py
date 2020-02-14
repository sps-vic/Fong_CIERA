#!/usr/bin/env python

"Python script showing how to perform subpixel shifts of data in python."
"Author: Kerry Paterson"

import numpy as np
import matplotlib.pyplot as plt

def convertfreqs(x): #function to convert x to frequency
    return np.mod((x + N/2),N) - N/2

def fft(x): #function to perform real part of fft
    return np.fft.fft2(x)

def ifft(x): #function to perform imaginary part of fft
    return np.fft.ifft2(x) 

N = 30 #N needs to larger than the size of the and the shift you want to apply

xshift = 2.5 #x shift in pixels
yshift = 1.5 #y shift in pixels

xi, yi = np.meshgrid(np.linspace(-1,1,25), np.linspace(-1,1,25)) #create a grid for sample data
d = np.sqrt(xi**2+yi**2) #calculate distance on grid
data = np.exp(-((d)**2/(2.0*0.5**2))) #create 2D guassian for example

f = np.zeros((N,N)) #create a new array with the new size for the shifted data
f[:data.shape[0],:data.shape[1]] = data[:,:] #add the data to the new array

x = np.tile(np.arange(N),(N,1)).T #create x for shifted data
x = convertfreqs(x) #convert x to frequencies
y = x.T #create y for shifted data

shifted = ifft(fft(f)*np.exp(1j*2*np.pi*(-xshift*x/N-yshift*y/N))) #calculate fft of the data with the shift
shifted = np.abs(shifted) #use only the real part

plt.figure() #plot to show the original data and the shifted data
plt.subplot(121)
plt.imshow(f, interpolation='none')
plt.title('Original data')
plt.subplot(122)
plt.imshow(shifted, interpolation='none')
plt.title('Shifted data')
plt.show()
