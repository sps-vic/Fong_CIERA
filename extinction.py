import numpy as np
import random

def uv_extinction(w):
    Rv=3.1
    x=1/w 
    Fax=-0.04473*(x-5.9)**2-0.009779*(x-5.9)**3
    Fbx=0.2130*(x-5.9)**2+0.1207*(x-5.9)**3
    A = []
    if x<=1/5.9:
        a=1.752-0.316*x-0.104/((x-4.67)**2+0.341)+Fax
        b=-3.090+1.825*x+1.206/((x-4.62)**2+0.263)+Fbx
    else:
        a=1.752-0.316*x-0.104/((x-4.67)**2+0.341)
        b=-3.090+1.825*x+1.206/((x-4.62)**2+0.263)
    return a+b/Rv

def opt_extinction(w):
    Rv=3.1
    y=1/w-1.82
    a=1+0.17699*y-0.50447*y**2-0.02427*y**3+0.72085*y**4+0.01979*y**5-0.77530*y**6+0.32999*y**7
    b=1.41338*y+2.28305*y**2+1.07233*y**3-5.38434*y**4-0.62251*y**5+5.30260*y**6-2.09002*y**7
    return a+b/Rv

def nir_extinction(w):
    Rv=3.1
    x=1/w
    a=0.574*x**1.61
    b=-0.527*x**1.61
    return a+b/Rv

AV = 0.096
z = 0.5
wvs = np.linspace(4000,15000,100)
fluxes = np.random.normal(10,1,100)
fluxes_corr = []
wv_rest = wvs/1e4/(1+z) #the functions need wavelength in microns and it must be rest wavelength
for i in range(len(wv_rest)):
    if wv_rest[i]<=0.303:
        A = uv_extinction(wv_rest[i])
    elif wv_rest[i]>0.303 and wv_rest[i]<=0.91:
        A = opt_extinction(wv_rest[i])
    else:
        A = nir_extinction(wv_rest[i])
    mag = -2.5*np.log10(fluxes[i])
    magcorr = mag - A*AV
    fcorr = 10**(-0.4*magcorr)
    fluxes_corr.append(fcorr)