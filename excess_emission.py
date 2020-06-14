##import packages
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt 
import math as m
import random
import csv
import pandas as pd
import os, os.path
import emcee
import corner

##set options for fitting
#################################
start = 100 #default is 0, otherwise enter an integer 
end = True #default is true, otherwise enter an integer
logCenter = True #probably should always stay True to log center data
go = True #only used to skip bursts, set to false and uncomment line below setting it to true if you want to skip bursts

#Note: for macs you might have to change format of directories or use os.path.join(dir1, dir2, ..., dirx)
#straightforward examples: double = 140930B, single = 080905A, triple = 070809
GRBName = 'GRB140930B' #Enter name of GRB as 'GRB######(X)'
GRBBaseDir = 'C:/Users/Administrator/Box/Research Stuff/XRT_Data/' #Enter base directory
GRBDir = 'lightcurves' #If applicable, enter directory where lightcurve files stored
samples_dir = 'samples/' #Directory to save .npy samples files
directory = GRBBaseDir + GRBDir #Sets full name of directory to get lightcurves
file_type = '.txt' #Specify lightcurve file type as string (should be .txt)
nameBool = True #True if lightcurve files have a name row first
saveFig = True #True if figures should all be saved
saveParam = True #True if parameters should all be saved
params_file = 'fit_params.csv' #file name to save parameters to (will overwrite if it exists)
overwrite = True #True if params should be saved even if burst already exists in params file (currently not implemented)
folder = True #True if you want to loop through an entire folder of lightcurves and fit all of them in succession

function = 'powerLaw' #If using curve fit, what function to use
model_type = 'double' #If using emcee, what type to fit as string (single, double, triple)

fit_type = 'emcee' #curve_fit or emcee

#set save location for all plots
if(fit_type == 'emcee'):
	GRBSaveDir = 'GRB_chi2_fits/emcee'
else:
	GRBSaveDir = 'GRB_chi2_fits'

nwalkers = 100 #number of walkers, I use 100
niter = 15000 #number of iterations, I use 15000

#min and max values used to set initial parameters in the emcee walkers
alpha_min = -3
alpha_max = 0
amp_min = -15 #amp is log scale so [-15, -4] is usually good
amp_max = -4

discard = 500 #number of iterations to discard 
delta = 0.1 #broken power law smoothing parameter


legendLoc = 'upper right' #location of legend (currently not implemented)
figsize = (8,6) #figure size
p0_cf = (1e9,0) #default initial parameters if using curve_fit (ignore)
column = 'time' #default column to get (ignore)
#################################
#curve fit models?
#simple power law
def powerLaw(x, amplitude, alpha):
	return amplitude * x ** alpha

#piecewise broken power law
def brokenPowerLaw(x, amplitude, x_break, alpha_1, alpha_2):
	alpha = np.where(x < x_break, alpha_1, alpha_2)
	xx = x / x_break
	return amplitude * xx ** (alpha)

#smoothly broken power law
def smbpl(x, amplitude, x_break, alpha1, alpha2):
    return amplitude * ((x / x_break) ** (-1 * alpha1)) * ((0.5 * (1 + ((x / x_break) ** (1 / delta)))) ** ((alpha1 - alpha2) * delta))

#triple power law
def tbpl_pw(x, amp, xb1, xb2, a1, a2, a3):
    y=[]
    x1 = x[np.where(x <= xb1)]
    x2 = x[np.where((x > xb1) & (x <= xb2))]
    x3 = x[np.where(x > xb2)]
    
    model1 = amp * x1 ** a1
    
    amp2 = (amp * xb1 ** (a1 - a2))
    model2 = amp2 * x2 ** a2
    
    amp3 = (amp2 * xb2 ** (a2 - a3))
    model3 = amp3 * x3 ** a3
    y = np.concatenate((model1, model2, model3), axis=0)
    y = y[0:len(x)]
    return y

#model functions for emcee (amplitudes and break times are log variables)
def model(theta, xs):
	if(model_type == 'single'):
		amp, alpha = theta
		model = 10 ** amp * xs ** alpha
	elif(model_type == 'double'):
		amp, alpha1, alpha2, x_break = theta
		model = (10 ** amp) * ((xs / (10 ** x_break)) ** (alpha1)) * ((0.5 * (1 + ((xs /(10 ** x_break)) ** (1 / delta)))) ** ((alpha2 - alpha1) * delta))
	elif(model_type== 'triple'):
		amp, alpha1, alpha2, alpha3, x_break1, x_break2 = theta
		model = []
		x1 = xs[np.where(xs <= 10 ** x_break1)]
		x2 = xs[np.where((xs > 10 ** x_break1) & (xs <= 10 ** x_break2))]
		x3 = xs[np.where(xs > 10 ** x_break2)]

		model1 = 10 ** amp * x1 ** alpha1
		
		amp2 = ((10 ** amp) * (10 ** x_break1) ** (alpha1 - alpha2))
		model2 = amp2 * x2 ** alpha2
		
		amp3 = (amp2 *  (10 ** x_break2) ** (alpha2 - alpha3))
		model3 = amp3 * x3 ** alpha3
		model = np.concatenate((model1, model2, model3), axis=0)
		model = model[0:len(xs)]
	return model

#shifts data points so that they are log centered
def logCenterTime(dataPoint, plus, minus):
	a = dataPoint - np.absolute(minus)
	b = dataPoint + plus
	delta = ((np.log10(b) - np.log10(a)) / 2) + np.log10(a)
	minus = (10 ** delta - a)
	plus = b - (10 ** delta)
	return [(10 ** delta), minus, plus]

#reads data given a location
def readData(name, directory, fileType='.txt', nameBool=True, logCenter = True):
	nameExt = name + fileType
	fileName = os.path.join(directory, nameExt)
	data = np.genfromtxt(fileName, names = nameBool)
	index = 0
	time = data['duration']
	if(logCenter and (type(time) == list)):
		for t in data['duration']:
			plus = data['time_plus'][index]
			minus = data['time_minus'][index]
			LCData = logCenterTime(t, plus, minus)
			data['duration'][index] = LCData[0]
			data['time_minus'][index] = LCData[1]
			data['time_plus'][index] = LCData[2]
			index += 1
	return data

#selects column of data for manipulation
def selectData(name, directory, fileType = '.txt', nameBool = True, column = 'time', logCenter = False):
	data = readData(name, directory, fileType, nameBool, logCenter = logCenter)		
	if column == 'time':
		return data['duration']
	elif column == 'flux':
		return data['flux']
	elif column == 'timeErr':
		timeErrPlus = (data['time_plus'])
		timeErrMinus = (np.abs(data['time_minus']))
		timeErr = [timeErrMinus, timeErrPlus]
		return timeErr 
	elif column == 'fluxErr':
		errFactor = 68 / 90
		fluxErrPlus = (data['flux_plus'] * errFactor)
		fluxErrMinus = (data['flux_minus'] * errFactor)
		fluxErr = ((fluxErrPlus + np.abs(fluxErrMinus)) / 2)
		return fluxErr

#gets index of data between start and end
def getInd(data, start=0, end=True):
	time = data['duration']
	if end:
		end = m.ceil(time[len(time) - 1])
	ind = np.ravel(np.argwhere(np.logical_and(time > start, time < end)))
	return ind

#clips data using index
#acceptable columns are 'time','timeErr','flux','fluxErr'
def clipData(data, start=0, end=True, column = 'time'):
	
	time = data['duration']

	flux = data['flux']

	errFactor = (68 / 90)

	ind  = getInd(data, start, end)

	#average the time errors
	timeErrPlus = (data['time_plus'])
	timeErrMinus = (np.abs(data['time_minus']))
	timeErr = ((timeErrPlus + np.abs(timeErrMinus)) / 2)

	#average flux errors
	fluxErrPlus = (data['flux_plus'] * errFactor)
	fluxErrMinus = (data['flux_minus'] * errFactor)
	fluxErr = ((fluxErrPlus + np.abs(fluxErrMinus)) / 2)

	#reduce data to only include points between start and end time
	if column=='time':
		timeRed = time[ind]
		return timeRed
	elif column =='timeErr':	
		timeErrPlusRed = timeErrPlus[ind]
		timeErrMinusRed = timeErrMinus[ind]
		timeErrRed = [timeErrMinusRed, timeErrPlusRed]
		return timeErrRed
	elif column == 'flux':	
		fluxRed = flux[ind]
		return fluxRed
	elif column == 'fluxErr':	
		fluxErrPlusRed = fluxErrPlus[ind]
		fluxErrMinusRed = fluxErrMinus[ind]
		fluxErrRed = ((fluxErrPlusRed + np.abs(fluxErrMinusRed)) / 2)
		return fluxErrRed

#gets all data necessary for emcee
def getAllData(data, start = 0, end = True):
	time = clipData(data, start = start, end = end, column = 'time')
	timeErr = clipData(data, start = start, end = end, column = 'timeErr')
	flux = clipData(data, start = start, end = end, column = 'flux')
	fluxErr = clipData(data, start = start, end = end, column = 'fluxErr')
	return time, flux, fluxErr, timeErr

#log likelihood
def lnlike(theta, x, y, yerr):
	LnLike = -0.5 * np.sum(((y - model(theta, x)) ** 2) / (yerr ** 2))
	return LnLike

#define priors (the values in the conditionals can be changed based on your assumptions of what the parameters should be)
#keep return lines as is
def lnprior(theta):
	if(model_type == 'single'):
		amp, alpha = theta
		if -5 < alpha < 3 and -15 < amp < -4:
			return 0.0
		else:
			return -np.inf
	elif(model_type == 'double'):
		amp, alpha1, alpha2, x_break = theta
		if -3 < alpha1 < 3 and -15 < amp < -5 and -3 < alpha2 < 3 and np.log10(time[0]) < x_break < np.log10(time[len(time)-1]):
			return 0.0
		else:
			return -np.inf
	elif(model_type == 'triple'):
		amp, alpha1, alpha2, alpha3, x_break1, x_break2 = theta
		if -15 < amp < -5 and -3 < alpha1 < 0 and -2 < alpha2 < 3 and -3 < alpha3 < 0 and np.log10(time[0]) < x_break1 < np.log10(time[len(time)-1]) and np.log10(time[1]) < x_break2 < np.log10(time[len(time) - 1]):
			return 0.0
		else:
			return -np.inf

#log probability, combines log prior and log likelihood to be used in emcee
def lnprob(theta, x, y, yerr):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y, yerr)

#returns initial parameters to be used for the walkers and overwrites default parameters
def set_init_params(time, nwalkers = 100, niter = 10000, alpha_min = -3, alpha_max = 0, amp_min = -15, amp_max = -5):
	p0 = []

	x_break_min = np.log10(time[0])
	x_break_max = np.log10(time[len(time) - 1])

	if(model_type == 'single'):
		initial = np.array([0, -1])
		ndim = len(initial)
		for i in range(nwalkers):
			p00 = np.array(initial)[0] + ((amp_max - amp_min) * np.random.random_sample() + amp_min)
			p01 = np.array(initial)[1] + ((alpha_max - alpha_min) * np.random.random_sample() + (alpha_min))
			p0.append((p00, p01))
	elif(model_type == 'double'):
		initial = np.array([0, 0, 0, 0])
		ndim = len(initial)
		for i in range(nwalkers):
			p00 = np.array(initial)[0] + ((amp_max - amp_min) * np.random.random_sample() + amp_min)
			p01 = np.array(initial)[1] + ((alpha_max - alpha_min) * np.random.random_sample() + (alpha_min))
			p02 = np.array(initial)[2] + ((alpha_max - alpha_min) * np.random.random_sample() + (alpha_min))
			p03 = np.array(initial)[3] + ((x_break_max - x_break_min) * np.random.random_sample() + (x_break_min))
			p0.append((p00, p01, p02, p03))
	elif(model_type == 'triple'):
		initial = np.array([0, 0, 0, 0, 0, 0])
		ndim = len(initial)
		for i in range(nwalkers):
			p00 = np.array(initial)[0] + ((amp_max - amp_min) * np.random.random_sample() + amp_min)
			p01 = np.array(initial)[1] + ((alpha_max - alpha_min) * np.random.random_sample() + (alpha_min))
			p02 = np.array(initial)[2] + ((alpha_max - alpha_min) * np.random.random_sample() + (alpha_min))
			p03 = np.array(initial)[3] + ((alpha_max - alpha_min) * np.random.random_sample() + (alpha_min))
			p04 = np.array(initial)[4] + ((x_break_max - x_break_min) * np.random.random_sample() + (x_break_min))
			p05 = np.array(initial)[5] + ((x_break_max - x_break_min) * np.random.random_sample() + (x_break_min))
			p0.append((p00, p01, p02, p03, p04, p05))
	return p0, nwalkers, niter, ndim

#runs the EnsembleSampler and returns the sampler (I don't know what pos, prob, and state are)
def main(p0,nwalkers,niter,ndim,lnprob):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(time, flux, fluxErr))

    print("Running burn-in...")
    p00, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p00, niter, progress=True)
    return sampler, pos, prob, state

#runs the fit and returns data based on what type of fitting we're using (emcee of curve_fit)
def fit(function = function, fit_type = 'emcee', start=0, end=True, p0_cf=(1e9, -1)):
	if(fit_type == 'curve_fit'):
		popt, pcov = curve_fit(function, time, flux, p0_cf, sigma=fluxErr)
		return popt
	elif(fit_type == 'emcee'):
		p0, nwalkers0, niter0, ndim = set_init_params(time, nwalkers = nwalkers, niter = niter, alpha_min = alpha_min,
			alpha_max = alpha_max, amp_min = amp_min, amp_max = amp_max)
		print('hello')
		sampler, pos, prob, state = main(p0 = p0, nwalkers = nwalkers, niter = niter, ndim = ndim, lnprob = lnprob)
		return sampler, pos, prob, state, ndim

#calculate and return a reduced chi squared value for curve_fit
def calcChiSq(GRBName='GRB070429B', GRBBaseDir='C:/Users/Administrator/Box/Research Stuff/XRT_Data/', GRBDir = 'control',
 fileType = '.txt', nameBool = True, start = 0, end = True, column = 'time', p0=(1e9,-1), function = powerLaw):
	data = readData(GRBName, (GRBBaseDir + GRBDir) , nameBool = nameBool)
	modelData = createModel(GRBName = GRBName, GRBBaseDir = GRBBaseDir, GRBDir = GRBDir, fileType = fileType, 
		nameBool = nameBool, start = start, end = end, column = column, p0 = p0, function = function, t='data', fit_type = fit_type, model_type = model_type)
	t = modelData[0]
	fluxExp = modelData[1] #fluxExp
	fluxRed = clipData(data, start = start, end = end, column = 'flux') # fluxRed
	fluxErrRed = clipData(data, start = start, end = end, column = 'fluxErr')
	print(fluxErrRed)
	chisq = sum(((fluxRed - fluxExp) ** 2) / (fluxErrRed ** 2))
	# if(model_type == 'single'):
	# 	fp = 2
	# elif(model_type == 'double'):
	# 	fp = 4
	# else:
	# 	fp = 6
	dchisq = chisq / (len(fluxRed) - 1)# - fp)
	return dchisq

#returns model. curve_fit will return the model itself, emcee will returns medians for each parameters, along with uncertainties and chisq
def createModel(GRBName='GRB070429B', GRBBaseDir='C:/Users/Administrator/Box/Research Stuff/XRT_Data/', GRBDir = 'control',
 fileType = '.txt', nameBool = True, start = 0, end = True, column = 'time', p0=(1e9,-1), function = powerLaw, t = 'model', 
 fit_type = 'emcee', model_type = 'single'):
	dir1 = GRBBaseDir + GRBDir + '/'
	time = selectData(GRBName, dir1, fileType, nameBool, 'time')
	if(fit_type == 'curve_fit'):
		dir1 = GRBBaseDir + GRBDir + '/'
		popt = fit(function, data, start, end, p0)

		if t == "model":
			t = np.logspace(np.log10(time[0] / 2), np.log10(time[len(time)-1] * 2))
		elif t == 'data':
			t = clipData(data = data, start = start, end = end, column = 'time')

		if len(popt) == 2:
			model = powerLaw(t, popt[0], popt[1])
		elif len(popt) == 4:
			model = brokenPowerLaw(t, popt[0], popt[1], popt[2], popt[3])
		elif len(popt) == 6:
			model = tbpl_pw(t, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
		return (t, model)
	elif(fit_type == 'emcee'):
		sampler, pos, prob, state, ndim = fit(start = start, end = end, fit_type = fit_type)
		samples = sampler.get_chain(flat=True, discard = 0)
		# print(samp.shape)
		# samples = sampler.flatchain
		# print(samples.shape)
		theta_max = samples[np.argmax(sampler.flatlnprobability)]
		log_likelihood = np.max(sampler.flatlnprobability)
		if(model_type == 'single'):
			fp = 2
		elif(model_type == 'double'):
			fp = 4
		else:
			fp = 6
		chisq_red = (-2 * log_likelihood) / (len(time) - fp)
		medians = []
		plus_uncertainties = []
		minus_uncertainties = []
		for i in range(ndim):
			mcmc = np.percentile(samples[:, i], [16, 50, 84])
			medians.append(mcmc[1])
			plus_uncertainties.append(mcmc[2] - mcmc[1])
			minus_uncertainties.append(mcmc[0] - mcmc[1])
		return medians, sampler, pos, prob, state, ndim, chisq_red, plus_uncertainties, minus_uncertainties

#plot curve_fit
def plot_cf(GRBName='GRB070429B', GRBBaseDir='C:/Users/Administrator/Box/Research Stuff/XRT_Data/', GRBDir = 'control',
 fileType = '.txt', nameBool = True, start = 0, end = True, column = 'time', p0=(1e9,-1), function = powerLaw, 
 legendLoc = 'upper right', figsize = (8,6)):
	directory = GRBBaseDir + GRBDir
	modelData = createModel(GRBName, GRBBaseDir, GRBDir, fileType, nameBool, start, end, column, p0, function)
	data = readData(GRBName, directory, nameBool = nameBool)
	dataRaw = readData(GRBName, directory, nameBool = nameBool, logCenter = False)
	t = modelData[0]
	model = modelData[1]

	time = selectData(GRBName, directory, logCenter = True)
	timeRaw = selectData(GRBName, directory, logCenter = False)

	flux = selectData(GRBName, directory, column = 'flux')

	timeErr = selectData(GRBName, directory, column = 'timeErr', logCenter = True)
	fluxErr = selectData(GRBName, directory, column = 'fluxErr')
	fluxErrAsym = selectData(GRBName, directory, column = 'fluxErrAsym')

	fig = plt.figure(figsize = figsize)

	dchisq = calcChiSq(GRBName = GRBName, function = function, p0 = p0, start = start, end = end)
	popt = fit(function = function, data = data, start = start, end = end, p0 = p0)

	if function == powerLaw:
		title = ("Light curve for " + GRBName + ' and best fitting model' + '\n' + r'$\alpha_1 $ = ' + str(round((-1 * popt[1]),4)) + 
		'\n' + r'$t_{start}$ = ' + str(start) + r', $t_{end}$ = ' + str(end))
	elif function == brokenPowerLaw:
		title = ('Light curve for ' + GRBName + ' and best fitting model ' + '\n' + r'$\alpha_1 $ = ' + str(round((-1 * popt[2]),4))+ r', $\alpha_2 $ = ' 
		+ str(round((-1 * popt[3]),4)) + r', $\chi ^2$ = ' + str(round(dchisq,4)) + '\n' + r'$t_{start}$ = ' + str(start) + r', $t_{end}$ = ' + str(end))
		plt.axvline(popt[1])
	elif len(p0) == 6:
		title = ('Light curve for ' + GRBName + ' and best fitting model ' + '\n' + r'$\alpha_1 $ = ' + str(round((popt[3]),4))+ r', $\alpha_2 $ = '
		 + str(round((popt[4]),4)) + '\n' + r'$\alpha_3 $ = ' + str(round((popt[5]),4))+ r', $\chi ^2$ = ' + str(round(dchisq,4)) + '\n'
		  + r'$t_{start}$ = ' + str(start) + r', $t_{end}$ = ' + str(end))
		plt.axvline(popt[1])
		plt.axvline(popt[2])

	plt.title(title)
	plt.xlabel('Time (s)')
	plt.ylabel('Flux (erg/cm$^2$/s)')
	plt.loglog(t, model, label = 'Model')
	plt.loglog(time, flux, 'o', color = 'red', label='Data')
	plt.errorbar(time, flux, yerr = fluxErr, xerr = timeErr, color = 'red')
	plt.legend(prop={'size': 16}, loc = legendLoc)
	plt.tight_layout()
	if not folder:
		plt.show()

#plot emcee
def plot_mc(sampler, GRBName='GRB070429B', GRBBaseDir='C:/Users/Administrator/Box/Research Stuff/XRT_Data/', GRBDir = 'control',
 fileType = '.txt', nameBool = True, start = 0, end = True, column = 'time', p0=(1e9,-1), function = powerLaw, 
 legendLoc = 'upper right', figsize = (8,6), GRBSaveDir = 'GRB_chi2_fits/emcee', saveFig = True, chisq_red = 'NaN'):
	samples = sampler.get_chain(flat=True, discard = 0)
	# theta_max  = samples[np.argmax(sampler.flatlnprobability)]

	x_model = np.logspace(np.log10(time[0]), np.log10(time[len(time) - 1]))
	best_fit_model = model(medians, xs=x_model)
	plt.figure(figsize=(10,7))
	plt.loglog(time,flux, 'o', color = 'red', label='Data')
	plt.errorbar(time, flux, yerr = fluxErr, xerr = timeErr, color = 'red')
	plt.loglog(x_model,best_fit_model,label='Highest Likelihood Model')
	if(model_type== 'single'):
		plt.title('Light curve for ' + GRBName + ' and best fitting model ' + '\n' + r'$\log$(amp) = ' + str(round(medians[0],3)) + r', $\alpha = $' + str(round(medians[1],3)) + r', $\chi ^2$= ' + str(round(chisq_red,3)))
	elif(model_type == 'double'):
		plt.title('Light curve for ' + GRBName + ' and best fitting model ' + '\n'  + r'$\log$(amp) = ' + str(round(medians[0], 3)) + r', $\log$(x_break) = ' + str(round(medians[3], 3)) + '\n' + r'$\alpha_1 $ = ' + str(round((medians[1]),3))+ r', $\alpha_2 $ = ' 
			+ str(round((medians[2]),3)) + r', $\chi ^2$ = ' + str(round(chisq_red,3)) )
		plt.axvline(x = 10 ** medians[3], linestyle = '--', color = 'black')
	elif(model_type == 'triple'):
		plt.title('Light curve for ' + GRBName + ' and best fitting model ' + '\n' + r'$\log$(amp) = ' + str(round(theta_max[0], 3)) + r', $\log$(break1) = ' + str(round(theta_max[4], 3)) + '\n' + r'$\log$(break2) = ' + str(round(theta_max[5], 3)) + ', ' + r'$\alpha_1 $ = ' + str(round((theta_max[1]),3))+ '\n' + r'$\alpha_2 $ = ' 
			+ str(round((theta_max[2]),3)) + r', $\alpha_3=$' + str(round(theta_max[3], 3)))
		plt.axvline(x = 10 ** theta_max[4], linestyle = '--', color = 'black')
		plt.axvline(x = 10 ** theta_max[5], linestyle = '--', color = 'black')
	if saveFig:
		saveLoc = GRBBaseDir + GRBSaveDir + '/' + model_type
		saveName = saveLoc + '/' + GRBName + 'plot.png'
		plt.savefig(saveName)
	if folder:
		plt.close()

#plot corner plots and walker plots
def plot_corner(samples, GRBName='GRB070429B', GRBBaseDir='C:/Users/Administrator/Box/Research Stuff/XRT_Data/', GRBDir = 'control',
 fileType = '.txt', nameBool = True, start = 0, end = True, column = 'time', p0=(1e9,-1), function = powerLaw, 
 legendLoc = 'upper right', figsize = (8,6), discard = 0, GRBSaveDir = 'GRB_chi2_fits/emcee', saveFig = True):
	if(model_type == 'single'):
		labels = ['log(amp)', 'alpha']
		fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84], discard = 0)#, range = [(-5e-7, 1e-6), (-5, 3)])
		if saveFig:
			saveLoc = GRBBaseDir + GRBSaveDir + '/' + model_type
			saveName = saveLoc + '/' + GRBName + 'corner.png'
			plt.savefig(saveName)
		if folder:
			plt.close()
		fig1, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
		samples1 = sampler.get_chain(flat=False, discard = discard)
	elif(model_type == 'double'):
		labels = ['log(amp)', 'alpha1', 'alpha2', 'log(break)']
		fig = corner.corner(samples, show_titles = True, labels = labels, plot_datapoints = True, quantiles = [0.16, 0.5, 0.84], discard= 0)
		if saveFig:
			saveLoc = GRBBaseDir + GRBSaveDir + '/' + model_type
			saveName = saveLoc + '/' + GRBName + 'corner.png'
			plt.savefig(saveName)
		if folder:
			plt.close()
		fig1, axes = plt.subplots(4, figsize = (10, 7), sharex = True)
		samples1 = sampler.get_chain(flat=False, discard = discard)
	elif(model_type == 'triple'):
		labels = ['log(amp)', 'alpha1', 'alpha2', 'alpha3', 'log(break1)', 'log(break2)']
		fig = corner.corner(samples, show_titles = True, labels = labels, plot_datapoints = True, quantiles = [0.16, 0.5, 0.84], discard = 0)
		if saveFig:
			saveLoc = GRBBaseDir + GRBSaveDir + '/' + model_type
			saveName = saveLoc + '/' + GRBName + 'corner.png'
			plt.savefig(saveName)
		if folder:
			plt.close()
		fig1, axes = plt.subplots(6, figsize = (10, 7), sharex = True)
		samples1 = sampler.get_chain(flat=False, discard = discard)
	for i in range(ndim):
	    ax = axes[i]
	    print(ax)
	    ax.plot(samples1[:, :, i], "k")
	    ax.set_xlim(0, len(samples1))
	    # ax.set_xlim(0,500) #can use these limits instead of the line above to see the first 500 steps for debugging purposes
	    ax.set_ylabel(labels[i])
	    ax.yaxis.set_label_coords(-0.1, 0.5)

	axes[-1].set_xlabel("step number")
	if saveFig:
		saveLoc = GRBBaseDir + GRBSaveDir + '/' + model_type
		saveName = saveLoc + '/' + GRBName + 'walkers.png'
		plt.savefig(saveName)
	if not folder:	
		fig.show()
		plt.show()
		input()
	plt.close()

#saves parameters in .csv to be easily plotted or referenced
def saveParams(theta_max, chisq_red, plus_uncertainties, minus_uncertainties):
	grb_list = []
	nameFound = False
	with open(params_file, 'r', newline='') as g:
		grbs = csv.reader(g)
		grb_list.extend(grbs)
	for line, row in enumerate(grb_list):
		if GRBName in row:
			if(model_type == 'single'):
				amp, alpha = medians
				amp_plus, alpha_plus = plus_uncertainties
				amp_minus, alpha_minus = minus_uncertainties
				line_to_override = {line:[GRBName, amp, amp_plus, amp_minus, alpha, alpha_plus, alpha_minus, chisq_red, row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23], row[24], row[25], row[26], row[27]]}
			elif(model_type == 'double'):
				amp, alpha1, alpha2, xb = medians
				amp_plus, alpha1_plus, alpha2_plus, xb_plus = plus_uncertainties
				amp_minus, alpha1_minus, alpha2_minus, xb_minus = minus_uncertainties
				line_to_override = {line:[GRBName, row[1], row[2], row[3], row[4], row[5], row[6], row[7], amp, amp_plus, amp_minus, alpha1, alpha1_plus, alpha1_minus, alpha2, alpha2_plus, alpha2_minus, xb, xb_plus, xb_minus, chisq_red, row[21], row[22], row[23], row[24], row[25], row[26], row[27]]}
			else:
				amp, alpha1, alpha2, alpha3, xb1, xb2 = medians
				line_to_override = {line:[GRBName, row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], amp, alpha1, alpha2, alpha3, xb1, xb2, chisq_red]}
			nameFound = True
			with open(params_file, 'w', newline='\n') as b:
				writer = csv.writer(b)
				for line, row in enumerate(grb_list):
					data = line_to_override.get(line, row)
					writer.writerow(data)
			break
	if not nameFound:
		print(('Adding %s to the csv file' % GRBName))
		with open(params_file, mode='a', newline='\n') as GRBfile:
			GRBwriter = csv.writer(GRBfile, delimiter = ',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
			if(model_type == 'single'):
				amp, alpha = theta_max
				GRBwriter.writerow([GRBName, amp, alpha, chisq_red, 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL'])
			elif(model_type == 'double'):
				amp, alpha1, alpha2, xb = theta_max
				GRBwriter.writerow([GRBName, 'NULL', 'NULL', 'NULL', amp, alpha1, alpha2, xb, chisq_red, 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL'])
			else:
				amp, alpha1, alpha2, alpha3, xb1, xb2 = theta_max
				GRBwriter.writerow([GRBName, 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', 'NULL', amp, alpha1, alpha2, alpha3, xb1, xb2, chisq_red])

#now call the necessary functions to fit, plot, and save based on the options at the top of the code
if not folder:
	rawData = readData(name = GRBName, directory = directory, fileType = file_type, nameBool = nameBool, logCenter = logCenter)
	time, flux, fluxErr, timeErr = getAllData(rawData, start = start, end = end)


	medians, sampler, pos, prob, state, ndim, chisq_red, plus_uncertainties, minus_uncertainties = createModel(GRBName = GRBName, GRBBaseDir = GRBBaseDir, GRBDir = GRBDir, fileType = file_type, 
		nameBool = nameBool, start = start, end = end, column = 'time', p0 = p0_cf, function = function, t = 'model', 
		fit_type = fit_type, model_type = model_type)
	plot_mc(sampler, GRBName = GRBName, GRBBaseDir = GRBBaseDir, GRBDir = GRBDir,
	 fileType = file_type, nameBool = nameBool, start = start, end = end, column = 'time', p0=p0_cf, function = function, 
	 legendLoc = legendLoc, figsize = figsize, saveFig = saveFig, chisq_red = chisq_red)
	samples = sampler.get_chain(flat=True, discard = discard)
	print(samples.shape)
	save_loc = os.path.join(GRBBaseDir, samples_dir, model_type, GRBName)
	save_loc = save_loc.replace('\\', '/')
	np.save(save_loc, samples)
	plot_corner(samples, GRBName = GRBName, GRBBaseDir = GRBBaseDir, GRBDir = GRBDir,
	 fileType = file_type, nameBool = nameBool, start = start, end = end, column = 'time', p0=p0_cf, function = powerLaw, 
	 legendLoc = legendLoc, figsize = figsize, discard = discard, saveFig = saveFig)
	if saveParam:
		saveParams(medians, chisq_red, plus_uncertainties, minus_uncertainties)
elif folder:
	files = ([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
	print(len(files))
	for k in np.linspace(0,(len(files) - 1), len(files)):
		fileName = directory + files[int(k)]
		file = fileName.replace(directory, '')
		GRBName = file.replace('.txt', '')
		print(GRBName)
		# if(GRBName == 'GRB101224A'): #uncomment these two lines and change GRBName to designate a start point in your fitting
		# 	go = True                  #in this example, we would only fit bursts after and including GRB101224A in the directory
		rawData = readData(name = GRBName, directory = directory, fileType = file_type, nameBool = nameBool, logCenter = logCenter)
		if(not (rawData.shape == ()) and go):
			time, flux, fluxErr, timeErr = getAllData(rawData, start = start, end = end)
			if((model_type == 'double' and len(time) > 3) or (model_type == 'triple' and len(time) > 5) or (model_type == 'single' and len(time) > 1)):
				medians, sampler, pos, prob, state, ndim, chisq_red, plus_uncertainties, minus_uncertainties = createModel(GRBName = GRBName, GRBBaseDir = GRBBaseDir, GRBDir = GRBDir, fileType = file_type, 
				 nameBool = nameBool, start = start, end = end, column = 'time', p0 = p0_cf, function = function, t = 'model', 
				 fit_type = fit_type, model_type = model_type)
				plot_mc(sampler, GRBName = GRBName, GRBBaseDir = GRBBaseDir, GRBDir = GRBDir,
				 fileType = file_type, nameBool = nameBool, start = start, end = end, column = 'time', p0=p0_cf, function = function, 
			 	 legendLoc = legendLoc, figsize = figsize, saveFig = saveFig, chisq_red = chisq_red)
				samples = sampler.get_chain(flat=True, discard = discard)
				save_loc = os.path.join(GRBBaseDir, samples_dir, model_type, GRBName)
				save_loc = save_loc.replace('\\', '/')
				print(samples.shape)
				np.save(save_loc, samples)
				plot_corner(samples, GRBName = GRBName, GRBBaseDir = GRBBaseDir, GRBDir = GRBDir,
				 fileType = file_type, nameBool = nameBool, start = start, end = end, column = 'time', p0=p0_cf, function = powerLaw, 
			 	 legendLoc = legendLoc, figsize = figsize, discard = discard, saveFig = saveFig)
				if saveParam:
					saveParams(medians, chisq_red,plus_uncertainties, minus_uncertainties)
# plot(GRBName = 'GRB051210',  function = tbpl_pw, p0=(1e9, 150, 350, -0.6, -2.5, -5.6), end = 560)
#print(logCenterTime(559.208, 755.462, -94.512))
# print(calcChiSq(GRBName = 'GRB051210', function = tbpl_pw, p0=(1e9, 150, 350, -0.6, -2.5, -5.6), end = 560))
