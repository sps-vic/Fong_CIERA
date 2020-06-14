import numpy as np
import matplotlib.pyplot as plt
import os, os.path

#names = ['GRB050509B', 'GRB051210', 'GRB051221A', 'GRB060121', 'GRB060313', 'GRB060801', 'GRB061006', 'GRB061201', 'GRB061210', 'GRB070429B']
alpha_total= []
alpha1_total = []
alpha2_total = []
logbreak_total = []
GRBBaseDir = 'C:/Users/Administrator/Box/Research Stuff/XRT_Data/' #Enter base directory
GRBDir = 'samples' #enter directory where .npy files are stored

###################################
model_type = 'double' #what model type do you want to plot (single, double)
parameter = 'alpha2' #what parameter to plot if using double  (alpha1, alpha2, break)
###################################

directory = os.path.join(GRBBaseDir, GRBDir, model_type)
directory = directory.replace('\\', '/')

files = ([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

for k in np.linspace(0,(len(files) - 1), len(files)):
    #get the entire filename with directory, then reduce it to just the name of the GRB
    file = os.path.join(directory, files[int(k)])
    file = file.replace('\\', '/')
    print(file)
    samples = np.load(file)
    samples = samples.T

    if(model_type == 'single'):
        logamp = samples[0]
        alpha = samples[1]
        for j in alpha:
            alpha_total.append(j)
    elif(model_type == 'double'):
        logamp = samples[0]
        alpha1 = samples[1]
        alpha2 = samples[2]
        logbreak = samples[3]
        if(parameter == 'alpha1'):    
            for j in alpha1:
                alpha1_total.append(j)
        elif(parameter == 'alpha2'):
            for j in alpha2:
                alpha2_total.append(j)
        elif(parameter == 'break'):
            for j in logbreak:
                if(not (10 ** j == 0)):
                    logbreak_total.append((10 ** j))

if(model_type == 'single'):
    print(len(alpha_total))
    plt.hist(alpha_total, bins = 20, density = True, stacked = True, histtype = 'step', linewidth = 1.5)
    plt.title("Alpha values for short Gamma-ray bursts fit by a single power law")
    plt.xlabel("Alpha")
    plt.show()
elif(model_type == 'double'):
    if(parameter == 'alpha1'):
        values, bins, __ = plt.hist(alpha1_total, bins = 20, density = True, stacked = True, histtype = 'step', linewidth = 1.5, log = False)
        plt.title("Alpha1 values for short Gamma-ray bursts fit by a double power law")
        plt.xlabel("Alpha1")
    elif(parameter == 'alpha2'):
        values, bins, __ = plt.hist(alpha2_total, bins = 20, density = True, stacked = True, histtype = 'step', linewidth = 1.5, log = False)
        plt.title("Alpha2 values for short Gamma-ray bursts fit by a double power law")
        plt.xlabel('Alpha2')
    elif(parameter == 'break'):
        values, bins, __ = plt.hist(logbreak_total, bins = 20, density = True, stacked = True, histtype = 'step', linewidth = 1.5, log = False)
        plt.title("log(break) values for short Gamma-ray bursts fit by a double power law")
        plt.xlabel('log(break)')
    area = sum(np.diff(bins) * values)
    print(area) #check that area under curve is 1
    plt.show()