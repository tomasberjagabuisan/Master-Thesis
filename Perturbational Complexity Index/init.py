# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:51:39 2022

@author: tomas
"""

import numpy as np
from scipy.io import loadmat
from scipy.io import savemat   # to load mat files
import PCI_f as pci

#Import Jrank data calculated previously
we=np.arange(0,2.525,0.025)
subjects = np.arange(0,100)
PCI_val = np.zeros_like(we)
for s,h in enumerate(subjects):
 sub = s+1
 for j,t in enumerate(we):
    perturb = j+1;
    path = "Jrank/Jrank_"+"%s_"%perturb+"%s.mat"%sub
        
    binJrank = loadmat(path)['binJrank']
    #Parameters will depend on the perturbation performed
    timepoint_perturb=999;
    timepoints_PCI =301; 
    time_perturbation=10;
    ind_binJ=timepoint_perturb+time_perturbation+np.arange(timepoints_PCI)
    PCI_val[j] = pci.calculate(binJrank[:,ind_binJ])

 mdic = {"PCI": PCI_val, "label": "PCI"}
 savemat('PCI/PCI_'+str(sub)+'.mat', mdic)
