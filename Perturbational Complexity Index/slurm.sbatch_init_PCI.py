#!/bin/bash
#SBATCH --job-name=PCI
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=
#SBATCH --array=1-100
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=PCI%A_%a.out
#SBATCH --error=PCI%A_%a.err
#Load Python 3.6.4
ml Python
python <<-EOF

import os
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat   # to load mat files
import PCI_utils as pci_u
#Code for calculating Jrank in SLURM Cluster
s = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
#Import data
we=np.arange(0,2.525,0.025)

for j,t in enumerate(we):
    we = j+1; 
    path = "data/PerturbPCI_"+"%s_"%we+"%s.mat"%s #import the perturbation
        
    ts = loadmat(path)['neuro_act']
    ts= ts.reshape(1,2001,90)
    ts= np.transpose(ts, (0, 2, 1))
    
    #Inizialization parameters will depend on the perturbation performed
    times=np.arange(-1000,1001,1)
    timepoint_perturb=999;
    N_bootstrap = 500;
    alpha = 0.01;
    
    #Preprocessing
    signal_centre_norm, signalThresh, signal_binary= pci_u.preprocess_bootstrap(ts,timepoint_perturb,N_bootstrap,alpha)
    
    signal_binary = signal_binary.mean(axis=0);  
    binJrank =  pci_u.sort_binJ(signal_binary)

    mdic = {"binJrank": binJrank, "label": "binJrank_values"}
    savemat('Jrank/Jrank_'+str(we)+'_'+str(s)+'.mat', mdic)

EOF
