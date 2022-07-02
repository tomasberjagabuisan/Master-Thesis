# -*- coding: utf-8 -*-
"""

@author: Tomas Berjaga Buisan

Improved, Adapted and modified from Kevin Ancourt

"""

import numpy as np

def preprocess_bootstrap(data,t_perturb,bootstraps,alpha):
    ''' extract and preprocess relevant data 
    '''  
    percentile = 100 - alpha                 
    nrepetitions,nodes,ntime = data.shape
    means_prestim = np.mean(data[:,:,:t_perturb], axis = 2) # prestim mean to 0
    signal_centralized =data / means_prestim[:,:,np.newaxis]
    std_prestim = np.std(signal_centralized[:,:,:t_perturb], axis = 2)
    
    # prestim std to 1
    signal_centre_norm = signal_centralized / std_prestim[:,:,np.newaxis]
    
    signalcn_tuple = tuple(signal_centre_norm)# not affected by shuffling    
    signal_prestim_shuffle = signal_centre_norm[:,:,:t_perturb]
    
    max_absval_shuffled = np.zeros(bootstraps)
    
    for i_shuffle in range(bootstraps):
        for i_nodes in range(nodes):
            for i_repetition in range(nrepetitions):
                signal_curr = signal_prestim_shuffle[i_repetition, i_nodes]
                np.random.shuffle(signal_curr)
                signal_prestim_shuffle[i_repetition, i_nodes] = signal_curr
                                
                #average over trials
                shuffle_avg = np.mean(signal_prestim_shuffle, axis = 0)
                max_absval_shuffled[i_shuffle] = np.max(np.abs(shuffle_avg))

#%% estimate significance threshold
    max_sorted = np.sort(max_absval_shuffled)
    signalThresh = max_sorted[-int(bootstraps/percentile)]
    
#%% binarise 
    signalcn = np.array(signalcn_tuple)
    signal_binary = np.abs(signalcn) > signalThresh
          
    return signal_centre_norm, signalThresh, signal_binary


def sort_binJ(binJ):
    ''' sort binJ as in Casali et al 2013
    '''
    SumCh=np.sum(binJ,axis=1)#axis=1;
    Irank=SumCh.argsort()
    return binJ[Irank,:]

