# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 09:58:45 2020

@author: BkTaha
"""

import json
import pickle as pkl
import numpy as np
from sklearn.metrics import pairwise_distances

def get_hyparams(dataset_name):
    json_param = "datasets_parameters.json"
    with open(json_param) as jf:
        info = json.load(jf)
        d = info[dataset_name]
        path = d['path']
                
    X_train, y_train, X_test, y_test = pkl.load(open(path, 'rb')) 
    #Using median
    D = pairwise_distances(X_train[:,0,:,0], X_train[:,0,:,0], metric='seuclidean')
    D_T = []
    for i in range(D.shape[0]):
        for j in range(i+1, D.shape[1]):
            D_T.append(D[i,j])
    gamma = np.median(D_T)
    lbda = 10**-(np.ceil(np.log(gamma)/np.log(10))-2)
    return gamma, lbda
    