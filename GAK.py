# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:31:54 2020

@author: BkTaha
"""
import numpy as np
import tensorflow as tf


def tf_cdist(s1, s2):
    M1 = s1.shape[0]
    M2 = s2.shape[0]
    p1 = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.square(s1), 1), 1),
                       tf.ones(shape=(1, M2), dtype=tf.float64))
    p2 = tf.transpose(tf.matmul(
        tf.reshape(tf.reduce_sum(tf.square(s2), 1), shape=[-1, 1]),
        tf.ones(shape=(M1, 1), dtype=tf.float64),
        transpose_b=True
        ))
    
    res = tf.add(p1, p2) - 2 * tf.matmul(s1, s2, transpose_b=True)
    return res
    

def tf_gak(S1, S2, gamma, path_limit=np.inf, random_kill=5):
    assert S1.shape==S2.shape, "GAK input shapes  mismatch"
    assert len(S1.shape) > 2, "GAK input's shape error"
    kill = lambda: np.random.choice([True, False], 1, p=[1/random_kill, 1-(1/random_kill)])[0]
    gak_dist_list = []
    for s_ind in range(S1.shape[0]):
        s1 = S1[s_ind:s_ind+1]
        s2 = S2[s_ind:s_ind+1]
        if len(s1.shape)>2:
            s1 = tf.reshape(s1, s1.shape[-2:])
            s2 = tf.reshape(s2, s2.shape[-2:])
            
        M1 = s1.shape[0]
        kga_gram = tf.exp(- tf.divide(tf_cdist(s1, s2),gamma)) 
        gak_dist = {}
        for i in range(M1):
            for j in range(M1):
                gak_dist[(i,j)] = 0
        gak_dist[(0, 0)] = kga_gram[0, 0]
        
        for i in range(1, M1):
            gak_dist[(0, i)] = tf.multiply(kga_gram[0, i], gak_dist[(0, i-1)])
            gak_dist[(i, 0)] = tf.multiply(kga_gram[i, 0], gak_dist[(i-1, 0)])
            
        for i in range(1, M1):
            for j in range(1, M1):
                if np.abs(i-j) > path_limit:
                    gak_dist[(i, j)] = 0
                elif kill():
                    gak_dist[(i, j)] = 0
                else:
                    gak_dist[(i, j)] = tf.multiply(kga_gram[i, j], 
                                                   tf.reduce_sum(tf.convert_to_tensor([gak_dist[(i, j-1)],
                                                                  gak_dist[(i-1, j)],
                                                                  gak_dist[(i-1, j-1)]], dtype=tf.float64)))
                    
        gak_dist_list.append(tf.math.log(tf.convert_to_tensor(gak_dist[M1-1, M1-1], dtype=tf.float64)))
    return gak_dist_list



