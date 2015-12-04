# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:04:04 2015

@author: zc
"""

import classify_processed
import  numpy as np


accu_matrix=np.ones([4,4])
dataset=[1,2,3,4]
size=[5,10,15,20]
repeat=range(2)
for i in range(len(dataset)):#dataset
    for j in range(len(size)):#size
        temp=0
        for k in repeat:
            temp=temp+classify_processed.classify(dataset[i],size[j])
        temp=temp/len(repeat)
        accu_matrix[i][j]=temp