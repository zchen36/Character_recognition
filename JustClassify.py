# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:44:11 2015

@author: zc
"""

from sklearn.externals import joblib
#
#clf = joblib.load('svmClassifier.pkl') 

import read_data
import  numpy as np

a=read_data.readData(1,'rbf');
print a

#data_set=[1,2,3,4]
#kernel=['sig','rbf','poly','linear']
#
#accu_matrix=np.ones([4,4])
#for i in range(0,4):
#    for j in range(0,4):
#        accu_matrix[i,j]=read_data.readData(data_set[i],kernel[j])
#        print data_set[i],kernel[j],accu_matrix[i,j]
        
#      ([[ 0.8590604 ,  0.8590604 ,  0.17785235,  0.8590604 ],
#       [ 0.8283611 ,  0.8407594 ,  0.51452925,  0.82874855],
#       [ 0.72736626,  0.71707819,  0.16049383,  0.72530864],
#       [ 0.68787328,  0.69047001,  0.34484549,  0.687873        ]])
