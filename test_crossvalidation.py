# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:34:49 2015

@author: zc
"""

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
import random

iris = datasets.load_iris()
clf = svm.SVC(kernel='linear', C=1)
a=iris.data
b=iris.target
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=2)
print scores

a=range(1,101)
b=random.sample(a,5)
c=list(set(a)-set(b))
print b