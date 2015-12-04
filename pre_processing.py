# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:37:34 2015

@author: zc
"""
import glob
import  numpy as np
from skimage.io import imread
#from skimage.viewer import ImageViewer
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage import filter
from skimage.feature import hog
from skimage.filter import threshold_otsu
from skimage.morphology import closing, square
from sklearn import svm
import pickle
from sklearn.externals import joblib
import random
from sklearn.metrics import confusion_matrix

#class 1..62
#def readData(data_set,kernel):
#    print( "{0},{1}",(data_set,kernel))
#    print data_set,kernel
data_set=4
kernel='rbf'
print data_set,kernel
class_set1=[1,2,3,4,5,6,7,8,9,10]
class_set2=range(1,37)
class_set3=range(36,62)
class_set4=range(1,63)
if data_set==1:
    class_set=class_set1
elif data_set==2:
    class_set=class_set2
elif data_set==3:
    class_set=class_set3
elif data_set==4:
    class_set=class_set4
elif data_set==5:
    class_set=[7,7]
else:
    data_set=1

train_size=20

x_test=[]
y_verify=[]
for classNum in class_set:
    x_train=[]
    y_train=[]
    path='./Bmp/Sample0%.2d/*' % classNum
    class_files=glob.glob(path)
    file_num=len(class_files)
    all_file=range(file_num)
    train_file=random.sample(all_file,train_size)
    test_file=list(set(all_file)-set(train_file))
#    train_num=int(file_num*split_ratio)
#    test_num=file_num-train_num
    for i in all_file:
        img=imread(class_files[i],as_grey=True)
        img_resize=resize(img,[20,20])
        img_denoise = filter.denoise_tv_chambolle(img_resize, weight=0.01)
        thresh=threshold_otsu(img_denoise)
#        img_otsu=closing(img_denoise > thresh, square(2))
        img_otsu=img_denoise>thresh
        img_feature,img_hog=hog(img_denoise, orientations=8,
                pixels_per_cell=(5, 5),
                cells_per_block=(1, 1), visualise=True)
        x_train.append(img_feature)        
#        x_train.append(np.reshape(img_otsu,[1,400])[0])
        if classNum==24:
            y_train.append(0)
        elif classNum==50:
            y_train.append(0)
        else:
            y_train.append(classNum)
    x_pickle_filename='./data/Sample0%.2dX.p' % classNum
    pickle.dump(x_train,open(x_pickle_filename,'w'))
    y_pickle_filename='./data/Sample0%.2dY.p' % classNum
    pickle.dump(y_train,open(y_pickle_filename,'w'))
        
    
    print 'class %.2d done' % classNum
        