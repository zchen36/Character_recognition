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
#import pickle
from sklearn.externals import joblib
import random

#class 1..62
#def readData(data_set,kernel):
#    print( "{0},{1}",(data_set,kernel))
#    print data_set,kernel
data_set=1
kernel='rbf'
print data_set,kernel
class_set1=[1,2,3,4,5,6,7,8,9,10]
class_set2=range(1,36)
class_set3=range(36,62)
class_set4=range(1,62)
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

train_size=5
x_train=[]
y_train=[]
x_test=[]
y_verify=[]
for classNum in class_set:
    path='./Bmp/Sample0%.2d/*' % classNum
    class_files=glob.glob(path)
    file_num=len(class_files)
    all_file=range(file_num)
    train_file=random.sample(all_file,train_size)
    test_file=list(set(all_file)-set(train_file))
#    train_num=int(file_num*split_ratio)
#    test_num=file_num-train_num
    for i in train_file:
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
        y_train.append(classNum)
        
    for i in test_file:
        img=imread(class_files[i],as_grey=True)
        img_resize=resize(img,[20,20])
        img_denoise = filter.denoise_tv_chambolle(img_resize, weight=0.01)
        thresh=threshold_otsu(img_denoise)
#        img_otsu=closing(img_denoise > thresh, square(2))
        img_otsu=img_denoise>thresh
        img_feature,img_hog=hog(img_denoise, orientations=8,
                pixels_per_cell=(5, 5),
                cells_per_block=(1, 1), visualise=True)
        x_test.append(img_feature)
#        x_test.append(np.reshape(img_otsu,[1,400])[0])
        y_verify.append(classNum)
    
    print 'class %.2d done' % classNum
        
#        print class_files[i]
    if classNum==3:
        img=imread(class_files[21],as_grey=True)
        plt.subplot(2,3,2)
        plt.imshow(img,cmap="Greys")
        plt.title('greyscale')
        
        plt.subplot(2,3,1)
        plt.imshow(img)
        plt.title('original')
        
        img_resize=resize(img,[20,20])
        plt.subplot(2,3,3)
        plt.imshow(img_resize,cmap="Greys")
        plt.title('resized')
        
        img_denoise = filter.denoise_tv_chambolle(img_resize, weight=0.01)    
        plt.subplot(2,3,4)
        plt.imshow(img_denoise,cmap="Greys")
        plt.title('denoised')
        
        thresh=threshold_otsu(img_denoise)
    #    img_otsu=closing(img_denoise > thresh, square(2))
        img_otsu=img_denoise>thresh
        plt.subplot(2,3,5)
        plt.imshow(img_otsu,cmap="Greys")
        plt.title('highContrast')
        
        
        
        img_feature,img_hog=hog(img_denoise, orientations=8,
                    pixels_per_cell=(5, 5),
                    cells_per_block=(1, 1), visualise=True)
        plt.subplot(2,3,6)
        plt.imshow(img_hog,cmap="Greys")
        plt.title('HOG')
    
    
#    print train_num,test_num,file_num

#    clf = svm.SVC(kernel='sigmoid',gamma=0.01, C=100)
clf = svm.SVC(kernel='poly',degree=2)
#clf = svm.SVC(kernel='rbf',gamma=0.3)
#clf = svm.SVC(kernel='linear') 
#clf=svm.LinearSVC(C=1)
if kernel=='sig':
    clf = svm.SVC(kernel='sigmoid',gamma=0.01, C=100,coef0=0)
elif kernel=='poly':
    clf = svm.SVC(kernel='poly',gamma=0.01,degree=2,coef0=1)
elif kernel=='rbf':
    clf = svm.SVC(kernel='rbf',gamma=0.35,C=1000)
#    clf=svm.LinearSVC(C=100)
elif kernel=='linear':
    clf = svm.SVC(kernel='linear')
else:
    clf = svm.SVC(kernel='sigmoid',gamma=0.01, coef0=0)
    
clf.fit(x_train,y_train)
joblib.dump(clf, 'svmClassifier.pkl')
#    clf = joblib.load('svmClassifier.pkl')
#y_test = clf.predict(x_test)
#y_diff=y_test-y_verify
#y_nonzero=np.count_nonzero(y_diff)
#accuracy=1.0-float(y_nonzero)/len(y_verify)
accuracy=clf.score(x_test,y_verify)
print accuracy

        
#readData()
    
