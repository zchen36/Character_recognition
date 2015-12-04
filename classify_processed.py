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
def classify(data_set=1,sampleSize=5):
    print data_set,sampleSize
    #data_set=2
    kernel='rbf'
    print data_set,kernel
    class_set1=[1,2,3,4,5,6,7,8,9,10]
    class_set2=range(11,37)
    class_set3=range(37,63)
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
    
    train_size=25
    x_train=[]
    y_train=[]
    x_test=[]
    y_verify=[]
    for classNum in class_set:
        Xpath='./data/Sample0%.2dX.p' % classNum
        Ypath='./data/Sample0%.2dY.p' % classNum
    #    class_files=glob.glob(path)
        X=pickle.load(open(Xpath,'r'))
        Y=pickle.load(open(Ypath,'r'))
        file_num=len(Y)
        all_file=range(file_num)
        train_file=random.sample(all_file,train_size)
        test_file=list(set(all_file)-set(train_file))
    
        for i in train_file:
            x_train.append(X[i])
            y_train.append(Y[i])
            
        for i in test_file:
            x_test.append(X[i])
            y_verify.append(Y[i])
        
        print 'class %.2d done' % classNum
            
    #        print class_files[i]
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
    y_test = clf.predict(x_test)
    #y_diff=y_test-y_verify
    #y_nonzero=np.count_nonzero(y_diff)
    #accuracy=1.0-float(y_nonzero)/len(y_verify)
    accuracy=clf.score(x_test,y_verify)
    print accuracy
    return accuracy
    
    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.bone):
        plt.figure()    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
    #    tick_marks = np.arange(len(iris.target_names))
    #    plt.xticks(tick_marks, iris.target_names, rotation=45)
    #    plt.yticks(tick_marks, iris.target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
    #cm = confusion_matrix(y_verify, y_test)
    #np.set_printoptions(precision=2)
    #print('Confusion matrix, without normalization')
    #print(cm)
    #plot_confusion_matrix(cm)
    
            
    #readData()
    
