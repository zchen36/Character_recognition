

#dataset on google drive: https://drive.google.com/open?id=0BxPwNcmaNmLtNm5xRC1JVUhfQjA

#NCSU github: https://github.ncsu.edu/zchen15/CSC522_char_recognition.git

# pakcages needed:

* import glob
* import  numpy as np
* from skimage.io import imread
* from matplotlib import pyplot as plt
* from skimage.transform import resize
* from skimage import filter
* from skimage.feature import hog
* from skimage.filter import threshold_otsu
* from skimage.morphology import closing, square
* from sklearn import svm
* import pickle
* from sklearn.externals import joblib
* import random

# files
- pre-processed.py: script which does image pre-process, and save data to ./data
- read_data.py: functions to test different kernels
- JustClassify.py: script which calls functions in read_data.py, and generate results for different kernels and datasets
- classify_processed.py: function to train classifier with different sample size
- clsssify_size.py: script which calls functions in classify_processed.py and generate results
- classifier_crossValidation.py: script to run k-fold CV
- ratio_cm_script.py: script to generate confusion matrixes

# user manual
1. download dataset from https://drive.google.com/open?id=0BxPwNcmaNmLtNm5xRC1JVUhfQjA, and save it in the same dir with the code
2. run pre-processed.py to process raw data
3. if you want to get results for k-fold CV, run classifier_crossValidation.py
4. if you want to get confusion matrixes, run ratio_cm_script.py
5. if you want to get results for different kernels, run JustClassify.py
