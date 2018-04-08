# -*- coding: utf-8 -*-
"""
Christina Indudhara
Description: This program analyzes handwritten symbols and uses machine learning to predict target values.
"""

from scipy.misc import imread 
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from sklearn import svm


""" This function creates the boundaries for each image plot
"""
def boundaries(binarized,axis):
    # variables named assuming axis = 0; algorithm valid for axis=1
    # [1,0][axis] effectively swaps axes for summing
    rows = np.sum(binarized,axis = [1,0][axis]) > 0
    rows[1:] = np.logical_xor(rows[1:], rows[:-1])
    change = np.nonzero(rows)[0]
    ymin = change[::2]
    ymax = change[1::2]
    height = ymax-ymin
    too_small = 10 # real letters will be bigger than 10px by 10px
    ymin = ymin[height>too_small]
    ymax = ymax[height>too_small]
    return zip(ymin,ymax)

""" This function separates a column of handwritten symbols into individual images for each symbol
"""
def separate(img):
    orig_img = img.copy()
    pure_white = 255.
    white = np.max(img)
    black = np.min(img)
    thresh = (white+black)/2.0
    binarized = img<thresh
    row_bounds = boundaries(binarized, axis = 0) 
    cropped = []
    for r1,r2 in row_bounds:
        img = binarized[r1:r2,:]
        col_bounds = boundaries(img,axis=1)
        rects = [r1,r2,col_bounds[0][0],col_bounds[0][1]]
        cropped.append(np.array(orig_img[rects[0]:rects[1],rects[2]:rects[3]]/pure_white))
    return cropped

""" This function resizes the imgs to the specified dimensions 
"""
def resize_all(imgs, dim):
    imgs_resized = []
    for img in imgs:  
        img = resize(img, dim)
        imgs_resized.append(img)
    return imgs_resized

""" This function creates the data and target datasets to be used by the svc classifier
"""
def load_data(datasets):
    data = None
    for filename,target_val in datasets:
        az = imread(filename,True)
        imgs = separate(az)
        resized = resize_all(imgs,(10,10))
        resized = np.asarray(resized)  # must convert resized images into a numpy array; it was previously a list
        n_samples = len(imgs)
        n_features = -1
        new_data = resized.reshape((n_samples, n_features))
        new_target = np.ones(new_data.shape[0])*target_val  
        # The above line yields a specified number of ones whereby the number of ones is determined by the number 
        # of rows in new_data), all of which are multiplied by the target value.
        if data is None:
            data = new_data
            target = new_target
        else:
            data = np.concatenate((data,new_data))
            target = np.concatenate((target,new_target))
    return data,target

""" This function partitions the data and target arrays into training and testing datasets
"""
def partition(data, target, p):
    # p is percentage of data to train on
    i = np.random.rand(data.shape[0])   # return np array, in the shape of the rows of data, of random numbers between 0 and 1 that are less than p
    train = i < p  # use train and test to subset data in the proper format
    test = np.logical_not(train)
    train_data, train_target = data[train,:], target[train] 
    test_data, test_target = data[test,:], target[test]
    return train_data, train_target, test_data, test_target


datasets = [("a.png", 0),("b.png", 1),("c.png",2)]

data,target = load_data(datasets)

p = 0.9  # proportion of dataset to be trained on

train_data, train_target, test_data, test_target = partition(data,target,p) 

#classifier = svm.SVC(gamma= .1) # nonlinear model
classifier = svm.LinearSVC()  # the linear system sometimes can be more advantageous to ovoid overfitting 
classifier.fit(train_data, train_target)

predicted = classifier.predict(test_data)

print "Predicted: ", predicted  # np array of class values 
print "Truth: ", test_target    # np array of class values
print "Accuracy: ", (1. - (1.*np.nonzero(predicted-test_target)[0].size)/test_target.size)*100, "%"

""" The accuracy is caluclated by subtracting the proporiton of incorrect values from 1.
    The proportion of incorrect values is calculated by dividing the number of incorrect values 
    by the total number of values in the target array """