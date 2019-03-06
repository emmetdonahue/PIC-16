# -*- coding: utf-8 -*-
"""
PIC 16
Startup code for homework 7
"""

from scipy.misc import imread # using scipy's imread
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from skimage.transform import resize
from random import shuffle
import warnings

warnings.filterwarnings('ignore')

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


def reshape_image(img):
    return np.reshape(img,len(img)*len(img[0]))

big_imgE = imread("letterE.png", flatten = True) # flatten = True converts to grayscale
big_imgA = imread("letterA.png", flatten = True) # flatten = True converts to grayscale
big_imgU = imread("letterU.png", flatten = True) # flatten = True converts to grayscale

small_E = separate(big_imgE) # separates big_img (pure white = 255) into array/
                             # of little images (pure white = 1.0)
small_A = separate(big_imgA)
small_U = separate(big_imgU)

all_small_imgs = [] # holds the small images


for img in small_E: # add E images to list
    img = resize(img, (10,10))
    img = np.reshape(img, -1)
    all_small_imgs.append(img)

for img in small_A: # add A images to list
    img = resize(img, (10,10))
    img = np.reshape(img, -1)
    all_small_imgs.append(img)

for img in small_U: # add U images to list
    img = resize(img, (10,10))
    img = np.reshape(img, -1)
    all_small_imgs.append(img)

#dataset = images of correct size
labels = []
for i in range(3): #0 corresponds to E, 1 to A, 2 to U
    for j in range(10):
        labels.append(i)

all_small_imgs = np.array(all_small_imgs) 

# train dataset to labels
# p is the percent of dataset that is used for training data, remaining data
# be used for test cases
# dataset is an np array of images
# labels are the target for the images in dataset
# print the results of the prediction 
def partition(dataset, labels, p):
    train_size = int((p/100.0) * len(dataset))
    data_and_labels = []
    
    for i in range(len(dataset)):
        data_and_labels.append([dataset[i], labels[i]])
            
    shuffle(data_and_labels)
    
    train_data = []
    train_target = []
    test_data = []
    test_target = []
    for i in range(len(data_and_labels)):
        if i < train_size:
            train_data.append(data_and_labels[i][0])
            train_target.append(data_and_labels[i][1])
        elif i >= train_size:
            test_data.append(data_and_labels[i][0])
            test_target.append(data_and_labels[i][1])

    train_data = np.array(train_data)
    train_target = np.array(train_target)  
    test_data = np.array(test_data)
    
    clf = svm.SVC(gamma = 0.1)
    clf.fit(train_data, train_target)
    
    prediction = clf.predict(test_data)
    
    print 'Predicted:', prediction
    print 'Target:   ', np.array(test_target)
    
    correct = 0.0
    for i in range(len(test_target)):
        if test_target[i] == prediction[i]:
            correct += 1.0
                        
    accuracy =  (correct/len(test_data))*100
    print 'Accuracy:', accuracy, '%'
    
    return train_data, train_target, test_data, test_target

partition(all_small_imgs, labels, 20)
partition(all_small_imgs, labels, 30)
partition(all_small_imgs, labels, 40)
partition(all_small_imgs, labels, 50)
partition(all_small_imgs, labels, 60)
partition(all_small_imgs, labels, 70)
partition(all_small_imgs, labels, 80)
