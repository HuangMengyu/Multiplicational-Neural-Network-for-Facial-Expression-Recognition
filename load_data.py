import os
import pickle 
import cv2
import numpy as np
import scipy.fftpack as FFT
import logging


np.set_printoptions(suppress=True)
def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min)
    return x

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = []
    for i in range(num_classes):
        if i==labels_dense:
            labels_one_hot.append(1)
        else:
            labels_one_hot.append(0)
    return labels_one_hot

def pickle_2_img_single(data_file):
     if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
     with open(data_file, 'rb') as f:
        data = pickle.load(f)
     total_x2, total_x1,  total_y = [], [], []

     for j in range(len(data['labels'])):
         img = data['img'][j]

         img =  FFT.fftn(img)
         img = FFT.fftshift(img)
         img1 = img.real
         img2 = img.imag

         label = int(data['labels'][j])

         label1 = dense_to_one_hot(label,6)

         total_x1.append(img1)
         total_x2.append(img2)
         total_y.append(label1)

     return total_x1, total_x2,total_y
