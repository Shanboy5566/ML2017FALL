# coding: utf-8
import sys, time
import numpy as np
import csv

import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Flatten
# from keras.layers import normalization, Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import normalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(prefix):
    with open(prefix) as fp:
        x_test = []
        next(fp)
        for line in fp:
            line = line.strip('\n').split(',')
            x_test.append(line[1].split(' '))
    
    x_test = np.array(x_test, dtype=float)
    
    return x_test


def outputResult(prefix, Y_test):
    with open(prefix, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(('id', 'label'))
        for i in range(len(Y_test)):
            writer.writerow((i, Y_test[i]))

def Normalize(X_train):
    return (X_train / 255) * 2 - 1

if __name__ == '__main__':
    argv_1 = sys.argv[1] # location of testing data
    argv_2 = sys.argv[2] # location of predict file
    model = sys.argv[3] # location of model
    batch_size = 256
    nb_classes = 7
    nb_epoch = 150

    # loading testing data
    X_test = load_data(argv_1)
    
    # normalizing testing data, by loading the means_and_stds.csv
    X_test = Normalize(X_test)
    
    # resize X_test
    X_test = np.resize(X_test, (X_test.shape[0], 48, 48, 1))
    
    # loadind model
    model = load_model(model)
    
    # predict answers
    Y_proba = model.predict(X_test)
    Y_test = Y_proba.argmax(axis=-1)
    
    # save result
    outputResult(argv_2, Y_test)